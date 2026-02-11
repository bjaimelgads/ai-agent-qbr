"""
QBR Document Processor

Complete pipeline for:
1. Extracting content from PPTX using Kreuzberg
2. Enhancing with LLM using DSPY
3. Storing in SQLAlchemy database
"""

import base64
import json
import os
import re
import zipfile
import time
from dataclasses import replace
from datetime import datetime
from enum import Enum
from pathlib import Path
import xml.etree.ElementTree as ET

import kreuzberg
from kreuzberg import (
    ChunkingConfig,
    ExtractionConfig,
    ExtractionResult,
    ImageExtractionConfig,
    KeywordAlgorithm,
    KeywordConfig,
    LanguageDetectionConfig,
    OcrConfig,
    PageConfig,
)
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from qbr_intelligence.db.models import (
    Base,
    Chart,
    ChartType,
    Chunk,
    Client,
    Document,
    DocumentStatus,
    Entity,
    Image,
    Keyword,
    Metric,
    MetricAlias,
    MetricCatalog,
    Region,
    RegionCountry,
    Period,
    Slide,
    SlideType,
)
from qbr_intelligence.llm.modules import (
    MetricDeduplicator,
    MetricReviewer,
    QBREnhancementPipeline,
    ExecutiveSummarizer,
    create_lm,
    MetricContextExtractor,
)
from qbr_intelligence.pipeline.embeddings import EmbeddingSettings
from qbr_intelligence.pipeline.metric_context import (
    DocumentContext,
    HybridMetricContextStrategy,
    LLMMetricsContextStrategy,
    RuleBasedMetricContextStrategy,
    apply_context_strategies,
    infer_baseline_type,
)
from qbr_intelligence.pipeline.metric_scanner import (
    MetricCandidate,
    MetricScanArtifacts,
    MetricScanner,
    _metric_type_from_unit,
    _new_metric_id,
    build_metric_dictionary,
)
from qbr_intelligence.metrics.adapters import to_metric_candidate
from qbr_intelligence.metrics.catalog import build_metric_catalog, build_metric_catalog_from_db
from qbr_intelligence.metrics.pipeline import MetricExtractionPipeline, PipelineConfig
from qbr_intelligence.metrics.parsing import parse_pptx_deck, compute_deck_hash
from qbr_intelligence.metrics.adjudicator import LLMAdjudicator
from qbr_intelligence.metrics.regions import (
    EMEA_COUNTRIES,
    REGION_EMEA,
    REGION_US,
    infer_country_from_text,
    infer_region_from_text,
)
from qbr_intelligence.pipeline.post_embeddings import (
    PostEmbeddingSettings,
    apply_post_embeddings,
)
from qbr_intelligence.infrastructure.google_slides import GoogleSlidesClient


class QBRProcessor:
    """
    Complete QBR document processing pipeline.

    Handles extraction, LLM enhancement, and database storage.
    """

    CHUNK_MAX_CHARS = 1500
    CHUNK_OVERLAP = 200
    DOCUMENT_URLS_FILENAME = "document_urls.json"

    def __init__(
        self,
        database_url: str = "sqlite:///qbr_intelligence.db",
        output_dir: Path | str = "extraction_output",
        llm_model: str = "openai/gpt-4o-mini",
        enable_ocr: bool = False,
        extract_images: bool = False,
        embedding_settings: EmbeddingSettings | None = None,
    ):
        """
        Initialize the processor.

        Args:
            database_url: SQLAlchemy database URL
            output_dir: Directory for extracted images
            llm_model: LLM model to use for enhancement (LiteLLM picks up API keys from env)
            enable_ocr: Whether to run OCR on images (slow, ~5min for large docs)
            extract_images: Whether to extract images from document
            embedding_settings: Optional embedding settings override (defaults to env)
        """
        self.database_url = database_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.enable_ocr = enable_ocr
        self.extract_images = extract_images
        self.embedding_settings = embedding_settings or EmbeddingSettings.from_env()

        # Initialize database
        self.engine = create_engine(database_url, echo=False)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self._ensure_metric_columns()
        self._ensure_document_columns()
        self._ensure_slide_columns()
        self._seed_clients()
        self._client_names = self._load_client_names()
        self._seed_metric_catalog()
        self._seed_regions()
        self._region_map = self._load_region_map()
        self._metric_catalog_map = self._load_metric_catalog_map()

        # Initialize LLM (will be created when needed)
        self.llm_model = llm_model
        self._lm: "dspy.LM | None" = None

        # Enhancement pipeline (lazy loaded)
        self._enhancement_pipeline: QBREnhancementPipeline | None = None
        self._adjudicator_debug_count = 0

    @property
    def lm(self):
        """Get or create the LM instance."""
        if self._lm is None:
            self._lm = create_lm(
                model=self.llm_model,
                temperature=0.0,
            )
        return self._lm

    @property
    def enhancement_pipeline(self) -> QBREnhancementPipeline:
        """Get or create the enhancement pipeline."""
        if self._enhancement_pipeline is None:
            self._enhancement_pipeline = QBREnhancementPipeline(lm=self.lm)
        return self._enhancement_pipeline

    def _call_llm_text(self, prompt: str) -> str:
        """Best-effort text completion call for lightweight adjudication."""
        try:
            lm = self.lm
        except Exception:
            return ""
        debug_enabled = (os.getenv("METRIC_ADJUDICATOR_DEBUG") or "").lower() in {
            "1",
            "true",
            "yes",
        }
        def _extract_text(response) -> str:
            if isinstance(response, str):
                return response
            if isinstance(response, (list, tuple)) and response:
                if isinstance(response[0], str):
                    return response[0]
            if isinstance(response, dict):
                choices = response.get("choices")
                if choices:
                    choice = choices[0]
                    if isinstance(choice, dict):
                        message = choice.get("message") or {}
                        if isinstance(message, dict) and message.get("content"):
                            return message["content"]
                        if choice.get("text"):
                            return choice["text"]
                if response.get("content"):
                    return response["content"]
            choices = getattr(response, "choices", None)
            if choices:
                choice = choices[0]
                message = getattr(choice, "message", None)
                if message is not None:
                    content = getattr(message, "content", None)
                    if isinstance(content, str):
                        return content
                text = getattr(choice, "text", None)
                if isinstance(text, str):
                    return text
            for attr in ("text", "completion", "content", "output_text"):
                value = getattr(response, attr, None)
                if isinstance(value, str) and value.strip():
                    return value
            return ""
        for method_name in ("request", "complete", "__call__"):
            method = getattr(lm, method_name, None)
            if not callable(method):
                continue
            try:
                response = method(prompt)
            except Exception:
                continue
            if debug_enabled and self._adjudicator_debug_count < 3:
                summary = f"type={type(response).__name__}"
                if isinstance(response, dict):
                    summary += f" keys={list(response.keys())}"
                print(f"  [Metric Adjudicator Debug] response {summary}")
                self._adjudicator_debug_count += 1
            text = _extract_text(response)
            if text:
                return text
            try:
                return str(response)
            except Exception:
                continue
        return ""

    def _ensure_metric_columns(self) -> None:
        if self.engine.dialect.name != "sqlite":
            return
        inspector = inspect(self.engine)
        if "metrics" not in inspector.get_table_names():
            return
        existing = {col["name"] for col in inspector.get_columns("metrics")}
        needed = {
            "metric_catalog_id": "INTEGER",
            "baseline_text": "TEXT",
            "baseline_type": "TEXT",
            "country": "TEXT",
        }
        missing = {name: ddl for name, ddl in needed.items() if name not in existing}
        if not missing:
            return
        with self.engine.begin() as conn:
            for name, ddl in missing.items():
                conn.execute(text(f"ALTER TABLE metrics ADD COLUMN {name} {ddl}"))

    def _ensure_document_columns(self) -> None:
        if self.engine.dialect.name != "sqlite":
            return
        inspector = inspect(self.engine)
        if "documents" not in inspector.get_table_names():
            return
        existing = {col["name"] for col in inspector.get_columns("documents")}
        needed = {
            "half": "TEXT",
            "client_id": "INTEGER",
            "region_id": "INTEGER",
            "report_period_id": "INTEGER",
        }
        missing = {name: ddl for name, ddl in needed.items() if name not in existing}
        with self.engine.begin() as conn:
            for name, ddl in missing.items():
                conn.execute(text(f"ALTER TABLE documents ADD COLUMN {name} {ddl}"))
            refreshed = inspect(self.engine)
            doc_cols = {col["name"] for col in refreshed.get_columns("documents")}
            period_cols = (
                {col["name"] for col in refreshed.get_columns("periods")}
                if "periods" in refreshed.get_table_names()
                else set()
            )
            if (
                "report_period" in doc_cols
                and "report_period_id" in doc_cols
                and {"id", "period_label"}.issubset(period_cols)
            ):
                conn.execute(
                    text(
                        """
                        INSERT INTO periods (period_label, period_type, period_number, fiscal_year, start_date, end_date)
                        SELECT DISTINCT TRIM(d.report_period), NULL, NULL, NULL, NULL, NULL
                        FROM documents d
                        LEFT JOIN periods p
                          ON UPPER(TRIM(p.period_label)) = UPPER(TRIM(d.report_period))
                        WHERE d.report_period IS NOT NULL
                          AND TRIM(d.report_period) <> ''
                          AND p.id IS NULL
                        """
                    )
                )
                conn.execute(
                    text(
                        """
                        UPDATE documents
                        SET report_period_id = (
                            SELECT p.id
                            FROM periods p
                            WHERE UPPER(TRIM(p.period_label)) = UPPER(TRIM(documents.report_period))
                            ORDER BY p.id
                            LIMIT 1
                        )
                        WHERE report_period_id IS NULL
                          AND report_period IS NOT NULL
                          AND TRIM(report_period) <> ''
                        """
                    )
                )

    def _ensure_slide_columns(self) -> None:
        if self.engine.dialect.name != "sqlite":
            return
        inspector = inspect(self.engine)
        if "slides" not in inspector.get_table_names():
            return
        existing = {col["name"] for col in inspector.get_columns("slides")}
        needed = {
            "google_slide_id": "TEXT",
        }
        missing = {name: ddl for name, ddl in needed.items() if name not in existing}
        if not missing:
            return
        with self.engine.begin() as conn:
            for name, ddl in missing.items():
                conn.execute(text(f"ALTER TABLE slides ADD COLUMN {name} {ddl}"))

    @staticmethod
    def _extract_google_presentation_id(value: str | None) -> str | None:
        if not value:
            return None
        match = re.search(
            r"https?://docs\.google\.com/presentation/d/([a-zA-Z0-9_-]+)",
            value,
        )
        return match.group(1) if match else None

    def _resolve_document_url(self, file_path: Path) -> str:
        mapping_path = file_path.parent / self.DOCUMENT_URLS_FILENAME
        if not mapping_path.exists():
            return str(file_path.absolute())
        try:
            payload = json.loads(mapping_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  [Document URL] Failed to read {mapping_path}: {exc}")
            return str(file_path.absolute())
        if not isinstance(payload, dict):
            print(f"  [Document URL] Invalid mapping in {mapping_path}; expected JSON object.")
            return str(file_path.absolute())
        url = payload.get(file_path.name)
        if isinstance(url, str) and url.strip():
            return url.strip()
        return str(file_path.absolute())

    def _resolve_google_presentation_id(
        self,
        *,
        file_path: Path,
        result: ExtractionResult | None = None,
    ) -> str | None:
        env_value = (os.getenv("GOOGLE_SLIDES_PRESENTATION_ID") or "").strip()
        if env_value:
            return env_value
        document_url = self._resolve_document_url(file_path)
        candidates: list[str] = [document_url, str(file_path)]
        if result and result.metadata:
            source_url = result.metadata.get("source_url")
            if isinstance(source_url, str):
                candidates.append(source_url)
        for candidate in candidates:
            presentation_id = self._extract_google_presentation_id(candidate)
            if presentation_id:
                return presentation_id
        return None

    def _attach_google_slide_ids(
        self,
        *,
        slides_ordered: list[dict],
        slides_raw: list[dict],
        remap: list[dict] | None,
        file_path: Path,
        result: ExtractionResult,
    ) -> None:
        presentation_id = self._resolve_google_presentation_id(
            file_path=file_path,
            result=result,
        )
        if not presentation_id:
            return
        client = GoogleSlidesClient.from_env(base_dir=self.output_dir)
        if client is None:
            return
        try:
            slide_ids = client.list_slide_ids(presentation_id)
        except Exception as exc:
            print(f"[Google Slides] Failed to fetch slide IDs: {exc}")
            return
        if not slide_ids:
            return
        ids_by_number = {idx + 1: slide_id for idx, slide_id in enumerate(slide_ids)}

        for slide in slides_ordered:
            slide_num = slide.get("slide_number")
            if isinstance(slide_num, int):
                slide["google_slide_id"] = ids_by_number.get(slide_num)

        if remap:
            slides_by_num = {slide.get("slide_number"): slide for slide in slides_raw}
            for entry in remap:
                source_num = entry.get("source_page_number")
                new_num = entry.get("slide_number")
                if not isinstance(source_num, int) or not isinstance(new_num, int):
                    continue
                target = slides_by_num.get(source_num)
                if target is not None:
                    target["google_slide_id"] = ids_by_number.get(new_num)
        else:
            for slide in slides_raw:
                slide_num = slide.get("slide_number")
                if isinstance(slide_num, int):
                    slide["google_slide_id"] = ids_by_number.get(slide_num)

    def create_extraction_config(self) -> ExtractionConfig:
        """Create Kreuzberg extraction configuration."""
        embedding_config = self.embedding_settings.to_embedding_config()
        return ExtractionConfig(
            enable_quality_processing=True,
            force_ocr=False,
            ocr=OcrConfig(backend="tesseract", language="eng"),
            images=ImageExtractionConfig(
                extract_images=True,
                target_dpi=150,
                max_image_dimension=2048,
            ),
            pages=PageConfig(
                extract_pages=True,
                insert_page_markers=True,
            ),
            keywords=KeywordConfig(
                algorithm=KeywordAlgorithm.Yake,
                max_keywords=30,
                min_score=0.01,
                ngram_range=(1, 3),
                language="en",
            ),
            language_detection=LanguageDetectionConfig(
                enabled=True,
                detect_multiple=True,
                min_confidence=0.5,
            ),
            chunking=ChunkingConfig(
                max_chars=1500,
                max_overlap=200,
                embedding=embedding_config,
            ),
        )

    def extract_document(self, file_path: str | Path) -> ExtractionResult:
        """Extract content from document using Kreuzberg."""
        file_path = Path(file_path)
        config = self.create_extraction_config()

        print(f"[Extraction] Processing: {file_path.name}")
        result = kreuzberg.extract_file_sync(str(file_path), None, config)

        print(f"  - Pages: {result.get_page_count()}")
        print(f"  - Images: {len(result.images) if result.images else 0}")
        print(f"  - Chunks: {result.get_chunk_count()}")

        # Strip Notes blocks immediately after extraction to keep downstream metrics clean.
        try:
            result.content = self._strip_notes_blocks(result.content)
        except Exception:
            pass

        return result

    @staticmethod
    def _build_content_from_pages(pages: list[dict]) -> str:
        """Rebuild raw content from page blocks."""
        blocks: list[str] = []
        for page in pages:
            page_num = page.get("page_number")
            page_text = (page.get("content") or "").strip()
            blocks.append(f"<!-- PAGE {page_num} -->\n{page_text}\n")
        return "\n".join(blocks).strip()

    @staticmethod
    def _select_pages(
        pages: list[dict],
        *,
        slide_range: tuple[int, int] | None = None,
        max_slides: int | None = None,
    ) -> list[dict]:
        """Filter pages by range or max count, preserving order."""
        if not pages:
            return pages
        if slide_range is not None:
            start, end = slide_range
            return [page for page in pages if start <= int(page.get("page_number", 0)) <= end]
        if max_slides is not None:
            ordered = sorted(pages, key=lambda item: int(item.get("page_number", 0)))
            return ordered[:max_slides]
        return pages

    def parse_slides(self, content: str) -> list[dict]:
        """Parse content into individual slides."""
        slides = []
        content = content.replace("\r\n", "\n")
        pattern = r"<!-- PAGE (\d+) -->"
        parts = re.split(pattern, content)

        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                slide_num = int(parts[i])
                slide_content = parts[i + 1].strip()

                # Extract speaker notes (robust split + regex fallback)
                speaker_notes = None
                if "### Notes:" in slide_content:
                    head, tail = slide_content.split("### Notes:", 1)
                    slide_content = head.rstrip()
                    speaker_notes = tail.strip() or None
                    print(f"[Notes Parse] Slide {slide_num}: extracted speaker_notes via split.")
                else:
                    notes_match = re.search(
                        r"### Notes:\s*(.*?)(?=(?:\n\n)|$)", slide_content, re.DOTALL
                    )
                    speaker_notes = notes_match.group(1).strip() if notes_match else None
                    if notes_match:
                        print(f"[Notes Parse] Slide {slide_num}: extracted speaker_notes via regex.")
                        slide_content = slide_content[:notes_match.start()].rstrip()

                # Count images
                image_refs = re.findall(r"!\[.*?\]\(.*?\)", slide_content)

                slides.append({
                    "slide_number": slide_num,
                    "raw_text": slide_content,
                    "speaker_notes": speaker_notes,
                    "has_images": len(image_refs) > 0,
                    "image_count": len(image_refs),
                })

        return slides

    def extract_metrics(self, content: str) -> list[dict]:
        """Extract numeric metrics from content."""
        metrics = []
        pct_pattern = r"([+-]?\d+(?:\.\d+)?%)"
        currency_pattern = r"\$\d+(?:\.\d+)?"
        rate_pattern = r"(\d+(?:\.\d+)?)\s*(?:CTR|CPM|CPC|CPPC|CPV)"

        page_pattern = r"<!-- PAGE (\d+) -->"
        parts = re.split(page_pattern, content)

        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                slide_num = int(parts[i])
                slide_content = parts[i + 1]

                # Percentages
                for match in re.finditer(pct_pattern, slide_content):
                    start = max(0, match.start() - 50)
                    end = min(len(slide_content), match.end() + 50)
                    context = slide_content[start:end].replace("\n", " ").strip()
                    metrics.append({
                        "value": match.group(1),
                        "metric_type": "percentage",
                        "context": context,
                        "slide_number": slide_num,
                    })

                # Currency
                for match in re.finditer(currency_pattern, slide_content):
                    start = max(0, match.start() - 50)
                    end = min(len(slide_content), match.end() + 50)
                    context = slide_content[start:end].replace("\n", " ").strip()
                    metrics.append({
                        "value": match.group(),
                        "metric_type": "currency",
                        "context": context,
                        "slide_number": slide_num,
                    })

                # Rates
                for match in re.finditer(rate_pattern, slide_content):
                    start = max(0, match.start() - 50)
                    end = min(len(slide_content), match.end() + 50)
                    context = slide_content[start:end].replace("\n", " ").strip()
                    metrics.append({
                        "value": match.group(),
                        "metric_type": "rate",
                        "context": context,
                        "slide_number": slide_num,
                    })

        return metrics

    @staticmethod
    def _strip_notes_blocks(content: str) -> str:
        """Remove ### Notes: blocks up to the next page marker."""
        if not content:
            return content
        content = content.replace("\r\n", "\n")
        removed = 0
        out_parts: list[str] = []
        idx = 0
        while True:
            start = content.find("### Notes:", idx)
            if start == -1:
                out_parts.append(content[idx:])
                break
            out_parts.append(content[idx:start])
            end = content.find("<!-- PAGE", start)
            removed += 1
            if end == -1:
                idx = len(content)
                break
            idx = end
        if removed:
            print(f"[Notes Trim] Removing {removed} Notes block(s) from raw content.")
        return "".join(out_parts)

    @staticmethod
    def _normalize_match_text(text: str) -> str:
        """Normalize text for loose substring matching."""
        if not text:
            return ""
        return re.sub(r"[^a-z0-9]+", "", text.lower())

    def detect_charts(self, slides: list[dict]) -> list[dict]:
        """Detect chart patterns in slides."""
        charts = []

        for slide in slides:
            content = slide["raw_text"]
            slide_num = slide["slide_number"]

            # Multiple percentages suggest comparison chart
            pct_values = re.findall(r"[+-]?\d+(?:\.\d+)?%", content)
            if len(pct_values) >= 3:
                charts.append({
                    "slide_number": slide_num,
                    "chart_type": "comparison",
                    "raw_elements": pct_values,
                    "confidence": "medium",
                })

            # Key-value patterns suggest table
            colon_lines = re.findall(r"^.+:\s*.+$", content, re.MULTILINE)
            if len(colon_lines) >= 3:
                charts.append({
                    "slide_number": slide_num,
                    "chart_type": "table",
                    "raw_elements": colon_lines[:10],
                    "confidence": "high",
                })

        return charts

    @staticmethod
    def _extract_business_terms(content: str) -> set[str]:
        """Extract common QBR business terms from raw content."""
        business_terms: set[str] = set()
        patterns = [
            r"\b(?:CTR|CPM|CPC|CPPC|CPV|ROI|ROAS)\b",
            r"\b(?:reach|engagement|impressions|conversions?|clicks?)\b",
            r"\b(?:campaign|roadblock|carousel|banner|ad|creative)\b",
            r"\b(?:UK|DE|FR|IT|EMEA)\b",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, content, re.I)
            business_terms.update(m.upper() if len(m) <= 4 else m.title() for m in matches)
        return business_terms

    def _export_extraction_outputs(
        self,
        *,
        file_path: Path,
        result: ExtractionResult,
        slides: list[dict],
        metrics: list[dict],
        charts: list[dict],
        business_terms: set[str],
        output_dir: Path,
        chunks: list[dict] | None = None,
        content_override: str | None = None,
        pages_override: list[dict] | None = None,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        chunks_source = chunks if chunks is not None else (result.chunks or [])

        raw_content_export = content_override or result.content
        raw_content_export = self._strip_notes_blocks(raw_content_export)
        pages = pages_override or self._split_pages(raw_content_export)
        slide_texts = (
            self._load_pptx_slide_texts(file_path)
            if file_path.suffix.lower() == ".pptx" and pages_override is None
            else []
        )
        remap = (
            self._build_page_remap(pages, slide_texts)
            if slide_texts and pages and pages_override is None
            else None
        )

        slides_export = slides
        metrics_export = metrics
        charts_export = charts

        if remap:
            pages_by_num = {page["page_number"]: page for page in pages}
            raw_blocks: list[str] = []
            slides_by_num = {slide["slide_number"]: slide for slide in slides}
            slides_export = []
            for new_num, entry in enumerate(remap, start=1):
                old_num = entry["source_page_number"]
                page = pages_by_num.get(old_num)
                if page is None:
                    continue
                raw_blocks.append(f"<!-- PAGE {new_num} -->\n{page['content']}\n")
                slide = slides_by_num.get(old_num)
                if slide:
                    updated = dict(slide)
                    updated["slide_number"] = new_num
                    updated["source_slide_number"] = old_num
                    slides_export.append(updated)
            raw_content_export = "\n".join(raw_blocks).strip() + "\n"

            page_to_new = {entry["source_page_number"]: entry["slide_number"] for entry in remap}
            metrics_export = []
            for metric in metrics:
                updated = dict(metric)
                source_num = metric.get("slide_number")
                updated["slide_number"] = page_to_new.get(source_num, source_num)
                metrics_export.append(updated)
            charts_export = []
            for chart in charts:
                updated = dict(chart)
                source_num = chart.get("slide_number")
                updated["slide_number"] = page_to_new.get(source_num, source_num)
                charts_export.append(updated)

        output_metadata = {
            "source_file": str(file_path),
            "extraction_timestamp": datetime.now().isoformat(),
            "kreuzberg_version": kreuzberg.version("kreuzberg"),
            "mime_type": result.mime_type,
            "page_count": len(pages) if pages is not None else result.get_page_count(),
            "detected_languages": result.detected_languages,
            "table_count": len(result.tables),
            "image_count": len(result.images) if result.images else 0,
            "chunk_count": len(chunks_source),
            "metadata": result.metadata,
        }
        (output_dir / "01_extraction_metadata.json").write_text(
            json.dumps(output_metadata, indent=2, default=str),
            encoding="utf-8",
        )

        (output_dir / "02_raw_content.txt").write_text(raw_content_export, encoding="utf-8")
        (output_dir / "03_slides_parsed.json").write_text(
            json.dumps(slides_export, indent=2),
            encoding="utf-8",
        )
        (output_dir / "04_metrics_extracted.json").write_text(
            json.dumps(metrics_export, indent=2),
            encoding="utf-8",
        )
        (output_dir / "05_charts_detected.json").write_text(
            json.dumps(charts_export, indent=2),
            encoding="utf-8",
        )

        keywords_payload: dict[str, object] = {
            "business_terms_detected": sorted(business_terms),
        }
        if result.metadata and "keywords" in result.metadata:
            keywords_payload["raw_keywords"] = result.metadata["keywords"]
        (output_dir / "06_keywords_topics.json").write_text(
            json.dumps(keywords_payload, indent=2, default=str),
            encoding="utf-8",
        )

        images_payload: list[dict] = []
        for idx, img in enumerate(result.images or []):
            img_info = {
                "index": idx,
                "content_type": img.get("content_type", "unknown"),
                "size_bytes": len(img.get("data", b"")) if img.get("data") else 0,
                "source_location": img.get("source", "unknown"),
            }
            if img.get("data"):
                ext = "jpg"
                if "png" in img.get("content_type", ""):
                    ext = "png"
                elif "gif" in img.get("content_type", ""):
                    ext = "gif"
                img_path = images_dir / f"image_{idx:04d}.{ext}"
                data = img["data"]
                if isinstance(data, str):
                    data = base64.b64decode(data)
                with img_path.open("wb") as handle:
                    handle.write(data)
                img_info["saved_path"] = str(img_path)
            images_payload.append(img_info)
        (output_dir / "07_images_metadata.json").write_text(
            json.dumps(images_payload, indent=2),
            encoding="utf-8",
        )

        chunks_payload: list[dict] = []
        embedding_model = EmbeddingSettings.from_env().model_label()
        for idx, chunk in enumerate(chunks_source):
            embedding = chunk.get("embedding")
            chunks_payload.append(
                {
                    "chunk_index": idx,
                    "content": chunk.get("content", ""),
                    "char_count": len(chunk.get("content", "")),
                    "metadata": chunk.get("metadata", {}),
                    "has_embedding": embedding is not None,
                    "embedding_model": embedding_model if embedding is not None else None,
                    "embedding_dimensions": len(embedding) if embedding is not None else None,
                }
            )
        (output_dir / "08_chunks_rag.json").write_text(
            json.dumps(chunks_payload, indent=2),
            encoding="utf-8",
        )

        tables_payload: list[dict] = []
        for idx, table in enumerate(result.tables or []):
            tables_payload.append(
                {
                    "index": idx,
                    "headers": getattr(table, "headers", None),
                    "rows": getattr(table, "rows", None),
                    "raw": str(table),
                }
            )
        (output_dir / "09_tables_extracted.json").write_text(
            json.dumps(tables_payload, indent=2),
            encoding="utf-8",
        )

        llm_manifest = self._build_llm_task_manifest(
            document_path=str(file_path),
            slides=slides_export,
            metrics=metrics_export,
            charts=charts_export,
        )
        (output_dir / "10_llm_task_manifest.json").write_text(
            json.dumps(llm_manifest, indent=2, default=str),
            encoding="utf-8",
        )

        if remap:
            (output_dir / "00_slide_order_map.json").write_text(
                json.dumps(remap, indent=2),
                encoding="utf-8",
            )

    @staticmethod
    def _to_jsonable(value):
        """Convert LLM outputs to JSON-serializable structures."""
        if isinstance(value, dict):
            return {key: QBRProcessor._to_jsonable(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [QBRProcessor._to_jsonable(item) for item in value]
        if isinstance(value, Enum):
            return value.value
        if hasattr(value, "model_dump"):
            return QBRProcessor._to_jsonable(value.model_dump())
        if hasattr(value, "dict"):
            return QBRProcessor._to_jsonable(value.dict())
        if hasattr(value, "__dict__"):
            data = {
                key: val
                for key, val in value.__dict__.items()
                if not key.startswith("_")
            }
            if data:
                return QBRProcessor._to_jsonable(data)
        if isinstance(value, bytes):
            return base64.b64encode(value).decode("ascii")
        return value

    @staticmethod
    def _metric_candidate_to_dict(metric) -> dict:
        return {
            "id": metric.metric_id,
            "name": metric.name,
            "raw_value": metric.raw_value,
            "normalized_value": metric.normalized_value,
            "unit": metric.unit,
            "metric_type": metric.metric_type,
            "category": metric.category,
            "raw_context": metric.raw_context,
            "slide_number": metric.slide_number,
            "source": metric.source,
            "metric_catalog_id": metric.metric_catalog_id,
            "metric_catalog_slug": metric.metric_catalog_slug,
            "extraction_confidence": metric.extraction_confidence,
            "metadata": metric.metadata,
            "period_label": metric.period_label,
            "period_start": metric.period_start,
            "period_end": metric.period_end,
            "brand": metric.brand,
            "baseline_text": metric.baseline_text,
            "baseline_type": metric.baseline_type,
        }

    @staticmethod
    def _extract_context_window(
        text: str,
        anchors: list[str],
        *,
        line_window: int = 1,
        max_chars: int = 320,
    ) -> str | None:
        if not text:
            return None
        anchors_clean = [anchor.strip() for anchor in anchors if anchor and anchor.strip()]
        if not anchors_clean:
            return None
        lines = text.splitlines()
        for idx, line in enumerate(lines):
            line_lower = line.lower()
            if any(anchor.lower() in line_lower for anchor in anchors_clean):
                start = max(0, idx - line_window)
                end = min(len(lines), idx + line_window + 1)
                snippet = " ".join(l.strip() for l in lines[start:end] if l.strip())
                return snippet[:max_chars]
        text_lower = text.lower()
        for anchor in anchors_clean:
            pos = text_lower.find(anchor.lower())
            if pos >= 0:
                start = max(0, pos - max_chars // 2)
                end = min(len(text), pos + max_chars // 2)
                snippet = text[start:end].replace("\n", " ").strip()
                return snippet[:max_chars]
        return None

    @staticmethod
    def _build_metric_review_context(
        *,
        metric: MetricCandidate,
        slide: dict,
        catalog_entry: object | None,
    ) -> str:
        anchors = [metric.name, metric.raw_value]
        if catalog_entry is not None:
            aliases = [alias.alias for alias in getattr(catalog_entry, "aliases", [])]
            anchors.extend(aliases[:6])

        snippets: list[tuple[str, str]] = []
        if metric.raw_context:
            snippets.append(("raw_context", metric.raw_context.strip()))

        slide_text = slide.get("raw_text") or ""
        slide_snippet = QBRProcessor._extract_context_window(slide_text, anchors)
        if slide_snippet:
            snippets.append(("slide_text", slide_snippet))

        if not snippets and slide_text:
            fallback = slide_text.replace("\n", " ").strip()
            snippets.append(("slide_text", fallback[:320]))

        return "\n".join(f"{label}: {text}" for label, text in snippets if text)

    def _refine_metrics_with_llm(
        self,
        *,
        metrics: list,
        slides: list[dict],
        metric_dictionary: object,
    ) -> list:
        if not metrics:
            return metrics
        start_time = time.perf_counter()
        limit_env = (os.getenv("LLM_METRIC_SLIDE_LIMIT") or "").strip()
        slide_limit = int(limit_env) if limit_env.isdigit() else None
        slides_env = (os.getenv("LLM_METRIC_SLIDES") or "").strip()
        slide_allowlist: set[int] | None = None
        if slides_env:
            parsed = []
            for token in slides_env.split(","):
                token = token.strip()
                if not token:
                    continue
                if token.isdigit():
                    parsed.append(int(token))
            if parsed:
                slide_allowlist = set(parsed)
        slide_lookup = {s.get("slide_number"): s for s in slides}
        catalog_entries = self._load_metric_catalog_entries()
        catalog_by_name = {entry.name: entry for entry in catalog_entries}
        dictionary_by_name = {definition.name: definition for definition in metric_dictionary.definitions}

        metrics_by_group: dict[tuple[int | None, str], list] = {}
        for metric in metrics:
            metrics_by_group.setdefault((metric.slide_number, metric.name), []).append(metric)

        reviewer = MetricReviewer(lm=self.lm)
        refined: list = []
        processed = 0
        total_groups = len(metrics_by_group)
        for (slide_number, metric_name), group in metrics_by_group.items():
            if slide_number is None:
                refined.extend(group)
                continue
            if slide_allowlist is not None and slide_number not in slide_allowlist:
                refined.extend(group)
                continue
            if slide_limit is not None and processed >= slide_limit:
                refined.extend(group)
                continue
            slide = slide_lookup.get(slide_number, {})
            definition = dictionary_by_name.get(metric_name)
            catalog_entry = catalog_by_name.get(metric_name)
            context_snippets = self._build_metric_review_context(
                metric=group[0],
                slide=slide,
                catalog_entry=catalog_entry,
            )

            candidates = [
                {
                    "id": m.metric_id,
                    "raw_value": m.raw_value,
                    "normalized_value": m.normalized_value,
                    "unit": m.unit,
                    "source": m.source,
                    "context": (m.raw_context or "")[:240],
                    "confidence": m.extraction_confidence,
                }
                for m in group
            ]
            catalog_payload = {
                "name": metric_name,
                "expected_unit": getattr(catalog_entry, "expected_unit", None)
                or (definition.unit_hint if definition else None),
                "aliases": [alias.alias for alias in getattr(catalog_entry, "aliases", [])],
                "disambiguation": list(getattr(catalog_entry, "disambiguation", ())),
                "metric_id": getattr(catalog_entry, "metric_id", None),
            }
            try:
                result = reviewer(
                    metric_catalog=catalog_payload,
                    context_snippets=context_snippets,
                    candidates=candidates,
                )
            except Exception as exc:
                print(f"  [Metric Refine] LLM failed on slide {slide_number}: {exc}")
                refined.extend(group)
                continue
            processed += 1

            review = getattr(result, "review", None)
            chosen_index = getattr(review, "chosen_index", 0) if review is not None else 0
            if chosen_index is None or not isinstance(chosen_index, int):
                chosen_index = 0
            if chosen_index < 0 or chosen_index >= len(group):
                chosen_index = 0
            chosen = group[chosen_index]

            unit = getattr(review, "unit", None) if review is not None else None
            value = getattr(review, "normalized_value", None) if review is not None else None
            notes = getattr(review, "notes", None) if review is not None else None
            confidence = getattr(review, "confidence", None) if review is not None else None
            source_snippet = getattr(review, "source_snippet", None) if review is not None else None

            final_unit = unit or chosen.unit or (definition.unit_hint if definition else None)
            final_value = value if value is not None else chosen.normalized_value
            raw_value = chosen.raw_value or ""
            if final_value is not None:
                raw_value = str(final_value)
                if final_unit == "currency" and raw_value and not raw_value.startswith("$"):
                    raw_value = f"${raw_value}"
                if final_unit == "percent" and raw_value and not raw_value.endswith("%"):
                    raw_value = f"{raw_value}%"

            metadata = dict(chosen.metadata or {})
            metadata["llm_review"] = {
                "chosen_index": chosen_index,
                "notes": notes,
                "confidence": confidence,
                "source_snippet": source_snippet,
            }

            refined.append(
                MetricCandidate(
                    metric_id=_new_metric_id(),
                    name=chosen.name,
                    raw_value=raw_value,
                    normalized_value=final_value,
                    unit=final_unit,
                    raw_context=source_snippet or chosen.raw_context,
                    slide_number=slide_number,
                    metric_type=_metric_type_from_unit(final_unit),
                    source="llm_refined",
                    category=chosen.category,
                    extraction_confidence=confidence or chosen.extraction_confidence,
                    metadata=metadata,
                    baseline_text=chosen.baseline_text,
                    baseline_type=chosen.baseline_type,
                    period_label=chosen.period_label,
                    period_start=chosen.period_start,
                    period_end=chosen.period_end,
                    brand=chosen.brand,
                    metric_catalog_id=chosen.metric_catalog_id,
                    metric_catalog_slug=chosen.metric_catalog_slug,
                )
            )

        elapsed = time.perf_counter() - start_time
        print(
            f"  [Metric Refine] groups={total_groups} processed={processed} completed={elapsed:.2f}s"
        )
        return refined or metrics

    def _apply_metric_context_strategies(
        self,
        *,
        metrics: list[MetricCandidate],
        slides: list[dict],
        client_name: str | None,
        report_period: str | None,
        stage: str,
        allow_llm: bool,
    ) -> list[MetricCandidate]:
        if not metrics:
            return metrics
        strategies_env = (os.getenv("METRIC_CONTEXT_STRATEGIES") or "hybrid").strip()
        strategy_names = [s.strip().lower() for s in strategies_env.split(",") if s.strip()]
        document_context = DocumentContext(client_name=client_name, report_period=report_period)
        rule_strategy = RuleBasedMetricContextStrategy()
        strategies = []
        for name in strategy_names:
            if name == "rule_based":
                strategies.append(rule_strategy)
            elif name == "llm":
                if not allow_llm:
                    continue
                strategies.append(LLMMetricsContextStrategy(MetricContextExtractor(lm=self.lm)))
            elif name == "hybrid":
                if not allow_llm:
                    strategies.append(rule_strategy)
                    continue
                llm_strategy = LLMMetricsContextStrategy(MetricContextExtractor(lm=self.lm))
                strategies.append(
                    HybridMetricContextStrategy(
                        rule_strategy=rule_strategy,
                        llm_strategy=llm_strategy,
                    )
                )
        if not strategies:
            return metrics
        return apply_context_strategies(
            metrics=metrics,
            slides=slides,
            document_context=document_context,
            strategies=strategies,
            stage=stage,
        )

    @staticmethod
    def _infer_period_fields(period_label: str | None) -> dict:
        if not period_label:
            return {"period_type": None, "period_number": None, "fiscal_year": None}
        text = period_label.strip().upper()
        match = re.search(r"\bH([12])\s*FY(\d{2,4})\b", text)
        if match:
            return {
                "period_type": "half",
                "period_number": int(match.group(1)),
                "fiscal_year": QBRProcessor._parse_year(match.group(2)),
            }
        match = re.search(r"\bQ([1-4])\s*FY?(\d{2,4})\b", text)
        if match:
            return {
                "period_type": "quarter",
                "period_number": int(match.group(1)),
                "fiscal_year": QBRProcessor._parse_year(match.group(2)),
            }
        match = re.search(r"\bFY(\d{2,4})\b", text)
        if match:
            return {
                "period_type": "fy",
                "period_number": None,
                "fiscal_year": QBRProcessor._parse_year(match.group(1)),
            }
        return {"period_type": None, "period_number": None, "fiscal_year": None}

    @staticmethod
    def _parse_year(value: str) -> int:
        year = int(value)
        return 2000 + year if year < 100 else year

    def _upsert_periods(self, session, metrics: list[MetricCandidate]) -> dict[tuple, Period]:
        keys = {
            (m.period_label, m.period_start, m.period_end)
            for m in metrics
            if m.period_label or m.period_start or m.period_end
        }
        if not keys:
            return {}
        labels = {k[0] for k in keys if k[0]}
        existing = []
        if labels:
            existing = session.query(Period).filter(Period.period_label.in_(labels)).all()
        period_map: dict[tuple, Period] = {
            (p.period_label, p.start_date, p.end_date): p for p in existing
        }
        for period_label, period_start, period_end in keys:
            key = (period_label, period_start, period_end)
            if key in period_map:
                continue
            fields = self._infer_period_fields(period_label)
            period = Period(
                period_label=period_label or "Unknown",
                period_type=fields["period_type"],
                period_number=fields["period_number"],
                fiscal_year=fields["fiscal_year"],
                start_date=period_start,
                end_date=period_end,
            )
            session.add(period)
            period_map[key] = period
        session.flush()
        return period_map

    @classmethod
    def _build_chunks_from_pages(cls, pages: list[dict]) -> list[dict]:
        """Create chunk payloads from ordered page blocks."""
        chunks: list[dict] = []
        step = max(cls.CHUNK_MAX_CHARS - cls.CHUNK_OVERLAP, 1)
        for page in pages:
            page_num = page.get("page_number")
            page_text = (page.get("content") or "").strip()
            page_block = f"<!-- PAGE {page_num} -->\n{page_text}\n"
            offset = 0
            while offset < len(page_block):
                end = min(len(page_block), offset + cls.CHUNK_MAX_CHARS)
                chunk_text = page_block[offset:end]
                chunks.append(
                    {
                        "content": chunk_text,
                        "metadata": {
                            "byte_start": offset,
                            "byte_end": end,
                            "page_number": page_num,
                        },
                        "embedding": None,
                        "embedding_model": None,
                    }
                )
                if end >= len(page_block):
                    break
                offset += step
        return chunks

    @staticmethod
    def _infer_chunk_slide_range(chunk: dict) -> tuple[int | None, int | None]:
        metadata = chunk.get("metadata", {}) or {}
        page_number = metadata.get("page_number")
        if isinstance(page_number, int):
            return page_number, page_number
        content = chunk.get("content", "")
        match = re.search(r"<!-- PAGE (\d+) -->", content)
        if match:
            slide_num = int(match.group(1))
            return slide_num, slide_num
        return None, None

    @classmethod
    def _build_chunk_record(
        cls,
        *,
        document_id: int,
        chunk_data: dict,
        chunk_index: int,
        embedding_model: str,
    ) -> Chunk:
        start_slide, end_slide = cls._infer_chunk_slide_range(chunk_data)
        embedding = chunk_data.get("embedding")
        return Chunk(
            document_id=document_id,
            content=chunk_data.get("content", ""),
            chunk_index=chunk_index,
            byte_start=chunk_data.get("metadata", {}).get("byte_start"),
            byte_end=chunk_data.get("metadata", {}).get("byte_end"),
            char_count=len(chunk_data.get("content", "")),
            start_slide=start_slide,
            end_slide=end_slide,
            embedding=embedding,
            embedding_model=chunk_data.get("embedding_model")
            or (embedding_model if embedding is not None else None),
        )

    @staticmethod
    def _apply_slide_remap(
        *,
        slides: list[dict],
        metrics: list[dict],
        charts: list[dict],
        remap: list[dict] | None,
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """Remap slide/metric/chart slide numbers into PPTX order."""
        if not remap:
            return slides, metrics, charts

        slides_by_num = {slide["slide_number"]: slide for slide in slides}
        remapped_slides: list[dict] = []
        for entry in remap:
            old_num = entry["source_page_number"]
            slide = slides_by_num.get(old_num)
            if slide is None:
                continue
            updated = dict(slide)
            updated["slide_number"] = entry["slide_number"]
            updated["source_slide_number"] = old_num
            remapped_slides.append(updated)

        page_to_new = {entry["source_page_number"]: entry["slide_number"] for entry in remap}
        remapped_metrics = []
        for metric in metrics:
            updated = dict(metric)
            source_num = metric.get("slide_number")
            updated["slide_number"] = page_to_new.get(source_num, source_num)
            remapped_metrics.append(updated)

        remapped_charts = []
        for chart in charts:
            updated = dict(chart)
            source_num = chart.get("slide_number")
            updated["slide_number"] = page_to_new.get(source_num, source_num)
            remapped_charts.append(updated)

        return remapped_slides, remapped_metrics, remapped_charts

    @staticmethod
    def _build_llm_task_manifest(
        *,
        document_path: str,
        slides: list[dict],
        metrics: list[dict],
        charts: list[dict],
    ) -> dict:
        """Generate a manifest of LLM enhancement opportunities."""
        return {
            "generated_at": datetime.now().isoformat(),
            "document": document_path,
            "total_slides": len(slides),
            "total_metrics_found": len(metrics),
            "total_charts_detected": len(charts),
            "llm_tasks": {
                "slide_analysis": {
                    "description": "Analyze each slide to extract structured insights",
                    "task_type": "per_slide",
                    "count": len(slides),
                    "expected_output": {
                        "slide_type": "string (title, data, chart, summary, transition)",
                        "key_message": "string",
                        "metrics_structured": "array of {name, value, unit, trend}",
                        "action_items": "array of strings",
                        "confidence": "float 0-1",
                    },
                    "prompt_template": """Analyze this QBR slide and extract:
1. Slide type (title/data/chart/summary/transition)
2. Key message or insight
3. Structured metrics with names and values
4. Any action items or recommendations

Slide content:
{slide_content}

Speaker notes (if any):
{speaker_notes}
""",
                },
                "metric_normalization": {
                    "description": "Normalize and categorize all extracted metrics",
                    "task_type": "batch",
                    "count": len(metrics),
                    "expected_output": {
                        "metric_name": "string",
                        "value_normalized": "float",
                        "unit": "string",
                        "category": "string (performance, cost, reach, engagement)",
                        "trend": "string (up, down, stable, unknown)",
                        "benchmark_comparison": "string",
                    },
                    "prompt_template": """Given these metrics extracted from a QBR presentation:
{metrics_json}

For each metric:
1. Identify what it measures
2. Normalize the value
3. Categorize (performance/cost/reach/engagement)
4. Identify if it shows a trend
5. Compare to industry benchmarks if possible
""",
                },
                "chart_reconstruction": {
                    "description": "Reconstruct chart data from text elements",
                    "task_type": "per_chart",
                    "count": len(charts),
                    "expected_output": {
                        "chart_type": "string",
                        "title": "string",
                        "data_series": "array of {label, values}",
                        "x_axis": "string",
                        "y_axis": "string",
                        "insights": "array of strings",
                    },
                    "prompt_template": """These text elements were extracted from what appears to be a chart:
{chart_elements}

Reconstruct the chart data:
1. Determine the chart type
2. Identify labels and values
3. Structure as a data table
4. Extract key insights from the visualization
""",
                },
                "executive_summary": {
                    "description": "Generate executive summary from all slides",
                    "task_type": "aggregate",
                    "count": 1,
                    "expected_output": {
                        "summary": "string (2-3 paragraphs)",
                        "key_wins": "array of strings",
                        "areas_for_improvement": "array of strings",
                        "recommendations": "array of strings",
                        "next_steps": "array of strings",
                    },
                    "prompt_template": """Based on this QBR presentation content, generate an executive summary:

Full presentation text:
{full_content}

Include:
1. 2-3 paragraph summary
2. Key wins/achievements
3. Areas needing improvement
4. Strategic recommendations
5. Suggested next steps
""",
                },
                "image_analysis": {
                    "description": "Analyze images for charts, text, and brand elements",
                    "task_type": "per_image",
                    "count": "variable (see images output)",
                    "expected_output": {
                        "image_type": "string (chart, photo, logo, diagram, screenshot)",
                        "contains_text": "boolean",
                        "extracted_text": "string (if applicable)",
                        "chart_data": "object (if chart)",
                        "description": "string",
                    },
                    "prompt_template": """Analyze this image from a QBR presentation:
[Image attached]

1. What type of image is this?
2. Does it contain text? If so, extract it.
3. If it's a chart, extract the data.
4. Provide a description for accessibility.
""",
                },
                "entity_extraction": {
                    "description": "Extract named entities (companies, products, people, locations)",
                    "task_type": "aggregate",
                    "count": 1,
                    "expected_output": {
                        "companies": "array of strings",
                        "products": "array of strings",
                        "people": "array of strings",
                        "locations": "array of strings",
                        "dates": "array of strings",
                        "campaigns": "array of strings",
                    },
                    "prompt_template": """Extract all named entities from this QBR presentation:

{full_content}

Categories:
- Companies/Brands
- Products/Services
- People (speakers, stakeholders)
- Locations/Markets
- Dates/Time periods
- Campaign names
""",
                },
            },
            "priority_order": [
                "slide_analysis",
                "metric_normalization",
                "chart_reconstruction",
                "executive_summary",
                "entity_extraction",
                "image_analysis",
            ],
            "estimated_llm_calls": {
                "slide_analysis": len(slides),
                "metric_normalization": 1,
                "chart_reconstruction": len(charts),
                "executive_summary": 1,
                "entity_extraction": 1,
                "image_analysis": "depends on image count",
            },
        }

    @staticmethod
    def _split_pages(content: str) -> list[dict]:
        """Split raw content into page-numbered blocks."""
        pattern = r"<!-- PAGE (\d+) -->"
        parts = re.split(pattern, content)
        pages: list[dict] = []
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                page_num = int(parts[i])
                page_content = parts[i + 1].strip()
                pages.append({"page_number": page_num, "content": page_content})
        return pages

    @staticmethod
    def _normalize_tokens(text: str) -> set[str]:
        cleaned = re.sub(r"[^a-z0-9]+", " ", text.lower())
        return {token for token in cleaned.split() if len(token) > 2}

    @staticmethod
    def _normalize_year(raw_year: str) -> str:
        year = raw_year.strip()
        if len(year) == 2:
            return f"FY{year}"
        if len(year) == 4:
            return f"FY{year[-2:]}"
        return f"FY{year}"

    def _infer_client_name(self, file_stem: str) -> str | None:
        stem = file_stem.lower()
        matches: list[tuple[int, str]] = []
        for name in self._client_names:
            if not name:
                continue
            lower = name.lower()
            if lower in stem:
                matches.append((len(lower), name))
        if not matches:
            return None
        matches.sort(reverse=True)
        return matches[0][1]

    def _region_id_for_code(self, code: str | None) -> int | None:
        if not code:
            return None
        return self._region_map.get(code)

    def _infer_document_region(self, file_stem: str) -> tuple[int | None, str | None]:
        inferred_country = infer_country_from_text(file_stem)
        inferred_region = infer_region_from_text(file_stem)
        region_id = self._region_id_for_code(inferred_region)
        return region_id, inferred_country

    def _infer_metric_region(
        self,
        metric: MetricCandidate,
        *,
        document_region_id: int | None,
    ) -> tuple[int | None, str | None]:
        qualifiers = (metric.metadata or {}).get("qualifiers") or {}
        raw_context = metric.raw_context or ""
        qualifier_geo = qualifiers.get("geo")
        qualifier_country = qualifiers.get("country") or qualifiers.get("geo_country")

        if qualifier_country:
            inferred_country = qualifier_country
        else:
            inferred_country = infer_country_from_text(raw_context)

        inferred_region = None
        if qualifier_geo:
            inferred_region = infer_region_from_text(qualifier_geo)
        if not inferred_region and inferred_country:
            inferred_region = infer_region_from_text(inferred_country)
        if not inferred_region:
            inferred_region = infer_region_from_text(raw_context)

        region_id = self._region_id_for_code(inferred_region) or document_region_id
        return region_id, inferred_country

    def _seed_clients(self) -> None:
        with self.SessionLocal() as session:
            existing = session.query(Client).count()
            if existing:
                return
            session.add(Client(name="Disney+"))
            session.commit()

    def _load_client_names(self) -> list[str]:
        with self.SessionLocal() as session:
            rows = session.query(Client.name).all()
        return [row[0] for row in rows if row and row[0]]

    def _seed_regions(self) -> None:
        with self.SessionLocal() as session:
            existing = session.query(Region).count()
            if existing:
                return
            us = Region(code=REGION_US, name="United States")
            emea = Region(code=REGION_EMEA, name="Europe, Middle East, and Africa")
            session.add_all([us, emea])
            session.flush()
            session.add(
                RegionCountry(
                    region_id=us.id,
                    country_name="United States",
                    country_code="US",
                )
            )
            for country in sorted(EMEA_COUNTRIES):
                if country.lower() in {"uk", "uae"}:
                    continue
                session.add(
                    RegionCountry(
                        region_id=emea.id,
                        country_name=country,
                        country_code=None,
                    )
                )
            session.commit()

    def _load_region_map(self) -> dict[str, int]:
        with self.SessionLocal() as session:
            rows = session.query(Region.code, Region.id).all()
        return {code: region_id for code, region_id in rows if code}

    def _seed_metric_catalog(self) -> None:
        with self.SessionLocal() as session:
            existing = session.query(MetricCatalog).count()
            if existing:
                return
            applicability_notes = {
                "cost_per_acquisition": (
                    "Use Spend and Acquisitions only from placements targeting users who have not installed the app. "
                    "Exclude added value placements when computing CPA."
                ),
                "cost_per_install": (
                    "Use Spend and Installs only from placements targeting users who have not installed the app. "
                    "Exclude added value placements when computing CPI."
                ),
                "click_through_rate": "Clicks / Impressions for Homescreen placements.",
                "video_completion_rate": "Completes / Impressions for Video placements.",
            }
            definitions = build_metric_dictionary().definitions
            for definition in definitions:
                if not definition.name:
                    continue
                slug = definition.slug or definition.name.lower().replace(" ", "_")
                metric = MetricCatalog(
                    name=definition.name,
                    slug=slug,
                    category=definition.category,
                    default_unit=definition.unit_hint,
                    formula=definition.formula,
                    description=None,
                    applicability_notes=applicability_notes.get(slug),
                )
                session.add(metric)
                session.flush()
                for pattern in definition.patterns:
                    session.add(
                        MetricAlias(
                            metric_id=metric.id,
                            alias=definition.name,
                            pattern=pattern,
                            priority=0,
                            unit_override=None,
                        )
                    )
            session.commit()

    def _load_metric_catalog_map(self) -> dict[str, tuple[int, str]]:
        with self.SessionLocal() as session:
            rows = session.query(MetricCatalog.id, MetricCatalog.name, MetricCatalog.slug).all()
        return {name: (metric_id, slug) for metric_id, name, slug in rows if name}

    def _load_metric_catalog_entries(self) -> list:
        try:
            with self.SessionLocal() as session:
                entries = build_metric_catalog_from_db(session)
            return entries if entries else build_metric_catalog()
        except Exception:
            return build_metric_catalog()

    def _load_metric_unit_map(self) -> dict[str, str]:
        try:
            with self.SessionLocal() as session:
                rows = session.query(MetricCatalog.name, MetricCatalog.default_unit).all()
            return {name: unit for name, unit in rows if name}
        except Exception:
            return {definition.name: definition.unit_hint for definition in build_metric_dictionary().definitions}

    @staticmethod
    def _unit_matches_expected(actual: str | None, expected: str | None) -> bool:
        if not expected:
            return True
        actual = (actual or "").lower()
        expected = expected.lower()
        if expected == "percent":
            return actual == "percent"
        if expected == "currency":
            return actual == "currency"
        if expected == "time":
            return actual == "time"
        if expected == "count":
            return actual in {"count", ""}
        if expected == "ratio":
            return actual == "ratio"
        return True

    def _select_metrics_for_llm_refine(self, metrics: list[MetricCandidate]) -> list[MetricCandidate]:
        if not metrics:
            return []
        unit_map = self._load_metric_unit_map()
        by_slide_name: dict[tuple[int | None, str], list[MetricCandidate]] = {}
        for metric in metrics:
            by_slide_name.setdefault((metric.slide_number, metric.name), []).append(metric)

        conflicting: set[str] = set()
        for items in by_slide_name.values():
            values = {m.normalized_value for m in items}
            if len(values) > 1:
                for metric in items:
                    conflicting.add(metric.metric_id)

        selected: list[MetricCandidate] = []
        for metric in metrics:
            expected_unit = unit_map.get(metric.name)
            unit_mismatch = not self._unit_matches_expected(metric.unit, expected_unit)
            raw_value = (metric.raw_value or "").strip()
            raw_unit_mismatch = (
                ("%" in raw_value and metric.unit != "percent")
                or ("$" in raw_value and metric.unit != "currency")
            )
            ambiguous_label = metric.name.lower() in {"rate", "ratio", "value", "index"}
            reach_ambiguity = metric.name == "Reach" and any(
                token in (metric.raw_context or "").lower()
                for token in ("unique", "deduplicated", "unduplicated")
            )
            weak_source = metric.source == "speaker_notes"
            conflict = metric.metric_id in conflicting

            if unit_mismatch or raw_unit_mismatch or ambiguous_label or reach_ambiguity or weak_source or conflict:
                selected.append(metric)
        return selected

    def _apply_metric_catalog(self, metrics: list) -> list:
        if not metrics:
            return metrics
        updated = []
        for metric in metrics:
            catalog = self._metric_catalog_map.get(metric.name)
            if catalog:
                metric = replace(
                    metric,
                    metric_catalog_id=catalog[0],
                    metric_catalog_slug=catalog[1],
                )
            updated.append(metric)
        return updated

    def _ensure_client(self, session, name: str) -> Client | None:
        if not name:
            return None
        client = session.query(Client).filter(Client.name == name).one_or_none()
        if client:
            return client
        client = Client(name=name)
        session.add(client)
        session.flush()
        return client

    def _ensure_document_period(self, session, report_period: str | None) -> Period | None:
        if not report_period:
            return None

        label = self._normalize_period_label(report_period) or report_period.strip()
        if not label:
            return None

        period = session.query(Period).filter(Period.period_label == label).one_or_none()
        if period:
            return period

        fields = self._infer_period_fields(label)
        period = Period(
            period_label=label,
            period_type=fields["period_type"],
            period_number=fields["period_number"],
            fiscal_year=fields["fiscal_year"],
            start_date=None,
            end_date=None,
        )
        session.add(period)
        session.flush()
        return period

    def _infer_period_from_title(
        self, title: str
    ) -> tuple[str | None, str | None, str | None, str | None]:
        text = title.strip()
        if not text:
            return (None, None, None, None)

        half_match = re.search(r"\b(?:FY)?\s*'?(?P<year>\d{2,4})\s*H(?P<half>[12])\b", text, re.I)
        if half_match:
            fiscal_year = self._normalize_year(half_match.group("year"))
            half = f"H{half_match.group('half')}"
            return (f"{half} {fiscal_year}", fiscal_year, None, half)

        half_match = re.search(r"\bH(?P<half>[12])\s*(?:FY)?\s*'?(?P<year>\d{2,4})\b", text, re.I)
        if half_match:
            fiscal_year = self._normalize_year(half_match.group("year"))
            half = f"H{half_match.group('half')}"
            return (f"{half} {fiscal_year}", fiscal_year, None, half)

        q_match = re.search(r"\b(?:FY)?\s*'?(?P<year>\d{2,4})\s*Q(?P<quarter>[1-4])\b", text, re.I)
        if q_match:
            fiscal_year = self._normalize_year(q_match.group("year"))
            quarter = f"Q{q_match.group('quarter')}"
            return (f"{quarter} {fiscal_year}", fiscal_year, quarter, None)

        q_match = re.search(r"\bQ(?P<quarter>[1-4])\s*(?:FY)?\s*'?(?P<year>\d{2,4})\b", text, re.I)
        if q_match:
            fiscal_year = self._normalize_year(q_match.group("year"))
            quarter = f"Q{q_match.group('quarter')}"
            return (f"{quarter} {fiscal_year}", fiscal_year, quarter, None)

        fy_match = re.search(r"\bFY\s*'?(?P<year>\d{2,4})\b", text, re.I)
        if fy_match:
            fiscal_year = self._normalize_year(fy_match.group("year"))
            return (fiscal_year, fiscal_year, None, None)

        return (None, None, None, None)

    def _normalize_period_label(self, label: str | None) -> str | None:
        if not label:
            return None
        period_label, _, _, _ = self._infer_period_from_title(label)
        return period_label

    def _apply_document_period_defaults(
        self,
        metrics: list,
        *,
        document_period: str | None,
    ) -> list:
        if not metrics:
            return metrics
        doc_label = self._normalize_period_label(document_period)
        if not doc_label:
            return metrics
        filtered = []
        for metric in metrics:
            metric_label = self._normalize_period_label(getattr(metric, "period_label", None))
            if metric_label and metric_label != doc_label:
                continue
            if getattr(metric, "period_label", None) is None:
                metric = replace(metric, period_label=doc_label)
            filtered.append(metric)
        return filtered

    @staticmethod
    def _load_pptx_slide_texts(file_path: Path) -> list[dict]:
        """Load slide text in PPTX order from the PPTX XML."""
        ns = {
            "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
            "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
            "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        }
        with zipfile.ZipFile(file_path) as zf:
            pres_xml = zf.read("ppt/presentation.xml")
            rels_xml = zf.read("ppt/_rels/presentation.xml.rels")

            pres = ET.fromstring(pres_xml)
            rels = ET.fromstring(rels_xml)
            rId_to_target = {
                rel.attrib["Id"]: rel.attrib["Target"]
                for rel in rels.findall(
                    ".//{http://schemas.openxmlformats.org/package/2006/relationships}Relationship"
                )
            }
            slide_targets = []
            for sldId in pres.findall(".//p:sldIdLst/p:sldId", ns):
                rId = sldId.attrib.get(f"{{{ns['r']}}}id")
                target = rId_to_target.get(rId)
                if target:
                    slide_targets.append(target)

            slides: list[dict] = []
            for idx, target in enumerate(slide_targets, start=1):
                slide_xml = zf.read(f"ppt/{target}")
                slide = ET.fromstring(slide_xml)
                texts = [
                    t.text
                    for t in slide.findall(".//a:t", ns)
                    if t.text and t.text.strip()
                ]
                slide_text = " ".join(texts)
                slides.append({"slide_number": idx, "text": slide_text})
            return slides

    def _build_page_remap(self, pages: list[dict], slides: list[dict]) -> list[dict]:
        """Map extracted pages to PPTX slide order using token overlap."""
        if not pages or not slides:
            return []
        page_tokens = {
            page["page_number"]: self._normalize_tokens(page["content"]) for page in pages
        }
        slide_tokens = {
            slide["slide_number"]: self._normalize_tokens(slide.get("text", "")) for slide in slides
        }

        unassigned_pages = set(page_tokens)
        remap: list[dict] = []
        for slide in slides:
            slide_num = slide["slide_number"]
            best_page = None
            best_score = -1
            for page_num in unassigned_pages:
                score = len(slide_tokens[slide_num] & page_tokens[page_num])
                if score > best_score:
                    best_score = score
                    best_page = page_num
            if best_page is None:
                continue
            unassigned_pages.remove(best_page)
            remap.append(
                {
                    "slide_number": slide_num,
                    "source_page_number": best_page,
                    "token_overlap": best_score,
                }
            )
        return remap

    def process_document(
        self,
        file_path: str | Path,
        client_name: str | None = None,
        report_period: str | None = None,
        run_llm_metrics: bool = True,
        run_llm_enhancement: bool = True,
        run_llm_summary: bool = False,
        run_llm_adjudicator: bool = False,
        export_outputs: bool = False,
        output_dir: Path | str | None = None,
        slide_range: tuple[int, int] | None = None,
        max_slides: int | None = None,
    ) -> int:
        """
        Process a document through the full pipeline.

        Args:
            file_path: Path to the document
            client_name: Optional client name
            report_period: Optional report period
            run_llm_metrics: Whether to run LLM metric refinement
            run_llm_enhancement: Whether to run the full LLM enhancement pipeline
            run_llm_summary: Whether to run only the LLM executive summary
            slide_range: Optional (start, end) slide range to process
            max_slides: Optional max number of slides to process

        Returns:
            Document ID in the database
        """
        file_path = Path(file_path)
        document_url = self._resolve_document_url(file_path)

        # Step 1: Extract
        print("\n" + "=" * 60)
        print("STEP 1: EXTRACTION")
        print("=" * 60)

        result = self.extract_document(file_path)

        pages = self._split_pages(result.content)
        remap = None
        remapped_pages: list[dict] | None = None
        if file_path.suffix.lower() == ".pptx":
            slide_texts = self._load_pptx_slide_texts(file_path)
            if slide_texts and pages:
                remap = self._build_page_remap(pages, slide_texts)
        chunks_for_storage = result.chunks or []
        if remap:
            pages_by_num = {page["page_number"]: page for page in pages}
            remapped_pages = []
            for entry in remap:
                old_num = entry["source_page_number"]
                page = pages_by_num.get(old_num)
                if page is None:
                    continue
                remapped_pages.append(
                    {"page_number": entry["slide_number"], "content": page["content"]}
                )
            if remapped_pages:
                chunks_for_storage = self._build_chunks_from_pages(remapped_pages)

        pages_for_selection = remapped_pages or pages
        content_for_processing = result.content
        pages_override: list[dict] | None = None
        if slide_range is not None or max_slides is not None:
            selected_pages = self._select_pages(
                pages_for_selection,
                slide_range=slide_range,
                max_slides=max_slides,
            )
            if selected_pages:
                content_for_processing = self._build_content_from_pages(selected_pages)
                pages_override = selected_pages
                chunks_for_storage = self._build_chunks_from_pages(selected_pages)
        content_for_processing = self._strip_notes_blocks(content_for_processing)
        remaining_notes = len(re.findall(r"### Notes:", content_for_processing))
        if remaining_notes:
            print(f"[Notes Trim] WARNING: {remaining_notes} Notes block(s) remain after trim.")

        inferred_client = client_name or self._infer_client_name(file_path.stem)
        inferred_region_id, _ = self._infer_document_region(file_path.stem)
        inferred_period, inferred_fiscal_year, inferred_quarter, inferred_half = (
            self._infer_period_from_title(report_period or file_path.stem)
        )
        report_period = report_period or inferred_period
        fiscal_year = inferred_fiscal_year
        quarter = inferred_quarter
        half = inferred_half

        post_embedding_settings = PostEmbeddingSettings.from_env()
        try:
            filled = apply_post_embeddings(chunks_for_storage, post_embedding_settings)
            if filled:
                print(
                    "[Embeddings] Post-extraction embeddings filled "
                    f"{filled} chunk(s) using {post_embedding_settings.model_label()}"
                )
        except RuntimeError as exc:
            print(f"[Embeddings] Post-extraction embeddings failed: {exc}")

        # Step 2: Parse
        print("\n" + "=" * 60)
        print("STEP 2: PARSING")
        print("=" * 60)

        slides = self.parse_slides(content_for_processing)
        metrics = self.extract_metrics(content_for_processing)
        charts = self.detect_charts(slides)
        remap_for_parsing = remap if pages_override is None else None
        slides_ordered, metrics_ordered, charts_ordered = self._apply_slide_remap(
            slides=slides,
            metrics=metrics,
            charts=charts,
            remap=remap_for_parsing,
        )
        self._attach_google_slide_ids(
            slides_ordered=slides_ordered,
            slides_raw=slides,
            remap=remap_for_parsing,
            file_path=file_path,
            result=result,
        )
        business_terms = self._extract_business_terms(content_for_processing)

        tables_payload: list[dict] = []
        for idx, table in enumerate(result.tables or []):
            tables_payload.append(
                {
                    "index": idx,
                    "headers": getattr(table, "headers", None),
                    "rows": getattr(table, "rows", None),
                    "raw": str(table),
                }
            )

        metric_debug = None
        scanned_metrics: list[MetricCandidate] = []
        if file_path.suffix.lower() == ".pptx":
            deck = parse_pptx_deck(file_path)
            if slide_range is not None:
                start, end = slide_range
                deck = deck.__class__(
                    deck_id=deck.deck_id,
                    slides=tuple(
                        slide for slide in deck.slides if start <= slide.slide_index <= end
                    ),
                )
            elif max_slides is not None:
                deck = deck.__class__(
                    deck_id=deck.deck_id,
                    slides=tuple(deck.slides[:max_slides]),
                )
            cache_dir = self.output_dir / ".metric_adjudicator_cache"
            adjudicator = None
            if run_llm_adjudicator:
                adjudicator = LLMAdjudicator(
                    call_llm=self._call_llm_text,
                    cache_dir=cache_dir,
                    enabled=True,
                )
            catalog_entries = self._load_metric_catalog_entries()
            pipeline = MetricExtractionPipeline(
                config=PipelineConfig(),
                adjudicator=adjudicator,
                catalog_entries=catalog_entries,
            )
            deck_hash = compute_deck_hash(file_path)
            extracted_metrics, metric_debug = pipeline.extract_from_deck(
                deck, deck_hash=deck_hash
            )
            scanned_metrics = [to_metric_candidate(metric) for metric in extracted_metrics]
            if adjudicator is not None:
                print(
                    "  [Metric Adjudicator] "
                    f"requests={adjudicator.total_requests} "
                    f"cache_hits={adjudicator.cache_hits} "
                    f"llm_calls={adjudicator.llm_calls} "
                    f"parse_failures={adjudicator.parse_failures} "
                    f"empty_responses={adjudicator.empty_responses}"
                )
        else:
            metric_scanner = MetricScanner(build_metric_dictionary())
            artifacts = MetricScanArtifacts(
                raw_content=result.content,
                slides=slides_ordered,
                metrics=metrics_ordered,
                charts=charts_ordered,
                chunks=chunks_for_storage,
                tables=tables_payload,
                keywords=business_terms,
                export_dir=None,
            )
            scanned_metrics = metric_scanner.scan(artifacts)
            scanned_metrics = metric_scanner.dedupe_exact(scanned_metrics)

        llm_deduped_metrics = scanned_metrics
        refined_metrics = None
        use_legacy_llm_metrics = os.getenv("LEGACY_LLM_METRICS", "false").lower() in {
            "1",
            "true",
            "yes",
        }
        if run_llm_metrics and use_legacy_llm_metrics:
            llm_candidates = self._select_metrics_for_llm_refine(scanned_metrics)
            if llm_candidates:
                llm_deduped_subset = self._dedupe_metrics_with_llm(llm_candidates)
                refined_subset = self._refine_metrics_with_llm(
                    metrics=llm_deduped_subset,
                    slides=slides_ordered,
                    metric_dictionary=build_metric_dictionary(),
                )
                refined_ids = {metric.metric_id for metric in llm_candidates}
                llm_deduped_metrics = [
                    metric for metric in scanned_metrics if metric.metric_id not in refined_ids
                ] + list(refined_subset)
                refined_metrics = llm_deduped_metrics
            else:
                llm_deduped_metrics = scanned_metrics
        scanned_metrics = self._apply_metric_context_strategies(
            metrics=scanned_metrics,
            slides=slides_ordered,
            client_name=inferred_client,
            report_period=report_period,
            stage="scanned",
            allow_llm=run_llm_metrics and use_legacy_llm_metrics,
        )
        llm_deduped_metrics = self._apply_metric_context_strategies(
            metrics=llm_deduped_metrics,
            slides=slides_ordered,
            client_name=inferred_client,
            report_period=report_period,
            stage="deduped",
            allow_llm=run_llm_metrics and use_legacy_llm_metrics,
        )
        if refined_metrics is not None:
            refined_metrics = self._apply_metric_context_strategies(
                metrics=refined_metrics,
                slides=slides_ordered,
                client_name=inferred_client,
                report_period=report_period,
                stage="refined",
                allow_llm=run_llm_metrics and use_legacy_llm_metrics,
            )
        metrics_for_db = refined_metrics or llm_deduped_metrics

        if report_period:
            scanned_metrics = self._apply_document_period_defaults(
                scanned_metrics,
                document_period=report_period,
            )
            llm_deduped_metrics = self._apply_document_period_defaults(
                llm_deduped_metrics,
                document_period=report_period,
            )
            if refined_metrics is not None:
                refined_metrics = self._apply_document_period_defaults(
                    refined_metrics,
                    document_period=report_period,
                )
            metrics_for_db = self._apply_document_period_defaults(
                metrics_for_db,
                document_period=report_period,
            )

        scanned_metrics = self._apply_metric_catalog(scanned_metrics)
        llm_deduped_metrics = self._apply_metric_catalog(llm_deduped_metrics)
        if refined_metrics is not None:
            refined_metrics = self._apply_metric_catalog(refined_metrics)
        metrics_for_db = self._apply_metric_catalog(metrics_for_db)

        # Drop metrics sourced only from speaker notes to prevent note-only leakage.
        if metrics_for_db:
            metrics_for_db = [
                metric for metric in metrics_for_db if metric.source != "speaker_notes"
            ]
        # Remove metrics whose raw_context appears in speaker notes (normalized match).
        notes_by_slide: dict[int | None, str] = {}
        for slide in slides_ordered:
            slide_number = slide.get("slide_number")
            speaker_notes = slide.get("speaker_notes") or ""
            raw_text = slide.get("raw_text") or ""
            notes_section = ""
            if "### Notes:" in raw_text:
                notes_section = raw_text.split("### Notes:", 1)[1]
            combined = (speaker_notes + " " + notes_section).strip()
            if combined:
                notes_by_slide[slide_number] = self._normalize_match_text(combined)
        if notes_by_slide and metrics_for_db:
            before_count = len(metrics_for_db)
            filtered_metrics = []
            for metric in metrics_for_db:
                note_text = notes_by_slide.get(metric.slide_number)
                raw_context = (metric.raw_context or "").strip()
                if note_text and raw_context:
                    norm_context = self._normalize_match_text(raw_context)
                    if norm_context and norm_context in note_text:
                        continue
                filtered_metrics.append(metric)
            metrics_for_db = filtered_metrics
            removed = before_count - len(metrics_for_db)
            if removed:
                print(f"  - Metrics removed (speaker notes match): {removed}")

        print(f"  - Slides parsed: {len(slides_ordered)}")
        print(f"  - Metrics found: {len(metrics_ordered)}")
        print(f"  - Business metrics detected: {len(metrics_for_db)}")
        print(f"  - Charts detected: {len(charts_ordered)}")

        if export_outputs:
            base_dir = Path(output_dir) if output_dir else self.output_dir
            target_dir = base_dir / file_path.stem
            self._export_extraction_outputs(
                file_path=file_path,
                result=result,
                slides=slides,
                metrics=metrics,
                charts=charts,
                business_terms=business_terms,
                output_dir=target_dir,
                chunks=chunks_for_storage,
                content_override=content_for_processing if pages_override else None,
                pages_override=pages_override,
            )
            (target_dir / "11_business_metrics.json").write_text(
                json.dumps(
                    [self._metric_candidate_to_dict(m) for m in scanned_metrics],
                    indent=2,
                ),
                encoding="utf-8",
            )
            if metric_debug is not None:
                (target_dir / "11b_business_metrics_debug.json").write_text(
                    json.dumps(metric_debug.to_dict(), indent=2, default=str),
                    encoding="utf-8",
                )
            if run_llm_enhancement:
                (target_dir / "12_business_metrics_deduped.json").write_text(
                    json.dumps(
                        [self._metric_candidate_to_dict(m) for m in llm_deduped_metrics],
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                if refined_metrics is not None:
                    (target_dir / "13_business_metrics_refined.json").write_text(
                        json.dumps(
                            [self._metric_candidate_to_dict(m) for m in refined_metrics],
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
            print(f"  - Extraction outputs saved: {target_dir}")

        # Step 3: Create database records
        print("\n" + "=" * 60)
        print("STEP 3: DATABASE STORAGE")
        print("=" * 60)

        with self.SessionLocal() as session:
            client = self._ensure_client(session, inferred_client) if inferred_client else None
            doc_period = self._ensure_document_period(session, report_period)
            # Create document
            document = Document(
                filename=file_path.name,
                file_path=document_url,
                mime_type=result.mime_type,
                status=DocumentStatus.EXTRACTED.value,
                page_count=result.get_page_count(),
                slide_count=len(slides_ordered),
                image_count=len(result.images) if result.images else 0,
                chunk_count=len(chunks_for_storage),
                extraction_metadata=result.metadata,
                detected_languages=result.detected_languages,
                client_id=client.id if client else None,
                region_id=inferred_region_id,
                report_period_id=doc_period.id if doc_period else None,
                fiscal_year=fiscal_year,
                quarter=quarter,
                half=half,
            )
            session.add(document)
            session.flush()  # Get document ID

            print(f"  - Document ID: {document.id}")

            # Create slides (bulk insert)
            slides_to_add = [
                Slide(
                    document_id=document.id,
                    slide_number=s["slide_number"],
                    raw_text=s["raw_text"],
                    speaker_notes=s.get("speaker_notes"),
                    google_slide_id=s.get("google_slide_id"),
                    has_images=s["has_images"],
                    image_count=s["image_count"],
                )
                for s in slides_ordered
            ]
            session.add_all(slides_to_add)
            session.flush()

            # Build slide_number -> Slide mapping
            slide_map = {s.slide_number: s for s in slides_to_add}
            print(f"  - Slides created: {len(slide_map)}")

            # Create metrics (bulk insert)
            metrics_to_add: list[Metric] = []
            skipped_without_slide = 0
            for m in metrics_for_db:
                slide = slide_map.get(m.slide_number)
                if slide is None:
                    skipped_without_slide += 1
                    continue
                _, country = self._infer_metric_region(
                    m, document_region_id=inferred_region_id
                )
                metrics_to_add.append(
                    Metric(
                        slide_id=slide.id,
                        raw_value=m.raw_value,
                        raw_context=m.raw_context,
                        raw_metric_type=m.metric_type,
                        metric_catalog_id=m.metric_catalog_id,
                        name=m.name,
                        normalized_value=m.normalized_value,
                        unit=m.unit,
                        category=m.category,
                        extraction_confidence=m.extraction_confidence,
                        baseline_text=m.baseline_text,
                        baseline_type=m.baseline_type,
                        country=country,
                    )
                )
            if skipped_without_slide:
                print(f"  - Metrics skipped (missing slide mapping): {skipped_without_slide}")
            # Final guard: drop metrics whose raw_context matches notes text for the slide.
            notes_by_slide_id = {}
            for slide in slides_to_add:
                raw_text = slide.raw_text or ""
                notes_section = ""
                if "### Notes:" in raw_text:
                    notes_section = raw_text.split("### Notes:", 1)[1]
                combined = (slide.speaker_notes or "") + " " + notes_section
                combined = self._normalize_match_text(combined)
                if combined:
                    notes_by_slide_id[slide.id] = combined
            if notes_by_slide_id and metrics_to_add:
                before_count = len(metrics_to_add)
                filtered = []
                for metric in metrics_to_add:
                    note_text = notes_by_slide_id.get(metric.slide_id)
                    raw_context = (metric.raw_context or "").strip()
                    if note_text and raw_context:
                        norm_context = self._normalize_match_text(raw_context)
                        if norm_context and norm_context in note_text:
                            continue
                    filtered.append(metric)
                metrics_to_add = filtered
                removed = before_count - len(metrics_to_add)
                if removed:
                    print(f"  - Metrics removed (notes guard): {removed}")
            session.add_all(metrics_to_add)
            print(f"  - Metrics created: {len(metrics_to_add)}")

            # Post-insert cleanup: remove metrics whose raw_context matches notes text.
            if metrics_to_add:
                notes_by_slide_id = {}
                for slide in slides_to_add:
                    raw_text = slide.raw_text or ""
                    notes_section = ""
                    if "### Notes:" in raw_text:
                        notes_section = raw_text.split("### Notes:", 1)[1]
                    combined = (slide.speaker_notes or "") + " " + notes_section
                    combined = self._normalize_match_text(combined)
                    if combined:
                        notes_by_slide_id[slide.id] = combined
                if notes_by_slide_id:
                    to_remove = []
                    for metric in metrics_to_add:
                        note_text = notes_by_slide_id.get(metric.slide_id)
                        raw_context = (metric.raw_context or "").strip()
                        if note_text and raw_context:
                            norm_context = self._normalize_match_text(raw_context)
                            if norm_context and norm_context in note_text:
                                to_remove.append(metric)
                    if to_remove:
                        for metric in to_remove:
                            session.delete(metric)
                        print(f"  - Metrics removed (notes guard): {len(to_remove)}")

            # Remove metrics whose raw_context appears inside the Notes section only.
            deleted = session.execute(
                text(
                    """
                    DELETE FROM metrics
                    WHERE document_id = :document_id
                      AND raw_context IS NOT NULL
                      AND TRIM(raw_context) != ''
                      AND slide_id IN (
                        SELECT id
                        FROM slides
                        WHERE document_id = :document_id
                          AND speaker_notes IS NOT NULL
                          AND TRIM(speaker_notes) != ''
                          AND instr(speaker_notes, raw_context) > 0
                          AND instr(raw_text, '### Notes:') > 0
                          AND instr(substr(raw_text, instr(raw_text, '### Notes:')), raw_context) > 0
                      )
                    """
                ),
                {"document_id": document.id},
            ).rowcount
            if deleted:
                print(f"  - Metrics removed (speaker notes cleanup): {deleted}")

            # Create charts (bulk insert)
            charts_to_add = [
                Chart(
                    document_id=document.id,
                    slide_id=slide_map.get(c["slide_number"]).id if slide_map.get(c["slide_number"]) else None,
                    chart_type=c.get("chart_type", ChartType.UNKNOWN.value),
                    detection_confidence=c.get("confidence"),
                    raw_elements=c.get("raw_elements"),
                )
                for c in charts_ordered
            ]
            session.add_all(charts_to_add)
            print(f"  - Charts created: {len(charts_ordered)}")

            # Create chunks (bulk insert)
            if chunks_for_storage:
                embedding_model = self.embedding_settings.model_label()
                chunks_to_add = [
                    self._build_chunk_record(
                        document_id=document.id,
                        chunk_data=chunk_data,
                        chunk_index=idx,
                        embedding_model=embedding_model,
                    )
                    for idx, chunk_data in enumerate(chunks_for_storage)
                ]
                session.add_all(chunks_to_add)
                print(f"  - Chunks created: {len(chunks_for_storage)}")

            # Create images (bulk insert, skip loading image data)
            if result.images:
                images_to_add = [
                    Image(
                        document_id=document.id,
                        image_index=idx,
                        content_type=img_data.get("content_type"),
                        size_bytes=img_data.get("size", 0),  # Use metadata size, don't load bytes
                    )
                    for idx, img_data in enumerate(result.images)
                ]
                session.add_all(images_to_add)
                print(f"  - Images created: {len(result.images)}")

            # Extract and create keywords (bulk insert)
            keywords_to_add = [
                Keyword(document_id=document.id, keyword=term, category="business_term")
                for term in business_terms
            ]
            session.add_all(keywords_to_add)
            print(f"  - Keywords created: {len(business_terms)}")

            session.commit()
            doc_id = document.id

        # Step 4: LLM Enhancement (optional)
        if run_llm_enhancement:
            print("\n" + "=" * 60)
            print("STEP 4: LLM ENHANCEMENT")
            print("=" * 60)

            llm_results = self.enhance_document(
                doc_id,
                result.content,
                slides_ordered,
                [
                    {
                        "value": m.raw_value,
                        "metric_type": m.metric_type,
                        "context": m.raw_context,
                        "slide_number": m.slide_number,
                        "name": m.name,
                    }
                    for m in metrics_for_db
                ],
                charts_ordered,
            )
            if export_outputs and llm_results is not None:
                base_dir = Path(output_dir) if output_dir else self.output_dir
                target_dir = base_dir / file_path.stem
                (target_dir / "10_llm_outputs.json").write_text(
                    json.dumps(self._to_jsonable(llm_results), indent=2, default=str),
                    encoding="utf-8",
                )
                print(f"  - LLM outputs saved: {target_dir / '10_llm_outputs.json'}")
        elif run_llm_summary:
            print("\n" + "=" * 60)
            print("STEP 4: LLM SUMMARY")
            print("=" * 60)
            summary_result = self.enhance_summary(
                doc_id,
                content_for_processing,
            )
            if export_outputs and summary_result is not None:
                base_dir = Path(output_dir) if output_dir else self.output_dir
                target_dir = base_dir / file_path.stem
                (target_dir / "10_llm_outputs.json").write_text(
                    json.dumps(self._to_jsonable({"executive_summary": summary_result}), indent=2, default=str),
                    encoding="utf-8",
                )
                print(f"  - LLM outputs saved: {target_dir / '10_llm_outputs.json'}")

        print("\n" + "=" * 60)
        print("PROCESSING COMPLETE")
        print("=" * 60)
        print(f"Document ID: {doc_id}")

        return doc_id

    def enhance_document(
        self,
        document_id: int,
        full_content: str,
        slides: list[dict],
        metrics: list[dict],
        charts: list[dict],
    ) -> dict | None:
        """
        Run LLM enhancement on a document.

        Args:
            document_id: ID of the document to enhance
            full_content: Full text content
            slides: Parsed slide data
            metrics: Extracted metrics
            charts: Detected charts
        """
        try:
            print("  Running LLM enhancement pipeline...")

            # Get document context
            with self.SessionLocal() as session:
                doc = session.get(Document, document_id)
                context = f"Client: {doc.client_name or 'Unknown'}, Period: {doc.report_period or 'Unknown'}"

            # Run enhancement (limit for cost control)
            limit_env = (os.getenv("LLM_SLIDE_LIMIT") or "").strip()
            slide_limit = int(limit_env) if limit_env.isdigit() else 20
            limited_slides = slides[:slide_limit]
            limited_metrics = metrics[:50]
            limited_charts = charts[:10]

            results = self.enhancement_pipeline(
                slides=limited_slides,
                metrics=limited_metrics,
                charts=limited_charts,
                full_content=full_content[:30000],  # Limit content
                document_context=context,
            )

            print(f"  - Slides analyzed: {len(results.get('slides', []))}")
            print(f"  - Charts reconstructed: {len(results.get('charts', []))}")

            # Update database with enhancements
            self._save_enhancements(document_id, results)

            print("  Enhancement complete!")
            return results

        except Exception as e:
            print(f"  Enhancement failed: {e}")
            # Update status
            with self.SessionLocal() as session:
                doc = session.get(Document, document_id)
                if doc:
                    doc.status = DocumentStatus.EXTRACTED.value  # Keep as extracted
                    session.commit()
            return None

    def enhance_summary(self, document_id: int, full_content: str):
        try:
            print("  Running LLM executive summary...")
            with self.SessionLocal() as session:
                doc = session.get(Document, document_id)
                context = f"Client: {doc.client_name or 'Unknown'}, Period: {doc.report_period or 'Unknown'}"
            summarizer = ExecutiveSummarizer(lm=self.lm)
            summary = summarizer(full_content=full_content, document_metadata=context)
            self._save_enhancements(document_id, {"executive_summary": summary})
            print("  Summary complete!")
            return summary
        except Exception as exc:
            print(f"  Summary failed: {exc}")
            return None

    def _dedupe_metrics_with_llm(
        self,
        metrics: list,
    ) -> list:
        if not metrics:
            return metrics
        try:
            deduper = MetricDeduplicator(lm=self.lm)
            by_name_value: dict[tuple[str, str | None, float | None], list] = {}
            for metric in metrics:
                rounded_value = (
                    round(metric.normalized_value, 2)
                    if metric.normalized_value is not None
                    else None
                )
                key = (metric.name, metric.unit, rounded_value)
                by_name_value.setdefault(key, []).append(metric)

            remove_ids: set[str] = set()
            max_batch = 60
            for key, group in by_name_value.items():
                if len(group) < 2:
                    continue
                batches = [group[i : i + max_batch] for i in range(0, len(group), max_batch)]
                for batch in batches:
                    payload = [
                        {
                            "id": m.metric_id,
                            "name": m.name,
                            "value": m.normalized_value,
                            "unit": m.unit,
                            "raw_value": m.raw_value,
                            "context": m.raw_context,
                            "slide_number": m.slide_number,
                            "source": m.source,
                        }
                        for m in batch
                    ]
                    result = deduper(metrics=payload)
                    batch_remove = set(getattr(result, "remove_ids", []) or [])
                    remove_ids.update(batch_remove)

            if not remove_ids:
                return metrics
            filtered = [m for m in metrics if m.metric_id not in remove_ids]
            print(
                f"  [Metric Dedupe] Removed {len(remove_ids)} duplicate metrics via LLM."
            )
            return filtered
        except Exception as exc:
            print(f"  [Metric Dedupe] LLM dedupe failed: {exc}")
            return metrics

    def _save_enhancements(self, document_id: int, results: dict):
        """Save LLM enhancement results to database."""
        with self.SessionLocal() as session:
            doc = session.get(Document, document_id)
            if not doc:
                return

            # Update executive summary
            exec_summary = results.get("executive_summary")
            if exec_summary and not isinstance(exec_summary, dict):
                doc.executive_summary = getattr(exec_summary, "summary", None)
                doc.key_wins = getattr(exec_summary, "key_wins", None)
                doc.areas_for_improvement = getattr(
                    exec_summary, "areas_for_improvement", None
                )
                doc.recommendations = getattr(exec_summary, "recommendations", None)
                doc.next_steps = getattr(exec_summary, "next_steps", None)

            # Update slides with analysis
            slide_results = results.get("slides", [])
            for sr in slide_results:
                if "analysis" not in sr:
                    continue

                analysis = sr["analysis"]
                slide_num = sr.get("slide_number")

                # Find the slide
                slide = (
                    session.query(Slide)
                    .filter(
                        Slide.document_id == document_id,
                        Slide.slide_number == slide_num,
                    )
                    .first()
                )

                if slide:
                    slide.slide_type = getattr(
                        analysis, "slide_type", SlideType.UNKNOWN
                    ).value
                    slide.title = getattr(analysis, "title", None)
                    slide.key_message = getattr(analysis, "key_message", None)
                    slide.insights = getattr(analysis, "insights", None)
                    slide.action_items = getattr(analysis, "action_items", None)
                    slide.has_charts = getattr(analysis, "has_chart", False)
                    slide.classification_confidence = getattr(
                        analysis, "confidence", None
                    )

            # Update metrics with normalization
            metric_results = results.get("metrics")
            if metric_results and not isinstance(metric_results, dict):
                normalized_metrics = getattr(metric_results, "metrics", [])
                # Match normalized metrics back to database metrics
                # This is a simplified version - production would need better matching
                db_metrics = (
                    session.query(Metric)
                    .join(Slide, Slide.id == Metric.slide_id)
                    .filter(Slide.document_id == document_id)
                    .all()
                )

                for i, nm in enumerate(normalized_metrics[:len(db_metrics)]):
                    db_metric = db_metrics[i]
                    db_metric.name = getattr(nm, "metric_name", None)
                    db_metric.normalized_value = getattr(nm, "normalized_value", None)
                    db_metric.unit = getattr(nm, "unit", None)
                    category = getattr(nm, "category", None)
                    db_metric.category = (
                        category.value
                        if hasattr(category, "value")
                        else str(category) if category else "other"
                    )
                    # Trend/benchmark fields are no longer stored on metrics.

            # Update charts with reconstruction data
            chart_results = results.get("charts", [])
            for cr in chart_results:
                if not isinstance(cr, dict) or "reconstruction" not in cr:
                    continue
                reconstruction = cr["reconstruction"]
                slide_num = cr.get("slide_number")

                # Find the slide
                slide = (
                    session.query(Slide)
                    .filter(
                        Slide.document_id == document_id,
                        Slide.slide_number == slide_num,
                    )
                    .first()
                )

                if slide:
                    # Find chart(s) for this slide
                    db_charts = (
                        session.query(Chart).filter(Chart.slide_id == slide.id).all()
                    )
                    for db_chart in db_charts:
                        db_chart.title = getattr(reconstruction, "title", None)
                        db_chart.x_axis_label = getattr(
                            reconstruction, "x_axis_label", None
                        )
                        db_chart.y_axis_label = getattr(
                            reconstruction, "y_axis_label", None
                        )
                        db_chart.insights = getattr(reconstruction, "insights", None)
                        db_chart.trends_identified = getattr(
                            reconstruction, "trends", None
                        )

            # Save entities
            entity_results = results.get("entities")
            if entity_results and not isinstance(entity_results, dict):
                entities_list = getattr(entity_results, "entities", [])

                for entity_data in entities_list:
                    entity_type = getattr(entity_data, "entity_type", None)
                    entity = Entity(
                        document_id=document_id,
                        name=getattr(entity_data, "name", ""),
                        entity_type=(
                            entity_type.value
                            if hasattr(entity_type, "value")
                            else str(entity_type) if entity_type else "other"
                        ),
                        normalized_name=getattr(entity_data, "normalized_name", None),
                        description=getattr(entity_data, "context", None),
                        mention_count=getattr(entity_data, "mention_count", 1),
                    )
                    session.add(entity)

            # Update document status
            doc.status = DocumentStatus.ENHANCED.value
            doc.processed_at = datetime.utcnow()

            session.commit()

    def get_document(self, document_id: int) -> Document | None:
        """Get a document by ID."""
        with self.SessionLocal() as session:
            return session.get(Document, document_id)

    def get_document_summary(self, document_id: int) -> dict:
        """Get a summary of a document."""
        with self.SessionLocal() as session:
            doc = session.get(Document, document_id)
            if not doc:
                return {}

            return {
                "id": doc.id,
                "filename": doc.filename,
                "status": doc.status,
                "page_count": doc.page_count,
                "executive_summary": doc.executive_summary,
                "key_wins": doc.key_wins,
                "recommendations": doc.recommendations,
                "client_name": doc.client_name,
                "report_period": doc.report_period,
            }

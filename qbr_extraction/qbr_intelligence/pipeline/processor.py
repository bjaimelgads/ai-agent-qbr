"""
QBR Document Processor

Complete pipeline for:
1. Extracting content from PPTX using Kreuzberg
2. Enhancing with LLM using DSPY
3. Storing in SQLAlchemy database
"""

import base64
import json
import re
import zipfile
from datetime import datetime
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
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from qbr_intelligence.db.models import (
    Base,
    Chart,
    ChartType,
    Chunk,
    Document,
    DocumentStatus,
    Entity,
    Image,
    Keyword,
    Metric,
    Slide,
    SlideType,
)
from qbr_intelligence.llm.modules import (
    QBREnhancementPipeline,
    create_lm,
)
from qbr_intelligence.pipeline.embeddings import EmbeddingSettings
from qbr_intelligence.pipeline.post_embeddings import (
    PostEmbeddingSettings,
    apply_post_embeddings,
)


class QBRProcessor:
    """
    Complete QBR document processing pipeline.

    Handles extraction, LLM enhancement, and database storage.
    """

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

        # Initialize LLM (will be created when needed)
        self.llm_model = llm_model
        self._lm: "dspy.LM | None" = None

        # Enhancement pipeline (lazy loaded)
        self._enhancement_pipeline: QBREnhancementPipeline | None = None

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

        return result

    def parse_slides(self, content: str) -> list[dict]:
        """Parse content into individual slides."""
        slides = []
        pattern = r"<!-- PAGE (\d+) -->"
        parts = re.split(pattern, content)

        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                slide_num = int(parts[i])
                slide_content = parts[i + 1].strip()

                # Extract speaker notes
                notes_match = re.search(
                    r"### Notes:\s*(.*?)(?=\n\n|$)", slide_content, re.DOTALL
                )
                speaker_notes = notes_match.group(1).strip() if notes_match else None

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
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        slide_texts = self._load_pptx_slide_texts(file_path) if file_path.suffix.lower() == ".pptx" else []
        pages = self._split_pages(result.content)
        remap = self._build_page_remap(pages, slide_texts) if slide_texts and pages else None

        slides_export = slides
        metrics_export = metrics
        charts_export = charts
        raw_content_export = result.content

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
            "page_count": result.get_page_count(),
            "detected_languages": result.detected_languages,
            "table_count": len(result.tables),
            "image_count": len(result.images) if result.images else 0,
            "chunk_count": result.get_chunk_count(),
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
        for idx, chunk in enumerate(result.chunks or []):
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

        if remap:
            (output_dir / "00_slide_order_map.json").write_text(
                json.dumps(remap, indent=2),
                encoding="utf-8",
            )

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
        run_llm_enhancement: bool = True,
        export_outputs: bool = False,
        output_dir: Path | str | None = None,
    ) -> int:
        """
        Process a document through the full pipeline.

        Args:
            file_path: Path to the document
            client_name: Optional client name
            report_period: Optional report period
            run_llm_enhancement: Whether to run LLM enhancement

        Returns:
            Document ID in the database
        """
        file_path = Path(file_path)

        # Step 1: Extract
        print("\n" + "=" * 60)
        print("STEP 1: EXTRACTION")
        print("=" * 60)

        result = self.extract_document(file_path)
        post_embedding_settings = PostEmbeddingSettings.from_env()
        try:
            filled = apply_post_embeddings(result.chunks or [], post_embedding_settings)
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

        slides = self.parse_slides(result.content)
        metrics = self.extract_metrics(result.content)
        charts = self.detect_charts(slides)
        business_terms = self._extract_business_terms(result.content)

        print(f"  - Slides parsed: {len(slides)}")
        print(f"  - Metrics found: {len(metrics)}")
        print(f"  - Charts detected: {len(charts)}")

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
            )
            print(f"  - Extraction outputs saved: {target_dir}")

        # Step 3: Create database records
        print("\n" + "=" * 60)
        print("STEP 3: DATABASE STORAGE")
        print("=" * 60)

        with self.SessionLocal() as session:
            # Create document
            document = Document(
                filename=file_path.name,
                file_path=str(file_path.absolute()),
                mime_type=result.mime_type,
                status=DocumentStatus.EXTRACTED.value,
                page_count=result.get_page_count(),
                slide_count=len(slides),
                image_count=len(result.images) if result.images else 0,
                chunk_count=result.get_chunk_count(),
                extraction_metadata=result.metadata,
                detected_languages=result.detected_languages,
                client_name=client_name,
                report_period=report_period,
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
                    has_images=s["has_images"],
                    image_count=s["image_count"],
                )
                for s in slides
            ]
            session.add_all(slides_to_add)
            session.flush()

            # Build slide_number -> Slide mapping
            slide_map = {s.slide_number: s for s in slides_to_add}
            print(f"  - Slides created: {len(slide_map)}")

            # Create metrics (bulk insert)
            metrics_to_add = [
                Metric(
                    document_id=document.id,
                    slide_id=slide_map.get(m["slide_number"]).id if slide_map.get(m["slide_number"]) else None,
                    raw_value=m["value"],
                    raw_context=m.get("context"),
                    raw_metric_type=m["metric_type"],
                )
                for m in metrics
            ]
            session.add_all(metrics_to_add)
            print(f"  - Metrics created: {len(metrics)}")

            # Create charts (bulk insert)
            charts_to_add = [
                Chart(
                    document_id=document.id,
                    slide_id=slide_map.get(c["slide_number"]).id if slide_map.get(c["slide_number"]) else None,
                    chart_type=c.get("chart_type", ChartType.UNKNOWN.value),
                    detection_confidence=c.get("confidence"),
                    raw_elements=c.get("raw_elements"),
                )
                for c in charts
            ]
            session.add_all(charts_to_add)
            print(f"  - Charts created: {len(charts)}")

            # Create chunks (bulk insert)
            if result.chunks:
                embedding_model = self.embedding_settings.model_label()
                chunks_to_add = [
                    Chunk(
                        document_id=document.id,
                        content=chunk_data.get("content", ""),
                        chunk_index=idx,
                        byte_start=chunk_data.get("metadata", {}).get("byte_start"),
                        byte_end=chunk_data.get("metadata", {}).get("byte_end"),
                        char_count=len(chunk_data.get("content", "")),
                        embedding=chunk_data.get("embedding"),
                        embedding_model=chunk_data.get("embedding_model")
                        or (
                            embedding_model
                            if chunk_data.get("embedding") is not None
                            else None
                        ),
                    )
                    for idx, chunk_data in enumerate(result.chunks)
                ]
                session.add_all(chunks_to_add)
                print(f"  - Chunks created: {len(result.chunks)}")

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

            self.enhance_document(doc_id, result.content, slides, metrics, charts)

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
    ):
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
            limited_slides = slides[:20]  # Limit slides for now
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

        except Exception as e:
            print(f"  Enhancement failed: {e}")
            # Update status
            with self.SessionLocal() as session:
                doc = session.get(Document, document_id)
                if doc:
                    doc.status = DocumentStatus.EXTRACTED.value  # Keep as extracted
                    session.commit()

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
                    .filter(Metric.document_id == document_id)
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
                    trend = getattr(nm, "trend", None)
                    db_metric.trend = (
                        trend.value
                        if hasattr(trend, "value")
                        else str(trend) if trend else "unknown"
                    )
                    db_metric.is_positive_trend = getattr(nm, "is_positive", None)
                    db_metric.change_percentage = getattr(nm, "change_percentage", None)
                    db_metric.context = getattr(nm, "significance", None)
                    db_metric.benchmark_comparison = getattr(nm, "benchmark_notes", None)

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

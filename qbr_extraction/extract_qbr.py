#!/usr/bin/env python3
"""
Comprehensive QBR Document Extraction Script using Kreuzberg 4.0

This script demonstrates all extraction capabilities of kreuzberg for complex
PowerPoint presentations containing unstructured data, images, charts, and metrics.

Outputs multiple JSON files showcasing different extraction capabilities.
LLM enhancement opportunities are marked with TODO comments.
"""

import base64
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import kreuzberg
from dotenv import load_dotenv
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

from qbr_intelligence.pipeline.embeddings import EmbeddingSettings

# Load environment variables from .env if present
load_dotenv()

# =============================================================================
# Configuration
# =============================================================================

INPUT_FILE = "disney.pptx"
OUTPUT_DIR = Path("extraction_output")


# =============================================================================
# Data Classes for Structured Output
# =============================================================================


@dataclass
class SlideData:
    """Structured data for a single slide."""

    slide_number: int
    raw_text: str
    speaker_notes: str | None
    has_images: bool
    image_count: int
    # TODO [LLM]: Extract structured metrics (percentages, KPIs) from raw_text
    # TODO [LLM]: Identify slide type (title, data, chart, summary, etc.)
    # TODO [LLM]: Extract key insights/takeaways
    llm_enhancement_opportunities: list[str]


@dataclass
class MetricData:
    """Extracted metric from presentation."""

    value: str
    metric_type: str  # percentage, currency, count, rate
    context: str
    slide_number: int
    # TODO [LLM]: Normalize metrics to standard format
    # TODO [LLM]: Identify what the metric measures (CTR, reach, cost, etc.)
    # TODO [LLM]: Compare with industry benchmarks


@dataclass
class ChartData:
    """Detected chart/visual data representation."""

    slide_number: int
    chart_type_guess: str  # bar, pie, line, table, etc.
    raw_elements: list[str]
    # TODO [LLM]: Reconstruct chart data from text elements
    # TODO [LLM]: Generate structured data table from visual representation
    # TODO [LLM]: Identify trends and patterns


# =============================================================================
# Extraction Functions
# =============================================================================


def create_extraction_config() -> ExtractionConfig:
    """Create comprehensive extraction configuration."""
    embedding_config = EmbeddingSettings.from_env().to_embedding_config()
    return ExtractionConfig(
        # Quality processing for better text extraction
        enable_quality_processing=True,
        # Don't force OCR - let kreuzberg decide when needed
        force_ocr=False,
        # OCR configuration (for images with text)
        ocr=OcrConfig(
            backend="tesseract",  # Can also use "easyocr" or "paddleocr"
            language="eng",
        ),
        # Extract embedded images
        images=ImageExtractionConfig(
            extract_images=True,
            target_dpi=150,  # Balance quality vs size
            max_image_dimension=2048,
        ),
        # Page/slide tracking (using default marker format)
        pages=PageConfig(
            extract_pages=True,
            insert_page_markers=True,
            # Note: default marker_format is "<!-- PAGE {n} -->" which kreuzberg substitutes correctly
        ),
        # Keyword extraction for topic analysis
        keywords=KeywordConfig(
            algorithm=KeywordAlgorithm.Yake,
            max_keywords=30,
            min_score=0.01,
            ngram_range=(1, 3),  # Extract 1-3 word phrases
            language="en",
        ),
        # Language detection
        language_detection=LanguageDetectionConfig(
            enabled=True,
            detect_multiple=True,
            min_confidence=0.5,
        ),
        # Chunking for RAG pipelines
        chunking=ChunkingConfig(
            max_chars=1500,
            max_overlap=200,
            embedding=embedding_config,
        ),
    )


def extract_document(file_path: str, config: ExtractionConfig) -> ExtractionResult:
    """Extract all data from document."""
    print(f"Extracting: {file_path}")
    result = kreuzberg.extract_file_sync(file_path, None, config)
    print(f"  - Pages: {result.get_page_count()}")
    print(f"  - Images: {len(result.images) if result.images else 0}")
    print(f"  - Tables: {len(result.tables)}")
    print(f"  - Chunks: {result.get_chunk_count()}")
    return result


def parse_slides_from_content(content: str) -> list[dict]:
    """Parse content into individual slides based on page markers."""
    slides = []

    # Split by page markers
    pattern = r"<!-- PAGE (\d+) -->"
    parts = re.split(pattern, content)

    # parts[0] is before first marker (usually empty)
    # then alternating: page_number, content, page_number, content...
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            slide_num = int(parts[i])
            slide_content = parts[i + 1].strip()

            # Extract speaker notes if present
            notes_match = re.search(
                r"### Notes:\s*(.*?)(?=\n\n|$)", slide_content, re.DOTALL
            )
            speaker_notes = notes_match.group(1).strip() if notes_match else None

            # Count images referenced in this slide
            image_refs = re.findall(r"!\[.*?\]\(.*?\)", slide_content)

            slides.append(
                {
                    "slide_number": slide_num,
                    "raw_text": slide_content,
                    "speaker_notes": speaker_notes,
                    "has_images": len(image_refs) > 0,
                    "image_count": len(image_refs),
                    "llm_enhancement_opportunities": [
                        "Extract structured metrics from bullet points",
                        "Identify slide type and purpose",
                        "Extract key insights and takeaways",
                        "Parse visual chart data from text elements",
                    ],
                }
            )

    return slides


def extract_metrics_from_content(content: str) -> list[dict]:
    """Extract numeric metrics from content."""
    metrics = []

    # Pattern for percentages
    pct_pattern = r"([+-]?\d+(?:\.\d+)?%)"
    # Pattern for currency
    currency_pattern = r"\$\d+(?:\.\d+)?"
    # Pattern for rates/ratios (like CTR, CPM)
    rate_pattern = r"(\d+(?:\.\d+)?)\s*(?:CTR|CPM|CPC|CPPC|CPV)"

    # Split by pages first
    page_pattern = r"<!-- PAGE (\d+) -->"
    parts = re.split(page_pattern, content)

    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            slide_num = int(parts[i])
            slide_content = parts[i + 1]

            # Extract percentages
            for match in re.finditer(pct_pattern, slide_content):
                # Get surrounding context (50 chars before and after)
                start = max(0, match.start() - 50)
                end = min(len(slide_content), match.end() + 50)
                context = slide_content[start:end].replace("\n", " ").strip()

                metrics.append(
                    {
                        "value": match.group(1),
                        "metric_type": "percentage",
                        "context": context,
                        "slide_number": slide_num,
                        "llm_enhancement": "Identify what this percentage measures and its business significance",
                    }
                )

            # Extract currency values
            for match in re.finditer(currency_pattern, slide_content):
                start = max(0, match.start() - 50)
                end = min(len(slide_content), match.end() + 50)
                context = slide_content[start:end].replace("\n", " ").strip()

                metrics.append(
                    {
                        "value": match.group(),
                        "metric_type": "currency",
                        "context": context,
                        "slide_number": slide_num,
                        "llm_enhancement": "Identify cost type (CPC, CPM, total spend, etc.)",
                    }
                )

            # Extract rate metrics (CTR, CPM, etc.)
            for match in re.finditer(rate_pattern, slide_content):
                start = max(0, match.start() - 50)
                end = min(len(slide_content), match.end() + 50)
                context = slide_content[start:end].replace("\n", " ").strip()

                metrics.append(
                    {
                        "value": match.group(),
                        "metric_type": "rate",
                        "context": context,
                        "slide_number": slide_num,
                        "llm_enhancement": "Identify rate type and benchmark comparison",
                    }
                )

    return metrics


def detect_chart_patterns(slides: list[dict]) -> list[dict]:
    """Detect patterns that suggest chart/visual data in slides."""
    charts = []

    for slide in slides:
        content = slide["raw_text"]
        slide_num = slide["slide_number"]

        # Detect bar chart patterns (multiple percentage values in sequence)
        pct_values = re.findall(r"[+-]?\d+(?:\.\d+)?%", content)
        if len(pct_values) >= 3:
            charts.append(
                {
                    "slide_number": slide_num,
                    "chart_type_guess": "bar_chart_or_comparison",
                    "raw_elements": pct_values,
                    "confidence": "medium",
                    "llm_enhancement": "Reconstruct the bar chart data with labels and values",
                }
            )

        # Detect comparison patterns (vs, compared to, etc.)
        if re.search(r"(?:vs\.?|versus|compared to|comparison)", content, re.I):
            charts.append(
                {
                    "slide_number": slide_num,
                    "chart_type_guess": "comparison_chart",
                    "raw_elements": re.findall(r"(?:^|\n)-\s*(.+?)(?=\n|$)", content)[
                        :10
                    ],
                    "confidence": "medium",
                    "llm_enhancement": "Extract comparison dimensions and values",
                }
            )

        # Detect tabular patterns (multiple colons or consistent formatting)
        colon_lines = re.findall(r"^.+:\s*.+$", content, re.MULTILINE)
        if len(colon_lines) >= 3:
            charts.append(
                {
                    "slide_number": slide_num,
                    "chart_type_guess": "table_or_key_value_pairs",
                    "raw_elements": colon_lines[:10],
                    "confidence": "high",
                    "llm_enhancement": "Parse into structured table format",
                }
            )

    return charts


def extract_keywords_and_topics(result: ExtractionResult) -> dict:
    """Extract keywords and organize by topic."""
    keywords_data = {
        "extraction_method": "YAKE algorithm via kreuzberg",
        "raw_keywords": [],
        "suggested_topics": [],
        "llm_enhancement": "Cluster keywords into business topics and generate topic summaries",
    }

    # Get keywords from metadata if available
    if result.metadata and "keywords" in result.metadata:
        keywords_data["raw_keywords"] = result.metadata["keywords"]

    # Manual keyword patterns for business metrics
    content = result.content
    business_terms = set()

    patterns = [
        r"\b(?:CTR|CPM|CPC|CPPC|CPV|ROI|ROAS)\b",
        r"\b(?:reach|engagement|impressions|conversions?|clicks?)\b",
        r"\b(?:campaign|roadblock|carousel|banner|ad|creative)\b",
        r"\b(?:market|region|country|UK|DE|FR|IT|EMEA)\b",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, content, re.I)
        business_terms.update(m.upper() if len(m) <= 4 else m.title() for m in matches)

    keywords_data["business_terms_detected"] = list(business_terms)

    return keywords_data


def process_images(result: ExtractionResult, output_dir: Path) -> list[dict]:
    """Process and save extracted images."""
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    image_data = []

    if not result.images:
        return image_data

    for idx, img in enumerate(result.images):
        img_info = {
            "index": idx,
            "content_type": img.get("content_type", "unknown"),
            "size_bytes": len(img.get("data", b"")) if img.get("data") else 0,
            "source_location": img.get("source", "unknown"),
            "llm_enhancement_opportunities": [
                "OCR text from image if contains text",
                "Describe image content for accessibility",
                "Extract chart data if image is a chart/graph",
                "Identify brand elements and logos",
            ],
        }

        # Save image if data is available
        if img.get("data"):
            ext = "jpg"
            if "png" in img.get("content_type", ""):
                ext = "png"
            elif "gif" in img.get("content_type", ""):
                ext = "gif"

            img_path = images_dir / f"image_{idx:04d}.{ext}"

            # Handle base64 encoded data
            data = img["data"]
            if isinstance(data, str):
                data = base64.b64decode(data)

            with open(img_path, "wb") as f:
                f.write(data)

            img_info["saved_path"] = str(img_path)

        image_data.append(img_info)

    return image_data


def create_chunks_output(result: ExtractionResult) -> list[dict]:
    """Process chunks for RAG pipeline."""
    chunks_data = []

    if not result.chunks:
        return chunks_data

    embedding_model = EmbeddingSettings.from_env().model_label()

    for idx, chunk in enumerate(result.chunks):
        embedding = chunk.get("embedding")
        chunk_info = {
            "chunk_index": idx,
            "content": chunk.get("content", ""),
            "char_count": len(chunk.get("content", "")),
            "metadata": chunk.get("metadata", {}),
            "has_embedding": embedding is not None,
            "embedding_model": embedding_model if embedding is not None else None,
            "embedding_dimensions": len(embedding) if embedding is not None else None,
            "llm_enhancement_opportunities": [
                "Generate semantic summary of chunk",
                "Extract entities (people, companies, products)",
                "Identify chunk topic for better retrieval",
                "Score chunk importance/relevance",
            ],
        }
        chunks_data.append(chunk_info)

    return chunks_data


def generate_llm_task_manifest(slides: list, metrics: list, charts: list) -> dict:
    """Generate a manifest of all LLM enhancement opportunities."""

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "document": INPUT_FILE,
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
            "metric_normalization": 1,  # Batch process
            "chart_reconstruction": len(charts),
            "executive_summary": 1,
            "entity_extraction": 1,
            "image_analysis": "depends on image count",
        },
    }

    return manifest


# =============================================================================
# Main Execution
# =============================================================================


def main():
    """Main extraction pipeline."""
    print("=" * 60)
    print("QBR Document Extraction Pipeline")
    print("Using Kreuzberg 4.0")
    print("=" * 60)

    # Create output directory per input file
    base_output_dir = OUTPUT_DIR
    target_dir = base_output_dir / Path(INPUT_FILE).stem
    target_dir.mkdir(parents=True, exist_ok=True)

    # Create extraction configuration
    config = create_extraction_config()

    # Extract document
    print("\n[1/7] Extracting document...")
    result = extract_document(INPUT_FILE, config)

    # Parse slides
    print("\n[2/7] Parsing slides...")
    slides = parse_slides_from_content(result.content)
    print(f"  - Parsed {len(slides)} slides")

    # Extract metrics
    print("\n[3/7] Extracting metrics...")
    metrics = extract_metrics_from_content(result.content)
    print(f"  - Found {len(metrics)} metrics")

    # Detect charts
    print("\n[4/7] Detecting chart patterns...")
    charts = detect_chart_patterns(slides)
    print(f"  - Detected {len(charts)} potential charts")

    # Extract keywords
    print("\n[5/7] Extracting keywords and topics...")
    keywords = extract_keywords_and_topics(result)
    print(
        f"  - Found {len(keywords.get('business_terms_detected', []))} business terms"
    )

    # Process images
    print("\n[6/7] Processing images...")
    images = process_images(result, target_dir)
    print(f"  - Processed {len(images)} images")

    # Process chunks
    print("\n[7/7] Processing chunks for RAG...")
    chunks = create_chunks_output(result)
    print(f"  - Created {len(chunks)} chunks")

    # Generate LLM task manifest
    llm_manifest = generate_llm_task_manifest(slides, metrics, charts)

    # ==========================================================================
    # Save all outputs
    # ==========================================================================

    print("\n" + "=" * 60)
    print("Saving outputs...")
    print("=" * 60)

    # 1. Raw extraction result metadata
    output_metadata = {
        "source_file": INPUT_FILE,
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
    with open(target_dir / "01_extraction_metadata.json", "w", encoding="utf-8") as f:
        json.dump(output_metadata, f, indent=2, default=str)
    print("  - Saved: 01_extraction_metadata.json")

    # 2. Full raw content
    with open(target_dir / "02_raw_content.txt", "w", encoding="utf-8") as f:
        f.write(result.content)
    print("  - Saved: 02_raw_content.txt")

    # 3. Slides parsed
    with open(target_dir / "03_slides_parsed.json", "w", encoding="utf-8") as f:
        json.dump(slides, f, indent=2)
    print("  - Saved: 03_slides_parsed.json")

    # 4. Metrics extracted
    with open(target_dir / "04_metrics_extracted.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("  - Saved: 04_metrics_extracted.json")

    # 5. Charts detected
    with open(target_dir / "05_charts_detected.json", "w", encoding="utf-8") as f:
        json.dump(charts, f, indent=2)
    print("  - Saved: 05_charts_detected.json")

    # 6. Keywords and topics
    with open(target_dir / "06_keywords_topics.json", "w", encoding="utf-8") as f:
        json.dump(keywords, f, indent=2)
    print("  - Saved: 06_keywords_topics.json")

    # 7. Images metadata
    with open(target_dir / "07_images_metadata.json", "w", encoding="utf-8") as f:
        json.dump(images, f, indent=2)
    print("  - Saved: 07_images_metadata.json")

    # 8. Chunks for RAG
    with open(target_dir / "08_chunks_rag.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)
    print("  - Saved: 08_chunks_rag.json")

    # 9. Tables (if any)
    tables_data = []
    for idx, table in enumerate(result.tables):
        tables_data.append(
            {
                "index": idx,
                "headers": table.headers if hasattr(table, "headers") else None,
                "rows": table.rows if hasattr(table, "rows") else None,
                "raw": str(table),
            }
        )
    with open(target_dir / "09_tables_extracted.json", "w", encoding="utf-8") as f:
        json.dump(tables_data, f, indent=2)
    print("  - Saved: 09_tables_extracted.json")

    # 10. LLM Enhancement Manifest (THE KEY FILE)
    with open(target_dir / "10_llm_task_manifest.json", "w", encoding="utf-8") as f:
        json.dump(llm_manifest, f, indent=2)
    print("  - Saved: 10_llm_task_manifest.json")

    # ==========================================================================
    # Summary
    # ==========================================================================

    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {target_dir.absolute()}")
    print("\nFiles generated:")
    print("  01_extraction_metadata.json  - Document metadata and extraction info")
    print("  02_raw_content.txt           - Full extracted text content")
    print("  03_slides_parsed.json        - Individual slide data")
    print("  04_metrics_extracted.json    - Numeric metrics with context")
    print("  05_charts_detected.json      - Detected chart patterns")
    print("  06_keywords_topics.json      - Keywords and business terms")
    print("  07_images_metadata.json      - Image metadata and paths")
    print("  08_chunks_rag.json           - Chunks ready for RAG pipeline")
    print("  09_tables_extracted.json     - Extracted tables")
    print("  10_llm_task_manifest.json    - LLM enhancement task definitions")
    print("  images/                      - Extracted images directory")

    print("\n" + "=" * 60)
    print("LLM ENHANCEMENT OPPORTUNITIES")
    print("=" * 60)
    print("""
The following tasks can be enhanced with LLM processing:

1. SLIDE ANALYSIS (per slide)
   - Classify slide type (title, data, chart, summary)
   - Extract key message and insights
   - Structure metrics into normalized format

2. METRIC NORMALIZATION (batch)
   - Identify what each metric measures
   - Categorize by type (performance, cost, reach)
   - Compare against benchmarks

3. CHART RECONSTRUCTION (per chart)
   - Rebuild chart data from text elements
   - Generate structured data tables
   - Extract visual insights

4. EXECUTIVE SUMMARY (aggregate)
   - Generate 2-3 paragraph summary
   - Identify key wins and improvements
   - Suggest recommendations

5. IMAGE ANALYSIS (per image)
   - OCR text from images
   - Extract chart data from chart images
   - Describe images for accessibility

6. ENTITY EXTRACTION (aggregate)
   - Extract companies, products, people
   - Identify markets and campaigns
   - Build relationship graph

See 10_llm_task_manifest.json for prompt templates and expected outputs.
""")


if __name__ == "__main__":
    main()

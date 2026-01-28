#!/usr/bin/env python3
"""
Example script demonstrating the QBR Intelligence pipeline.

This script shows how to:
1. Initialize the database
2. Process a QBR document (extract + enhance with LLM)
3. Query the processed data using the agent interface

Usage:
    python run_pipeline.py disney.pptx
    python run_pipeline.py disney.pptx --no-llm  # Skip LLM enhancement
    python run_pipeline.py --query-only          # Query existing data only
    python run_pipeline.py disney.pptx --max-slides 4
    python run_pipeline.py disney.pptx --slide-range 3-6

Environment variables (can be set in .env file):
    OPENAI_API_KEY      - OpenAI API key (required for LLM enhancement)
    ANTHROPIC_API_KEY   - Anthropic API key (alternative to OpenAI)
    LLM_MODEL           - Model to use (default: openai/gpt-4o-mini)
    DATABASE_URL        - Database connection string
    KREUZBERG_EMBEDDINGS_PRESET - Kreuzberg embedding preset (fast, balanced, quality, multilingual)
    KREUZBERG_EMBEDDINGS_ENABLED - Enable/disable embeddings (true/false)
"""

import argparse
import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from qbr_intelligence import QBRProcessor, QBRQueryInterface, init_db
from qbr_intelligence.db.models import Document
from qbr_intelligence.pipeline.embeddings import EmbeddingSettings
from qbr_intelligence.query.tools import (
    get_document_summary,
    get_key_insights,
    get_metrics_by_category,
    get_top_metrics,
    list_documents,
)

# Load environment variables from .env file
load_dotenv()


async def process_document(
    file_path: str,
    database_url: str,
    run_llm_enhancement: bool = True,
    export_outputs: bool = False,
    output_dir: str | None = None,
    override_existing: bool = False,
    slide_range: tuple[int, int] | None = None,
    max_slides: int | None = None,
) -> int:
    """
    Process a QBR document through the full pipeline.

    Returns the document ID.
    """
    print(f"\n{'=' * 60}")
    print("QBR Intelligence Pipeline")
    print(f"{'=' * 60}")

    # Initialize database
    print("\n[1/4] Initializing database...")
    # Convert async URL to sync URL for the processor (which uses sync SQLAlchemy)
    sync_database_url = database_url.replace("+aiosqlite", "").replace("+asyncpg", "")
    print(f"      Database ready: {database_url}")

    if override_existing:
        removed = await _delete_existing_documents(database_url, Path(file_path).name)
        if removed:
            print(f"      Removed {removed} existing document(s) for override.")

    # Check LLM config if enhancement enabled
    llm_model = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")
    if run_llm_enhancement:
        print("\n[2/4] LLM Enhancement enabled")
        print(f"      Model: {llm_model}")
        print("      LiteLLM picks up provider-specific env vars:")
        print("      - Databricks: DATABRICKS_API_KEY, DATABRICKS_API_BASE")
        print("      - OpenAI: OPENAI_API_KEY")
        print("      - Anthropic: ANTHROPIC_API_KEY")
        print("      - OpenRouter: OPENROUTER_API_KEY")
    else:
        print("\n[2/4] Skipping LLM enhancement (--no-llm)")

    embedding_settings = EmbeddingSettings.from_env()
    if embedding_settings.enabled:
        print("\n[2/4] Kreuzberg embeddings enabled")
        print(f"      Preset: {embedding_settings.preset}")
        print(f"      Normalize: {embedding_settings.normalize}")
        print(f"      Batch size: {embedding_settings.batch_size}")
    else:
        print("\n[2/4] Kreuzberg embeddings disabled")

    # Process document
    print(f"\n[3/4] Processing document: {file_path}")
    processor = QBRProcessor(database_url=sync_database_url, llm_model=llm_model)

    # Note: processor methods are sync (uses sync SQLAlchemy internally)
    # process_document returns the document ID
    start_dt = datetime.now(timezone.utc)
    start_time = time.perf_counter()
    doc_id = processor.process_document(
        file_path=file_path,
        run_llm_enhancement=run_llm_enhancement,
        export_outputs=export_outputs,
        output_dir=output_dir,
        slide_range=slide_range,
        max_slides=max_slides,
    )
    elapsed_seconds = time.perf_counter() - start_time
    end_dt = datetime.now(timezone.utc)

    timing_base_dir = Path(output_dir) if output_dir else Path("extraction_output")
    safe_stem = Path(file_path).stem.replace(" ", "_")
    timing_target_dir = timing_base_dir / safe_stem
    timing_target_dir.mkdir(parents=True, exist_ok=True)
    timing_path = timing_target_dir / f"pipeline_timing_{safe_stem}_{start_dt:%Y%m%d_%H%M%S}.json"
    timing_payload = {
        "file_path": str(Path(file_path).absolute()),
        "document_id": doc_id,
        "started_at": start_dt.isoformat(),
        "finished_at": end_dt.isoformat(),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "llm_enhancement": run_llm_enhancement,
        "llm_model": llm_model,
        "export_outputs": export_outputs,
        "output_dir": str(timing_target_dir),
    }
    timing_path.write_text(json.dumps(timing_payload, indent=2), encoding="utf-8")
    print(f"      Timing saved: {timing_path}")

    print(f"\n[4/4] Processing complete. Document ID: {doc_id}")

    return doc_id


async def query_demo(database_url: str, document_id: int | None = None):
    """
    Demonstrate querying capabilities using the agent interface.
    """
    print(f"\n{'=' * 60}")
    print("Query Interface Demo")
    print(f"{'=' * 60}")

    engine = await init_db(database_url)

    async with AsyncSession(engine) as session:
        # List all documents
        print("\n[Query 1] List all documents:")
        result = await list_documents(session)
        for doc in result.get("documents", [])[:5]:
            print(f"  - [{doc['id']}] {doc['filename']} ({doc['status']})")

        if not document_id and result.get("documents"):
            document_id = result["documents"][0]["id"]

        if not document_id:
            print("\nNo documents found. Process a document first.")
            await engine.dispose()
            return

        print(f"\n[Query 2] Document Summary (ID: {document_id}):")
        summary = await get_document_summary(session, document_id)
        if summary and not summary.get("error"):
            print(f"  Client: {summary.get('client_name')}")
            print(f"  Period: {summary.get('period')}")
            if summary.get("executive_summary"):
                print(f"  Summary: {summary['executive_summary'][:200]}...")
            if summary.get("recommendations"):
                print(f"  Recommendations: {len(summary['recommendations'])} items")

        print(f"\n[Query 3] Metrics by Category (ID: {document_id}):")
        metrics = await get_metrics_by_category(session, document_id)
        for category, items in metrics.get("categories", {}).items():
            print(f"  {category}: {len(items)} metrics")

        print(f"\n[Query 4] Top Metrics (ID: {document_id}):")
        top = await get_top_metrics(session, document_id, limit=5)
        for m in top.get("top_metrics", []):
            print(f"  - {m['name']}: {m['raw_value']} ({m.get('trend', 'N/A')})")

        print(f"\n[Query 5] Key Insights (ID: {document_id}):")
        insights = await get_key_insights(session, document_id)
        if insights.get("key_wins"):
            print(f"  Key Wins: {len(insights['key_wins'])} items")
        if insights.get("action_items"):
            print(f"  Action Items: {len(insights['action_items'])} items")

        # Also show raw interface usage
        print("\n[Query 6] Using QBRQueryInterface directly:")
        interface = QBRQueryInterface(session)
        doc = await interface.get_document(document_id)
        if doc:
            print(f"  Slides: {doc['slide_count']}")
            print(f"  Stats: {doc.get('stats', {})}")

    await engine.dispose()


async def export_data(database_url: str, document_id: int, output_dir: str):
    """
    Export all data for a document to JSON files.
    """
    print(f"\n{'=' * 60}")
    print("Exporting Data to JSON")
    print(f"{'=' * 60}")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    engine = await init_db(database_url)

    async with AsyncSession(engine) as session:
        interface = QBRQueryInterface(session)

        # Export document details
        print("\nExporting document details...")
        doc = await interface.get_document(document_id)
        if doc:
            with open(output_path / "document.json", "w") as f:
                json.dump(doc, f, indent=2, default=str)

        # Export metrics
        print("Exporting metrics...")
        metrics = await interface.get_metrics_by_category(document_id)
        with open(output_path / "metrics_by_category.json", "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        # Export slides by type
        print("Exporting slides...")
        slides = await interface.get_slides_by_type(document_id)
        with open(output_path / "slides_by_type.json", "w") as f:
            json.dump(slides, f, indent=2, default=str)

        # Export entities
        print("Exporting entities...")
        entities = await interface.get_entities(document_id=document_id)
        with open(output_path / "entities.json", "w") as f:
            json.dump(entities, f, indent=2, default=str)

        # Export charts
        print("Exporting charts...")
        charts = await interface.get_charts_summary(document_id)
        with open(output_path / "charts.json", "w") as f:
            json.dump(charts, f, indent=2, default=str)

        # Export insights
        print("Exporting insights...")
        insights = await interface.get_insights_summary(document_id)
        with open(output_path / "insights.json", "w") as f:
            json.dump(insights, f, indent=2, default=str)

        # Export keywords
        print("Exporting keywords...")
        keywords = await interface.get_keywords(document_id)
        with open(output_path / "keywords.json", "w") as f:
            json.dump(keywords, f, indent=2, default=str)

    await engine.dispose()
    print(f"\nData exported to: {output_path.absolute()}")


async def _delete_existing_documents(database_url: str, filename: str) -> int:
    engine = await init_db(database_url)
    removed = 0
    async with AsyncSession(engine) as session:
        result = await session.execute(select(Document).where(Document.filename == filename))
        documents = result.scalars().all()
        for document in documents:
            await session.delete(document)
        removed = len(documents)
        if removed:
            await session.commit()
    await engine.dispose()
    return removed


def _parse_slide_range(raw: str) -> tuple[int, int]:
    value = raw.strip()
    if "-" in value:
        parts = value.split("-", 1)
    elif ":" in value:
        parts = value.split(":", 1)
    else:
        num = int(value)
        return (num, num)
    start = int(parts[0].strip())
    end = int(parts[1].strip())
    if start > end:
        raise ValueError("Slide range start must be <= end.")
    return (start, end)


def main():
    parser = argparse.ArgumentParser(
        description="QBR Intelligence Pipeline - Process and query QBR documents"
    )
    parser.add_argument(
        "file",
        nargs="?",
        help="Path to PPTX file to process",
    )
    parser.add_argument(
        "--folder",
        metavar="DIR",
        help="Process all PPTX files in a folder (top-level only)",
    )
    default_db = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///qbr_intelligence.db")
    parser.add_argument(
        "--db",
        default=default_db,
        help=f"Database URL (default: {default_db})",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM enhancement",
    )
    parser.add_argument(
        "--query-only",
        action="store_true",
        help="Only run query demo (skip processing)",
    )
    parser.add_argument(
        "--document-id",
        type=int,
        help="Document ID for queries (uses latest if not specified)",
    )
    parser.add_argument(
        "--export",
        metavar="DIR",
        help="Export document data to JSON files in the specified directory",
    )
    parser.add_argument(
        "--export-extraction",
        action="store_true",
        help="Write extraction artifacts to the extraction output directory",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Delete existing documents with the same filename before processing",
    )
    parser.add_argument(
        "--extraction-output-dir",
        metavar="DIR",
        help="Override the extraction output directory (default: extraction_output)",
    )
    parser.add_argument(
        "--slide-range",
        metavar="RANGE",
        help="Process only slides in the given range (e.g., 1-4 or 3:7)",
    )
    parser.add_argument(
        "--max-slides",
        type=int,
        help="Process only the first N slides",
    )

    args = parser.parse_args()

    async def run():
        doc_id = args.document_id

        if args.file and args.folder:
            raise ValueError("Use either a single file or --folder, not both.")
        if args.slide_range and args.max_slides:
            raise ValueError("Use either --slide-range or --max-slides, not both.")
        slide_range = _parse_slide_range(args.slide_range) if args.slide_range else None
        max_slides = args.max_slides
        if max_slides is not None and max_slides <= 0:
            raise ValueError("--max-slides must be a positive integer.")

        # Process document if provided
        if args.folder and not args.query_only:
            folder = Path(args.folder)
            if not folder.is_absolute() and not folder.exists():
                candidate = Path(__file__).resolve().parent / folder
                if candidate.exists():
                    folder = candidate
            if not folder.exists():
                raise ValueError(f"Folder not found: {folder}")
            if not folder.is_dir():
                raise ValueError(f"Not a folder: {folder}")
            pptx_files = sorted(folder.glob("*.pptx"))
            if not pptx_files:
                print(f"No PPTX files found in {folder}")
            for pptx_path in pptx_files:
                doc_id = await process_document(
                    file_path=str(pptx_path),
                    database_url=args.db,
                    run_llm_enhancement=not args.no_llm,
                    export_outputs=args.export_extraction,
                    output_dir=args.extraction_output_dir,
                    override_existing=args.override,
                    slide_range=slide_range,
                    max_slides=max_slides,
                )
        elif args.file and not args.query_only:
            doc_id = await process_document(
                file_path=args.file,
                database_url=args.db,
                run_llm_enhancement=not args.no_llm,
                export_outputs=args.export_extraction,
                output_dir=args.extraction_output_dir,
                override_existing=args.override,
                slide_range=slide_range,
                max_slides=max_slides,
            )

        # Run query demo
        await query_demo(args.db, doc_id)

        # Export if requested
        if args.export and doc_id:
            await export_data(args.db, doc_id, args.export)

    asyncio.run(run())


if __name__ == "__main__":
    main()

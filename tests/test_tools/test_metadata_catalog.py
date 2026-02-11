from __future__ import annotations

import pytest

from ai_agent_qbr.models import MetadataCatalogArgs
from ai_agent_qbr.tools.metadata_catalog import metadata_catalog
from qbr_intelligence.metric_qa import MetricQueryEngine


@pytest.mark.asyncio
async def test_metadata_catalog_access_scope(metric_db, dummy_ctx):
    engine = MetricQueryEngine(database_url=metric_db)
    dummy_ctx.tool_context["metric_query_engine"] = engine

    result = await metadata_catalog(
        MetadataCatalogArgs(question="What access do we have in this database?"),
        dummy_ctx,
    )

    assert "Metadata catalog ready" in result.summary_text
    assert result.metadata.get("sections", {}).get("access", {}).get("available_key_objects")
    assert len(result.suggested_queries) >= 10


@pytest.mark.asyncio
async def test_metadata_catalog_documents_with_clients(metric_db, dummy_ctx):
    engine = MetricQueryEngine(database_url=metric_db)
    dummy_ctx.tool_context["metric_query_engine"] = engine

    result = await metadata_catalog(
        MetadataCatalogArgs(question="Which documents do we have with clients and regions?", limit=10),
        dummy_ctx,
    )

    sections = result.metadata.get("sections", {})
    docs_section = sections.get("documents", {})

    assert docs_section.get("available") is True
    assert docs_section.get("documents")
    assert docs_section.get("documents_by_client")
    clients = {row["client"] for row in docs_section.get("documents_by_client", [])}
    assert "Nike" in clients


@pytest.mark.asyncio
async def test_metadata_catalog_metric_catalog_inventory(metric_db, dummy_ctx):
    engine = MetricQueryEngine(database_url=metric_db)
    dummy_ctx.tool_context["metric_query_engine"] = engine

    result = await metadata_catalog(
        MetadataCatalogArgs(question="Which catalog metrics and aliases do we have?", limit=10),
        dummy_ctx,
    )

    sections = result.metadata.get("sections", {})
    metrics_section = sections.get("metrics", {})

    assert metrics_section.get("available") is True
    assert metrics_section.get("catalog")
    assert metrics_section.get("aliases")
    slugs = {row["slug"] for row in metrics_section.get("catalog", [])}
    assert "cost_per_acquisition" in slugs


@pytest.mark.asyncio
async def test_metadata_catalog_graceful_without_engine(dummy_ctx):
    result = await metadata_catalog(
        MetadataCatalogArgs(question="Which clients do we have?"),
        dummy_ctx,
    )

    assert "metric_query_engine is missing" in result.summary_text
    assert result.metadata.get("available") is False

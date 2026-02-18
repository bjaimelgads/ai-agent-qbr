"""
Query interface for QBR data retrieval.

Provides a comprehensive interface for agents to query QBR documents,
metrics, entities, and content with faceted filtering support.
"""

from datetime import datetime
from typing import Any

from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from qbr_intelligence.db.models import (
    Chart,
    Chunk,
    Client,
    Document,
    DocumentFacet,
    DocumentStatus,
    Entity,
    EntityType,
    Facet,
    FacetValue,
    Keyword,
    Metric,
    MetricCategory,
    Slide,
    SlideEntity,
    SlideType,
)
from qbr_intelligence.schemas.queries import (
    DocumentFilter,
    EntityFilter,
    MetricFilter,
    QueryRequest,
    QueryResponse,
    SlideFilter,
)


class QBRQueryInterface:
    """
    Comprehensive query interface for QBR data.

    Designed to be used by AI agents to answer questions about QBR documents.
    Supports:
    - Document-level queries
    - Metric analysis and filtering
    - Entity search and relationship traversal
    - Faceted filtering across all dimensions
    - Full-text search with RAG chunks
    - Cross-document aggregations
    """

    def __init__(self, session: AsyncSession):
        """Initialize with database session."""
        self.session = session

    # =========================================================================
    # Document Queries
    # =========================================================================

    async def list_documents(
        self,
        status: DocumentStatus | None = None,
        client_name: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """
        List documents with optional filtering.

        Args:
            status: Filter by document status
            client_name: Filter by client name (partial match)
            date_from: Filter documents created after this date
            date_to: Filter documents created before this date
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of document summaries
        """
        query = select(Document, Client)

        conditions = []
        if status:
            conditions.append(Document.status == status)
        if client_name:
            conditions.append(Client.name.ilike(f"%{client_name}%"))
        if date_from:
            conditions.append(Document.created_at >= date_from)
        if date_to:
            conditions.append(Document.created_at <= date_to)

        if conditions:
            query = query.outerjoin(Client).where(and_(*conditions))
        else:
            query = query.outerjoin(Client)

        query = query.order_by(Document.created_at.desc()).limit(limit).offset(offset)

        result = await self.session.execute(query)
        rows = result.all()

        return [
            {
                "id": doc.id,
                "filename": doc.filename,
                "client_name": client.name if client else None,
                "period": doc.period,
                "status": doc.status,
                "slide_count": doc.slide_count,
                "executive_summary": doc.executive_summary,
                "created_at": doc.created_at.isoformat() if doc.created_at else None,
            }
            for doc, client in rows
        ]

    async def get_document(self, document_id: int) -> dict[str, Any] | None:
        """
        Get full document details including sections and stats.

        Args:
            document_id: Document ID

        Returns:
            Full document data or None if not found
        """
        query = (
            select(Document)
            .options(
                selectinload(Document.sections),
                selectinload(Document.facets).selectinload(DocumentFacet.facet_value).selectinload(FacetValue.facet),
                selectinload(Document.client),
            )
            .where(Document.id == document_id)
        )

        result = await self.session.execute(query)
        doc = result.scalar_one_or_none()

        if not doc:
            return None

        # Get aggregate stats
        metric_count = await self.session.scalar(
            select(func.count(Metric.id))
            .join(Slide, Slide.id == Metric.slide_id)
            .where(Slide.document_id == document_id)
        )
        entity_count = await self.session.scalar(
            select(func.count(Entity.id)).where(Entity.document_id == document_id)
        )
        chart_count = await self.session.scalar(
            select(func.count(Chart.id))
            .join(Slide)
            .where(Slide.document_id == document_id)
        )

        return {
            "id": doc.id,
            "filename": doc.filename,
            "file_path": doc.file_path,
            "client_name": doc.client_name,
            "period": doc.period,
            "status": doc.status,
            "slide_count": doc.slide_count,
            "executive_summary": doc.executive_summary,
            "key_wins": doc.key_wins,
            "areas_for_improvement": doc.areas_for_improvement,
            "recommendations": doc.recommendations,
            "sections": [
                {
                    "id": s.id,
                    "name": s.name,
                    "description": s.description,
                    "start_slide": s.start_slide,
                    "end_slide": s.end_slide,
                }
                for s in doc.sections
            ],
            "facets": [
                {
                    "facet": df.facet_value.facet.name if df.facet_value and df.facet_value.facet else None,
                    "value": df.facet_value.value if df.facet_value else None,
                }
                for df in doc.facets
            ],
            "stats": {
                "metric_count": metric_count,
                "entity_count": entity_count,
                "chart_count": chart_count,
            },
            "created_at": doc.created_at.isoformat() if doc.created_at else None,
        }

    async def get_document_summary(self, document_id: int) -> dict[str, Any] | None:
        """
        Get executive summary and key insights for a document.

        Args:
            document_id: Document ID

        Returns:
            Summary data including wins, improvements, recommendations
        """
        query = select(Document).options(selectinload(Document.client)).where(
            Document.id == document_id
        )
        result = await self.session.execute(query)
        doc = result.scalar_one_or_none()

        if not doc:
            return None

        # Get top metrics
        top_metrics = await self.get_top_metrics(document_id, limit=5)

        # Get action items from slides
        action_items = await self._get_action_items(document_id)

        return {
            "document_id": doc.id,
            "client_name": doc.client_name,
            "period": doc.period,
            "executive_summary": doc.executive_summary,
            "key_wins": doc.key_wins,
            "areas_for_improvement": doc.areas_for_improvement,
            "recommendations": doc.recommendations,
            "top_metrics": top_metrics,
            "action_items": action_items,
        }

    # =========================================================================
    # Metric Queries
    # =========================================================================

    async def get_metrics(
        self,
        document_id: int | None = None,
        category: MetricCategory | None = None,
        name_contains: str | None = None,
        min_value: float | None = None,
        max_value: float | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Query metrics with flexible filtering.

        Args:
            document_id: Filter by document
            category: Filter by metric category
            name_contains: Filter by metric name (partial match)
            min_value: Minimum normalized value
            max_value: Maximum normalized value
            limit: Maximum results

        Returns:
            List of metrics with context
        """
        query = select(Metric, Slide.slide_number).outerjoin(
            Slide, Slide.id == Metric.slide_id
        )

        conditions = []
        if document_id:
            conditions.append(Slide.document_id == document_id)
        if category:
            conditions.append(Metric.category == category)
        if name_contains:
            conditions.append(Metric.name.ilike(f"%{name_contains}%"))
        if min_value is not None:
            conditions.append(Metric.normalized_value >= min_value)
        if max_value is not None:
            conditions.append(Metric.normalized_value <= max_value)

        if conditions:
            query = query.where(and_(*conditions))

        query = query.order_by(Metric.normalized_value.desc().nullslast()).limit(limit)

        result = await self.session.execute(query)
        rows = result.all()

        return [
            {
                "id": m.id,
                "document_id": m.document_id,
                "slide_id": m.slide_id,
                "slide_number": slide_number,
                "name": m.name,
                "raw_value": m.raw_value,
                "normalized_value": m.normalized_value,
                "unit": m.unit,
                "category": m.category if m.category else None,
            }
            for m, slide_number in rows
        ]

    async def get_metrics_by_category(
        self, document_id: int
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Get all metrics for a document grouped by category.

        Args:
            document_id: Document ID

        Returns:
            Dictionary mapping category names to lists of metrics
        """
        query = (
            select(Metric, Slide.slide_number)
            .join(Slide, Slide.id == Metric.slide_id)
            .where(Slide.document_id == document_id)
            .order_by(Metric.category, Metric.normalized_value.desc().nullslast())
        )

        result = await self.session.execute(query)
        rows = result.all()

        grouped: dict[str, list[dict[str, Any]]] = {}
        for m, slide_number in rows:
            cat = m.category if m.category else "uncategorized"
            if cat not in grouped:
                grouped[cat] = []
            grouped[cat].append({
                "id": m.id,
                "name": m.name,
                "raw_value": m.raw_value,
                "normalized_value": m.normalized_value,
                "unit": m.unit,
                "slide_id": m.slide_id,
                "slide_number": slide_number,
            })

        return grouped

    async def get_top_metrics(
        self, document_id: int, limit: int = 10
    ) -> list[dict[str, Any]]:
        """
        Get the highest-valued metrics for a document.

        Args:
            document_id: Document ID
            limit: Number of top metrics to return

        Returns:
            List of metrics ordered by normalized value
        """
        query = (
            select(Metric, Slide.slide_number)
            .join(Slide, Slide.id == Metric.slide_id)
            .where(Slide.document_id == document_id)
            .order_by(Metric.normalized_value.desc().nullslast())
            .limit(limit)
        )

        result = await self.session.execute(query)
        rows = result.all()

        return [
            {
                "name": m.name,
                "raw_value": m.raw_value,
                "normalized_value": m.normalized_value,
                "unit": m.unit,
                "category": m.category if m.category else None,
                "slide_id": m.slide_id,
                "slide_number": slide_number,
            }
            for m, slide_number in rows
        ]

    async def compare_metrics_across_documents(
        self,
        metric_name: str,
        document_ids: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Compare a specific metric across multiple documents.

        Args:
            metric_name: Name of the metric to compare
            document_ids: Optional list of document IDs to compare

        Returns:
            List of metric values across documents with context
        """
        query = (
            select(Metric, Client.name, Document.period)
            .join(Slide, Slide.id == Metric.slide_id)
            .join(Document, Document.id == Slide.document_id)
            .outerjoin(Client)
            .where(Metric.name.ilike(f"%{metric_name}%"))
        )

        if document_ids:
            query = query.where(Slide.document_id.in_(document_ids))

        query = query.order_by(Document.created_at.desc())

        result = await self.session.execute(query)
        rows = result.all()

        return [
            {
                "document_id": m.document_id,
                "client_name": client_name,
                "period": period,
                "metric_name": m.name,
                "raw_value": m.raw_value,
                "normalized_value": m.normalized_value,
            }
            for m, client_name, period in rows
        ]

    # =========================================================================
    # Slide Queries
    # =========================================================================

    async def get_slides(
        self,
        document_id: int,
        slide_type: SlideType | None = None,
        section_id: int | None = None,
        has_charts: bool | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get slides with optional filtering.

        Args:
            document_id: Document ID
            slide_type: Filter by slide type
            section_id: Filter by section
            has_charts: Filter for slides with/without charts

        Returns:
            List of slides with summary info
        """
        query = select(Slide).where(Slide.document_id == document_id)

        if slide_type:
            query = query.where(Slide.slide_type == slide_type)
        if section_id:
            query = query.where(Slide.section_id == section_id)

        query = query.order_by(Slide.slide_number)

        result = await self.session.execute(query)
        slides = result.scalars().all()

        # If filtering by charts, do a secondary query
        if has_charts is not None:
            slide_ids = [s.id for s in slides]
            chart_query = (
                select(Chart.slide_id)
                .where(Chart.slide_id.in_(slide_ids))
                .distinct()
            )
            chart_result = await self.session.execute(chart_query)
            slides_with_charts = {row[0] for row in chart_result.all()}

            if has_charts:
                slides = [s for s in slides if s.id in slides_with_charts]
            else:
                slides = [s for s in slides if s.id not in slides_with_charts]

        return [
            {
                "id": s.id,
                "slide_number": s.slide_number,
                "google_slide_id": s.google_slide_id,
                "slide_type": s.slide_type if s.slide_type else None,
                "title": s.title,
                "key_message": s.key_message,
                "insights": s.insights,
                "action_items": s.action_items,
            }
            for s in slides
        ]

    async def get_slides_by_type(
        self, document_id: int
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Get slides grouped by type.

        Args:
            document_id: Document ID

        Returns:
            Dictionary mapping slide types to lists of slides
        """
        slides = await self.get_slides(document_id)

        grouped: dict[str, list[dict[str, Any]]] = {}
        for slide in slides:
            slide_type = slide.get("slide_type") or "unknown"
            if slide_type not in grouped:
                grouped[slide_type] = []
            grouped[slide_type].append(slide)

        return grouped

    async def get_slide_detail(self, slide_id: int) -> dict[str, Any] | None:
        """
        Get full details for a specific slide.

        Args:
            slide_id: Slide ID

        Returns:
            Full slide data including charts, entities, and raw text
        """
        query = (
            select(Slide)
            .options(
                selectinload(Slide.charts),
                selectinload(Slide.images),
                selectinload(Slide.slide_entities).selectinload(SlideEntity.entity),
            )
            .where(Slide.id == slide_id)
        )

        result = await self.session.execute(query)
        slide = result.scalar_one_or_none()

        if not slide:
            return None

        return {
            "id": slide.id,
            "document_id": slide.document_id,
            "slide_number": slide.slide_number,
            "google_slide_id": slide.google_slide_id,
            "slide_type": slide.slide_type if slide.slide_type else None,
            "title": slide.title,
            "raw_text": slide.raw_text,
            "speaker_notes": slide.speaker_notes,
            "key_message": slide.key_message,
            "insights": slide.insights,
            "action_items": slide.action_items,
            "charts": [
                {
                    "id": c.id,
                    "chart_type": c.chart_type,
                    "title": c.title,
                    "data_series": c.data_series,
                    "insights": c.insights,
                }
                for c in slide.charts
            ],
            "images": [
                {
                    "id": img.id,
                    "image_type": img.image_type,
                    "description": img.description,
                }
                for img in slide.images
            ],
            "entities": [
                {
                    "name": se.entity.name,
                    "type": se.entity.entity_type if se.entity.entity_type else None,
                    "mention_context": se.mention_context,
                }
                for se in slide.slide_entities
            ],
        }

    # =========================================================================
    # Entity Queries
    # =========================================================================

    async def get_entities(
        self,
        document_id: int | None = None,
        entity_type: EntityType | None = None,
        name_contains: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Query entities with filtering.

        Args:
            document_id: Filter by document
            entity_type: Filter by entity type
            name_contains: Filter by name (partial match)
            limit: Maximum results

        Returns:
            List of entities
        """
        query = select(Entity)

        conditions = []
        if document_id:
            conditions.append(Entity.document_id == document_id)
        if entity_type:
            conditions.append(Entity.entity_type == entity_type)
        if name_contains:
            conditions.append(Entity.name.ilike(f"%{name_contains}%"))

        if conditions:
            query = query.where(and_(*conditions))

        query = query.order_by(Entity.mention_count.desc().nullslast()).limit(limit)

        result = await self.session.execute(query)
        entities = result.scalars().all()

        return [
            {
                "id": e.id,
                "document_id": e.document_id,
                "name": e.name,
                "entity_type": e.entity_type if e.entity_type else None,
                "normalized_name": e.normalized_name,
                "description": e.description,
                "mention_count": e.mention_count,
            }
            for e in entities
        ]

    async def get_entity_relationships(
        self, entity_id: int
    ) -> dict[str, Any]:
        """
        Get an entity with its relationships.

        Args:
            entity_id: Entity ID

        Returns:
            Entity data with related entities
        """
        from qbr_intelligence.db.models import EntityRelation

        # Get entity
        entity_query = select(Entity).where(Entity.id == entity_id)
        result = await self.session.execute(entity_query)
        entity = result.scalar_one_or_none()

        if not entity:
            return {}

        # Get relationships where this entity is source
        outgoing_query = (
            select(EntityRelation, Entity)
            .join(Entity, EntityRelation.target_entity_id == Entity.id)
            .where(EntityRelation.source_entity_id == entity_id)
        )
        outgoing_result = await self.session.execute(outgoing_query)
        outgoing = outgoing_result.all()

        # Get relationships where this entity is target
        incoming_query = (
            select(EntityRelation, Entity)
            .join(Entity, EntityRelation.source_entity_id == Entity.id)
            .where(EntityRelation.target_entity_id == entity_id)
        )
        incoming_result = await self.session.execute(incoming_query)
        incoming = incoming_result.all()

        return {
            "entity": {
                "id": entity.id,
                "name": entity.name,
                "entity_type": entity.entity_type if entity.entity_type else None,
                "description": entity.description,
            },
            "outgoing_relationships": [
                {
                    "relationship_type": rel.relationship_type,
                    "target": {
                        "id": target.id,
                        "name": target.name,
                        "type": target.entity_type if target.entity_type else None,
                    },
                }
                for rel, target in outgoing
            ],
            "incoming_relationships": [
                {
                    "relationship_type": rel.relationship_type,
                    "source": {
                        "id": source.id,
                        "name": source.name,
                        "type": source.entity_type if source.entity_type else None,
                    },
                }
                for rel, source in incoming
            ],
        }

    # =========================================================================
    # Facet Queries
    # =========================================================================

    async def get_available_facets(self) -> list[dict[str, Any]]:
        """
        Get all available facets and their values.

        Returns:
            List of facets with their possible values
        """
        query = (
            select(Facet)
            .options(selectinload(Facet.values))
            .order_by(Facet.name)
        )

        result = await self.session.execute(query)
        facets = result.scalars().all()

        return [
            {
                "id": f.id,
                "name": f.name,
                "description": f.description,
                "values": [
                    {"id": v.id, "value": v.value, "description": v.description}
                    for v in f.values
                ],
            }
            for f in facets
        ]

    async def filter_documents_by_facets(
        self,
        facet_filters: list[dict[str, Any]],
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Filter documents by facet values.

        Args:
            facet_filters: List of {facet_name: str, values: list[str]}
            limit: Maximum results

        Returns:
            Documents matching all facet criteria
        """
        # Build subquery for each facet filter
        base_query = select(Document.id)

        for ff in facet_filters:
            facet_name = ff.get("facet_name")
            values = ff.get("values", [])

            if not facet_name or not values:
                continue

            subquery = (
                select(DocumentFacet.document_id)
                .join(Facet)
                .join(FacetValue)
                .where(
                    and_(
                        Facet.name == facet_name,
                        FacetValue.value.in_(values),
                    )
                )
            )

            base_query = base_query.where(Document.id.in_(subquery))

        # Get matching documents
        base_query = base_query.limit(limit)
        result = await self.session.execute(base_query)
        doc_ids = [row[0] for row in result.all()]

        if not doc_ids:
            return []

        # Fetch full document data
        docs_query = (
            select(Document)
            .where(Document.id.in_(doc_ids))
            .order_by(Document.created_at.desc())
        )
        docs_result = await self.session.execute(docs_query)
        documents = docs_result.scalars().all()

        return [
            {
                "id": doc.id,
                "filename": doc.filename,
                "client_name": doc.client_name,
                "period": doc.period,
                "status": doc.status,
                "executive_summary": doc.executive_summary,
            }
            for doc in documents
        ]

    # =========================================================================
    # Search Queries
    # =========================================================================

    async def search_content(
        self,
        query_text: str,
        document_id: int | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Search document content using chunks.

        Args:
            query_text: Search query
            document_id: Optional document filter
            limit: Maximum results

        Returns:
            Matching chunks with context
        """
        # Simple text search (in production, use embeddings + vector search)
        search_terms = query_text.lower().split()

        query = select(Chunk)

        conditions = []
        for term in search_terms:
            conditions.append(Chunk.content.ilike(f"%{term}%"))

        if conditions:
            query = query.where(or_(*conditions))

        if document_id:
            query = query.where(Chunk.document_id == document_id)

        query = query.limit(limit)

        result = await self.session.execute(query)
        chunks = result.scalars().all()

        return [
            {
                "id": c.id,
                "document_id": c.document_id,
                "content": c.content[:500] + "..." if len(c.content) > 500 else c.content,
                "metadata": c.metadata,
                "slide_number": c.metadata.get("slide_number") if c.metadata else None,
            }
            for c in chunks
        ]

    async def get_keywords(
        self,
        document_id: int,
        min_score: float = 0.0,
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        """
        Get keywords for a document.

        Args:
            document_id: Document ID
            min_score: Minimum keyword score
            limit: Maximum results

        Returns:
            List of keywords with scores
        """
        query = (
            select(Keyword)
            .where(Keyword.document_id == document_id)
            .where(Keyword.score >= min_score)
            .order_by(Keyword.score.desc())
            .limit(limit)
        )

        result = await self.session.execute(query)
        keywords = result.scalars().all()

        return [
            {
                "keyword": k.keyword,
                "score": k.score,
            }
            for k in keywords
        ]

    # =========================================================================
    # Aggregation Queries
    # =========================================================================

    async def get_insights_summary(
        self, document_id: int
    ) -> dict[str, Any]:
        """
        Get aggregated insights from all slides.

        Args:
            document_id: Document ID

        Returns:
            Aggregated insights, action items, and recommendations
        """
        slides = await self.get_slides(document_id)

        all_insights = []
        all_action_items = []

        for slide in slides:
            if slide.get("insights"):
                if isinstance(slide["insights"], list):
                    all_insights.extend(slide["insights"])
                else:
                    all_insights.append(slide["insights"])
            if slide.get("action_items"):
                if isinstance(slide["action_items"], list):
                    all_action_items.extend(slide["action_items"])
                else:
                    all_action_items.append(slide["action_items"])

        # Get document-level recommendations
        doc = await self.get_document(document_id)

        return {
            "document_id": document_id,
            "insights": all_insights,
            "action_items": all_action_items,
            "recommendations": doc.get("recommendations") if doc else None,
            "key_wins": doc.get("key_wins") if doc else None,
            "areas_for_improvement": doc.get("areas_for_improvement") if doc else None,
        }

    async def get_charts_summary(
        self, document_id: int
    ) -> list[dict[str, Any]]:
        """
        Get all charts from a document with their data.

        Args:
            document_id: Document ID

        Returns:
            List of charts with reconstructed data
        """
        query = (
            select(Chart, Slide.slide_number)
            .join(Slide)
            .where(Slide.document_id == document_id)
            .order_by(Slide.slide_number)
        )

        result = await self.session.execute(query)
        rows = result.all()

        return [
            {
                "id": chart.id,
                "slide_number": slide_number,
                "chart_type": chart.chart_type,
                "title": chart.title,
                "data_series": chart.data_series,
                "data_table": chart.data_table,
                "insights": chart.insights,
            }
            for chart, slide_number in rows
        ]

    # =========================================================================
    # Helper Methods
    # =========================================================================

    async def _get_action_items(
        self, document_id: int
    ) -> list[dict[str, Any]]:
        """Get action items from all slides."""
        query = (
            select(Slide.slide_number, Slide.action_items)
            .where(Slide.document_id == document_id)
            .where(Slide.action_items.isnot(None))
            .order_by(Slide.slide_number)
        )

        result = await self.session.execute(query)
        rows = result.all()

        action_items = []
        for slide_number, items in rows:
            if isinstance(items, list):
                for item in items:
                    action_items.append({
                        "slide_number": slide_number,
                        "action": item,
                    })
            elif items:
                action_items.append({
                    "slide_number": slide_number,
                    "action": items,
                })

        return action_items

    # =========================================================================
    # Universal Query Interface
    # =========================================================================

    async def execute_query(
        self, request: QueryRequest
    ) -> QueryResponse:
        """
        Execute a universal query based on the query type.

        This is the main entry point for agent queries.

        Args:
            request: QueryRequest with query_type and filters

        Returns:
            QueryResponse with results
        """
        results = []
        total_count = 0

        try:
            if request.query_type == "documents":
                doc_filter = request.document_filter or DocumentFilter()
                results = await self.list_documents(
                    status=DocumentStatus(doc_filter.status) if doc_filter.status else None,
                    client_name=doc_filter.client_name,
                    date_from=doc_filter.date_from,
                    date_to=doc_filter.date_to,
                    limit=request.limit,
                    offset=request.offset,
                )
                total_count = len(results)

            elif request.query_type == "document_detail":
                if request.document_filter and request.document_filter.document_ids:
                    doc = await self.get_document(request.document_filter.document_ids[0])
                    results = [doc] if doc else []
                    total_count = 1 if doc else 0

            elif request.query_type == "metrics":
                metric_filter = request.metric_filter or MetricFilter()
                doc_id = None
                if request.document_filter and request.document_filter.document_ids:
                    doc_id = request.document_filter.document_ids[0]
                results = await self.get_metrics(
                    document_id=doc_id,
                    category=MetricCategory(metric_filter.category) if metric_filter.category else None,
                    name_contains=metric_filter.name_contains,
                    limit=request.limit,
                )
                total_count = len(results)

            elif request.query_type == "slides":
                slide_filter = request.slide_filter or SlideFilter()
                doc_id = None
                if request.document_filter and request.document_filter.document_ids:
                    doc_id = request.document_filter.document_ids[0]
                if doc_id:
                    results = await self.get_slides(
                        document_id=doc_id,
                        slide_type=SlideType(slide_filter.slide_type) if slide_filter.slide_type else None,
                        section_id=slide_filter.section_id,
                        has_charts=slide_filter.has_charts,
                    )
                    total_count = len(results)

            elif request.query_type == "entities":
                entity_filter = request.entity_filter or EntityFilter()
                doc_id = None
                if request.document_filter and request.document_filter.document_ids:
                    doc_id = request.document_filter.document_ids[0]
                results = await self.get_entities(
                    document_id=doc_id,
                    entity_type=EntityType(entity_filter.entity_type) if entity_filter.entity_type else None,
                    name_contains=entity_filter.name_contains,
                    limit=request.limit,
                )
                total_count = len(results)

            elif request.query_type == "search":
                if request.search_query:
                    doc_id = None
                    if request.document_filter and request.document_filter.document_ids:
                        doc_id = request.document_filter.document_ids[0]
                    results = await self.search_content(
                        query_text=request.search_query,
                        document_id=doc_id,
                        limit=request.limit,
                    )
                    total_count = len(results)

            elif request.query_type == "facets":
                results = await self.get_available_facets()
                total_count = len(results)

            return QueryResponse(
                results=results,
                total_count=total_count,
                query_type=request.query_type,
            )

        except Exception as e:
            return QueryResponse(
                results=[],
                total_count=0,
                query_type=request.query_type,
                error=str(e),
            )

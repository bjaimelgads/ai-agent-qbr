"""Metric extraction pipeline orchestrating parsing, linking, and scoring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from qbr_intelligence.metrics.adjudicator import AdjudicationRequest, LLMAdjudicator
from qbr_intelligence.metrics.candidates import (
    build_alias_patterns,
    build_candidates,
    extract_qualifiers,
)
from qbr_intelligence.metrics.catalog import build_metric_catalog, catalog_by_id
from qbr_intelligence.metrics.linking import link_table_candidates, link_text_candidates
from qbr_intelligence.metrics.models import (
    DeckContent,
    ExtractionDebug,
    LinkCandidate,
    MetricExtraction,
    MetricCatalogEntry,
    ValueCandidate,
    merge_qualifiers,
)
from qbr_intelligence.metrics.parsing import compute_deck_hash, parse_pptx_deck
from qbr_intelligence.metrics.resolver import resolve_metrics
from qbr_intelligence.metrics.scoring import ScoringWeights, score_link


@dataclass(frozen=True)
class PipelineConfig:
    low_confidence_threshold: float = 0.36
    tie_margin: float = 0.08
    adjudicate_max_candidates: int = 3


class MetricExtractionPipeline:
    def __init__(
        self,
        *,
        config: PipelineConfig | None = None,
        weights: ScoringWeights | None = None,
        adjudicator: LLMAdjudicator | None = None,
        catalog_entries: list[MetricCatalogEntry] | None = None,
    ) -> None:
        self._config = config or PipelineConfig()
        self._weights = weights or ScoringWeights()
        self._adjudicator = adjudicator
        self._catalog_entries = catalog_entries or build_metric_catalog()
        self._catalog_map = catalog_by_id(self._catalog_entries)
        self._alias_patterns = build_alias_patterns(self._catalog_entries)

    def extract_from_pptx(
        self, path: str | Path
    ) -> tuple[list[MetricExtraction], ExtractionDebug]:
        deck = parse_pptx_deck(path)
        deck_hash = compute_deck_hash(path)
        return self.extract_from_deck(deck, deck_hash=deck_hash)

    def extract_from_deck(
        self, deck: DeckContent, *, deck_hash: str | None = None
    ) -> tuple[list[MetricExtraction], ExtractionDebug]:
        labels: list[LabelCandidate] = []
        values: list[ValueCandidate] = []
        for slide in deck.slides:
            for block in slide.text_blocks:
                bundle = build_candidates(
                    slide_index=slide.slide_index,
                    source_type=block.source_type,
                    block_id=block.block_id,
                    text=block.text,
                    alias_patterns=self._alias_patterns,
                )
                labels.extend(bundle.labels)
                values.extend(bundle.values)
            for block in slide.notes_blocks:
                bundle = build_candidates(
                    slide_index=slide.slide_index,
                    source_type=block.source_type,
                    block_id=block.block_id,
                    text=block.text,
                    alias_patterns=self._alias_patterns,
                )
                labels.extend(bundle.labels)
                values.extend(bundle.values)
            for table in slide.tables:
                for cell in table.cells:
                    bundle = build_candidates(
                        slide_index=slide.slide_index,
                        source_type="table_cell",
                        block_id=cell.cell_id,
                        text=cell.text,
                        alias_patterns=self._alias_patterns,
                    )
                    labels.extend(bundle.labels)
                    values.extend(bundle.values)

        links: list[LinkCandidate] = []
        links.extend(link_text_candidates(labels, values))
        tables = [table for slide in deck.slides for table in slide.tables]
        links.extend(link_table_candidates(tables, labels, values))

        scored_links = [
            LinkCandidate(
                label=link.label,
                value=link.value,
                relation=link.relation,
                features=link.features,
                score=score_link(link, weights=self._weights, catalog_map=self._catalog_map),
            )
            for link in links
        ]

        selections = self._select_links(scored_links, deck_hash=deck_hash)
        metrics = [self._link_to_metric(deck, link) for link in selections]
        metrics = resolve_metrics(metrics, catalog_map=self._catalog_map)
        metrics = sorted(
            metrics,
            key=lambda m: (
                m.slide_index,
                m.metric_id,
                m.raw_value_text,
                m.label_text,
            ),
        )

        debug = ExtractionDebug(
            labels=tuple(labels),
            values=tuple(values),
            links=tuple(scored_links),
            notes=tuple(self._adjudicator_notes()),
        )
        return metrics, debug

    def _adjudicator_notes(self) -> list[str]:
        if not self._adjudicator or not self._adjudicator.enabled:
            return []
        return [
            f"adjudicator_total_requests={self._adjudicator.total_requests}",
            f"adjudicator_cache_hits={self._adjudicator.cache_hits}",
            f"adjudicator_llm_calls={self._adjudicator.llm_calls}",
            f"adjudicator_parse_failures={self._adjudicator.parse_failures}",
            f"adjudicator_empty_responses={self._adjudicator.empty_responses}",
        ]

    def _select_links(
        self, links: list[LinkCandidate], *, deck_hash: str | None
    ) -> list[LinkCandidate]:
        by_label: dict[tuple, list[LinkCandidate]] = {}
        for link in links:
            key = (link.label.slide_index, link.label.metric_id, link.label.block_id, link.label.span)
            by_label.setdefault(key, []).append(link)

        selections: list[LinkCandidate] = []
        for _, candidates in by_label.items():
            if not candidates:
                continue
            candidates_sorted = sorted(candidates, key=lambda item: item.score, reverse=True)
            best = candidates_sorted[0]
            if self._should_adjudicate(candidates_sorted) and self._adjudicator and self._adjudicator.enabled:
                adjudicated = self._adjudicate(candidates_sorted, deck_hash)
                if adjudicated:
                    best = adjudicated
            selections.append(best)
        return selections

    def _should_adjudicate(self, candidates: list[LinkCandidate]) -> bool:
        if not candidates:
            return False
        best = candidates[0]
        if best.score < self._config.low_confidence_threshold:
            return True
        if len(candidates) > 1 and (best.score - candidates[1].score) <= self._config.tie_margin:
            return True
        if best.label.metric_id in {"reach", "unique_reach"}:
            return True
        return False

    def _adjudicate(
        self, candidates: list[LinkCandidate], deck_hash: str | None
    ) -> LinkCandidate | None:
        if not self._adjudicator or not self._adjudicator.enabled:
            return None
        top = candidates[: self._config.adjudicate_max_candidates]
        request = AdjudicationRequest(
            deck_hash=deck_hash or "",
            slide_index=top[0].label.slide_index,
            metric_id=top[0].label.metric_id,
            metric_name=top[0].label.metric_name,
            label_text=top[0].label.label_text,
            context_snippet=top[0].label.context_text[:160],
            candidates=[
                {
                    "raw_value_text": cand.value.raw_value_text,
                    "normalized_value": cand.value.normalized_value,
                    "unit": cand.value.unit,
                    "snippet": cand.value.context_text[:160],
                }
                for cand in top
            ],
        )
        result = self._adjudicator.adjudicate(request)
        if not result or result.chosen_index < 0:
            return None
        if result.chosen_index >= len(top):
            return None
        chosen = top[result.chosen_index]
        return LinkCandidate(
            label=chosen.label,
            value=ValueCandidate(
                raw_value_text=chosen.value.raw_value_text,
                normalized_value=result.normalized_value
                if result.normalized_value is not None
                else chosen.value.normalized_value,
                unit=result.unit or chosen.value.unit,
                scale=chosen.value.scale,
                slide_index=chosen.value.slide_index,
                source_type=chosen.value.source_type,
                block_id=chosen.value.block_id,
                span=chosen.value.span,
                context_text=chosen.value.context_text,
                qualifiers=chosen.value.qualifiers,
                is_range=chosen.value.is_range,
            ),
            relation="llm_adjudicated",
            features=chosen.features,
            score=chosen.score,
        )

    def _link_to_metric(self, deck: DeckContent, link: LinkCandidate) -> MetricExtraction:
        label = link.label
        value = link.value
        qualifiers = merge_qualifiers(
            extract_qualifiers(label.context_text), value.qualifiers
        )
        provenance = {
            "source_type": value.source_type,
            "shape_id_or_cell_id": value.block_id,
            "snippet": value.context_text[:240],
        }
        return MetricExtraction(
            deck_id=deck.deck_id,
            slide_index=label.slide_index,
            metric_id=label.metric_id,
            metric_name=label.metric_name,
            value=value.normalized_value,
            unit=value.unit,
            scale=value.scale,
            raw_value_text=value.raw_value_text,
            label_text=label.label_text,
            qualifiers=qualifiers,
            confidence=link.score,
            extraction_method=link.relation,
            provenance=provenance,
        )

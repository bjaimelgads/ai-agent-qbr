"""
DSPY modules for QBR LLM enhancement.

These modules use DSPY 3.1 for structured output generation,
ensuring consistent and validated LLM responses.
"""

import json
from typing import Any

import dspy

from qbr_intelligence.schemas.llm_outputs import (
    ChartReconstructionOutput,
    EntityExtractionOutput,
    ExecutiveSummaryOutput,
    ImageAnalysisOutput,
    MetricDeduplicationOutput,
    MetricNormalizationOutput,
    MetricRefinementOutput,
    RefinedMetric,
    SlideAnalysisOutput,
)


# =============================================================================
# Slide Analysis Module
# =============================================================================


class SlideAnalysisSignature(dspy.Signature):
    """
    Analyze a QBR presentation slide to extract structured information.

    Given the raw text content and optional speaker notes from a slide,
    classify the slide type, extract key messages, metrics, and insights.
    """

    slide_content: str = dspy.InputField(
        desc="Raw text content extracted from the slide"
    )
    speaker_notes: str = dspy.InputField(
        desc="Speaker notes for this slide (may be empty)"
    )
    slide_number: int = dspy.InputField(desc="The slide number in the presentation")

    analysis: SlideAnalysisOutput = dspy.OutputField(
        desc="Structured analysis of the slide content"
    )


class SlideAnalyzer(dspy.Module):
    """
    DSPY module for analyzing individual slides.

    Uses chain-of-thought reasoning to:
    1. Classify the slide type
    2. Extract the key message
    3. Identify metrics with their values
    4. Generate insights and action items
    """

    def __init__(self, lm: dspy.LM | None = None):
        super().__init__()
        self.lm = lm
        self.analyze = dspy.ChainOfThought(SlideAnalysisSignature)

    def forward(
        self, slide_content: str, speaker_notes: str = "", slide_number: int = 0
    ) -> SlideAnalysisOutput:
        """Analyze a slide and return structured output."""
        with dspy.settings.context(lm=self.lm):
            result = self.analyze(
                slide_content=slide_content,
                speaker_notes=speaker_notes or "",
                slide_number=slide_number,
            )
        return result.analysis


# =============================================================================
# Metric Normalization Module
# =============================================================================


class MetricNormalizationSignature(dspy.Signature):
    """
    Normalize and categorize extracted metrics from a QBR presentation.

    Given a list of raw metrics with their context, identify what each
    metric measures, normalize values, categorize them, and assess their
    business significance.
    """

    metrics_json: str = dspy.InputField(
        desc="JSON array of raw metrics with value, type, and context"
    )
    document_context: str = dspy.InputField(
        desc="Context about the document (client, industry, time period)"
    )

    normalized: MetricNormalizationOutput = dspy.OutputField(
        desc="Normalized and categorized metrics with business insights"
    )


class MetricNormalizer(dspy.Module):
    """
    DSPY module for normalizing and categorizing metrics.

    Processes batches of metrics to:
    1. Identify metric names and types
    2. Normalize values to standard units
    3. Categorize by business function
    4. Identify trends and comparisons
    5. Assess business significance
    """

    def __init__(self, lm: dspy.LM | None = None):
        super().__init__()
        self.lm = lm
        self.normalize = dspy.ChainOfThought(MetricNormalizationSignature)

    def forward(
        self, metrics: list[dict], document_context: str = ""
    ) -> MetricNormalizationOutput:
        """Normalize a batch of metrics."""
        metrics_json = json.dumps(metrics, indent=2)
        with dspy.settings.context(lm=self.lm):
            result = self.normalize(
                metrics_json=metrics_json,
                document_context=document_context or "Business quarterly review",
            )
        return result.normalized


# =============================================================================
# Metric Deduplication Module
# =============================================================================


class MetricDeduplicationSignature(dspy.Signature):
    """
    Identify duplicate metrics that should be removed.

    Given a JSON array of metric candidates (with ids, names, values, units, and context),
    return the list of ids that should be removed as duplicates.
    """

    metrics_json: str = dspy.InputField(
        desc="JSON array of metric candidates with ids, names, values, units, contexts"
    )

    deduped: MetricDeduplicationOutput = dspy.OutputField(
        desc="List of metric ids to remove as duplicates"
    )


class MetricDeduplicator(dspy.Module):
    """DSPY module for deduplicating extracted metric candidates."""

    def __init__(self, lm: dspy.LM | None = None):
        super().__init__()
        self.lm = lm
        self.dedupe = dspy.ChainOfThought(MetricDeduplicationSignature)

    def forward(self, metrics: list[dict]) -> MetricDeduplicationOutput:
        metrics_json = json.dumps(metrics, indent=2)
        with dspy.settings.context(lm=self.lm):
            result = self.dedupe(metrics_json=metrics_json)
        return result.deduped


# =============================================================================
# Metric Refinement Module
# =============================================================================


class MetricRefinementSignature(dspy.Signature):
    """
    Refine per-slide metrics by selecting true KPI values and deltas.

    Given slide text and candidate metrics, return a clean list of metrics
    with primary values and optional deltas/baselines.
    """

    slide_text: str = dspy.InputField(desc="Full slide text")
    speaker_notes: str = dspy.InputField(desc="Speaker notes for the slide (may be empty)")
    candidates_json: str = dspy.InputField(
        desc="JSON array of candidate metrics with ids, names, values, units, and context"
    )
    metric_dictionary_json: str = dspy.InputField(
        desc="JSON array of allowed metrics with names, units, and formulas"
    )

    refined: MetricRefinementOutput = dspy.OutputField(
        desc="Refined metrics for the slide"
    )


class MetricRefiner(dspy.Module):
    """DSPY module to refine slide metrics using full context."""

    def __init__(self, lm: dspy.LM | None = None):
        super().__init__()
        self.lm = lm
        self.refine = dspy.ChainOfThought(MetricRefinementSignature)

    def forward(
        self,
        slide_text: str,
        speaker_notes: str,
        candidates: list[dict],
        metric_dictionary: list[dict],
    ) -> MetricRefinementOutput:
        payload_candidates = json.dumps(candidates, indent=2)
        payload_dictionary = json.dumps(metric_dictionary, indent=2)
        with dspy.settings.context(lm=self.lm):
            result = self.refine(
                slide_text=slide_text,
                speaker_notes=speaker_notes or "",
                candidates_json=payload_candidates,
                metric_dictionary_json=payload_dictionary,
            )
        return result.refined


# =============================================================================
# Chart Reconstruction Module
# =============================================================================


class ChartReconstructionSignature(dspy.Signature):
    """
    Reconstruct chart data from text elements extracted from a presentation.

    Given raw text elements that appear to represent chart data,
    determine the chart type and reconstruct the underlying data
    structure with labels, values, and insights.
    """

    raw_elements: str = dspy.InputField(
        desc="JSON array of text elements extracted from the chart area"
    )
    slide_context: str = dspy.InputField(
        desc="Context from the slide (title, surrounding text)"
    )

    chart: ChartReconstructionOutput = dspy.OutputField(
        desc="Reconstructed chart with data series and insights"
    )


class ChartReconstructor(dspy.Module):
    """
    DSPY module for reconstructing charts from text elements.

    Analyzes patterns in extracted text to:
    1. Determine chart type (bar, line, table, etc.)
    2. Identify data series and labels
    3. Reconstruct data table
    4. Extract visual insights
    """

    def __init__(self, lm: dspy.LM | None = None):
        super().__init__()
        self.lm = lm
        self.reconstruct = dspy.ChainOfThought(ChartReconstructionSignature)

    def forward(
        self, raw_elements: list[str], slide_context: str = ""
    ) -> ChartReconstructionOutput:
        """Reconstruct a chart from text elements."""
        elements_json = json.dumps(raw_elements, indent=2)
        with dspy.settings.context(lm=self.lm):
            result = self.reconstruct(
                raw_elements=elements_json,
                slide_context=slide_context or "QBR presentation slide",
            )
        return result.chart


# =============================================================================
# Executive Summary Module
# =============================================================================


class ExecutiveSummarySignature(dspy.Signature):
    """
    Generate an executive summary from a QBR presentation.

    Given the full text content of a QBR presentation, generate a
    comprehensive executive summary highlighting key wins, areas
    for improvement, and strategic recommendations.
    """

    full_content: str = dspy.InputField(
        desc="Full text content of the QBR presentation"
    )
    document_metadata: str = dspy.InputField(
        desc="Metadata about the document (client, period, etc.)"
    )

    summary: ExecutiveSummaryOutput = dspy.OutputField(
        desc="Executive summary with wins, improvements, and recommendations"
    )


class ExecutiveSummarizer(dspy.Module):
    """
    DSPY module for generating executive summaries.

    Analyzes the full presentation to:
    1. Generate a concise executive summary
    2. Identify key wins and achievements
    3. Highlight areas needing improvement
    4. Provide strategic recommendations
    5. Outline concrete next steps
    """

    def __init__(self, lm: dspy.LM | None = None):
        super().__init__()
        self.lm = lm
        self.summarize = dspy.ChainOfThought(ExecutiveSummarySignature)

    def forward(
        self, full_content: str, document_metadata: str = ""
    ) -> ExecutiveSummaryOutput:
        """Generate executive summary from full content."""
        # Truncate if too long (keep first and last parts)
        max_chars = 50000
        if len(full_content) > max_chars:
            half = max_chars // 2
            full_content = (
                full_content[:half]
                + "\n\n[... content truncated ...]\n\n"
                + full_content[-half:]
            )

        with dspy.settings.context(lm=self.lm):
            result = self.summarize(
                full_content=full_content,
                document_metadata=document_metadata or "QBR Presentation",
            )
        return result.summary


# =============================================================================
# Entity Extraction Module
# =============================================================================


class EntityExtractionSignature(dspy.Signature):
    """
    Extract named entities and relationships from a QBR presentation.

    Identify companies, brands, products, people, markets, campaigns,
    and dates mentioned in the presentation, along with relationships
    between entities.
    """

    content: str = dspy.InputField(
        desc="Text content to extract entities from"
    )
    context: str = dspy.InputField(
        desc="Context about the document (client, industry)"
    )

    entities: EntityExtractionOutput = dspy.OutputField(
        desc="Extracted entities with types and relationships"
    )


class EntityExtractor(dspy.Module):
    """
    DSPY module for named entity recognition and relationship extraction.

    Extracts:
    1. Company/brand names
    2. Product/service names
    3. People (speakers, stakeholders)
    4. Markets/regions
    5. Campaign names
    6. Dates/time periods
    7. Relationships between entities
    """

    def __init__(self, lm: dspy.LM | None = None):
        super().__init__()
        self.lm = lm
        self.extract = dspy.ChainOfThought(EntityExtractionSignature)

    def forward(
        self, content: str, context: str = ""
    ) -> EntityExtractionOutput:
        """Extract entities from content."""
        # Truncate if too long
        max_chars = 30000
        if len(content) > max_chars:
            content = content[:max_chars] + "\n[... truncated ...]"

        with dspy.settings.context(lm=self.lm):
            result = self.extract(
                content=content,
                context=context or "Business quarterly review presentation",
            )
        return result.entities


# =============================================================================
# Image Analysis Module (for future use with vision models)
# =============================================================================


class ImageAnalysisSignature(dspy.Signature):
    """
    Analyze an image from a QBR presentation.

    Determine the image type, extract text if present, and if the
    image is a chart, reconstruct the underlying data.
    """

    image_description: str = dspy.InputField(
        desc="Description or OCR text from the image"
    )
    slide_context: str = dspy.InputField(
        desc="Context from the slide containing this image"
    )

    analysis: ImageAnalysisOutput = dspy.OutputField(
        desc="Image analysis with type, text, and chart data if applicable"
    )


class ImageAnalyzer(dspy.Module):
    """
    DSPY module for image analysis.

    Note: Currently works with text descriptions/OCR.
    Future versions can integrate with vision models.
    """

    def __init__(self, lm: dspy.LM | None = None):
        super().__init__()
        self.lm = lm
        self.analyze = dspy.ChainOfThought(ImageAnalysisSignature)

    def forward(
        self, image_description: str, slide_context: str = ""
    ) -> ImageAnalysisOutput:
        """Analyze an image based on its description."""
        with dspy.settings.context(lm=self.lm):
            result = self.analyze(
                image_description=image_description,
                slide_context=slide_context or "QBR presentation",
            )
        return result.analysis


# =============================================================================
# Section Detector Module
# =============================================================================


class SectionDetectionSignature(dspy.Signature):
    """
    Detect logical sections in a QBR presentation.

    Given slide titles and content summaries, identify logical
    groupings/sections and their hierarchy.
    """

    slides_summary: str = dspy.InputField(
        desc="JSON array of slide numbers, titles, and brief content summaries"
    )

    sections: list[dict] = dspy.OutputField(
        desc="Detected sections with name, description, and slide ranges"
    )


class SectionDetector(dspy.Module):
    """
    DSPY module for detecting document sections.

    Analyzes slide flow to identify:
    1. Major sections (Performance, Recommendations, etc.)
    2. Subsections
    3. Section boundaries (start/end slides)
    """

    def __init__(self, lm: dspy.LM | None = None):
        super().__init__()
        self.lm = lm
        self.detect = dspy.ChainOfThought(SectionDetectionSignature)

    def forward(self, slides_summary: list[dict]) -> list[dict]:
        """Detect sections from slide summaries."""
        summary_json = json.dumps(slides_summary, indent=2)
        with dspy.settings.context(lm=self.lm):
            result = self.detect(slides_summary=summary_json)
        return result.sections


# =============================================================================
# Full Enhancement Pipeline
# =============================================================================


class QBREnhancementPipeline(dspy.Module):
    """
    Complete QBR enhancement pipeline.

    Orchestrates all enhancement modules to process a full QBR document:
    1. Analyze each slide
    2. Normalize metrics
    3. Reconstruct charts
    4. Extract entities
    5. Generate executive summary
    6. Detect sections
    """

    def __init__(self, lm: dspy.LM | None = None):
        super().__init__()
        self.lm = lm
        self.slide_analyzer = SlideAnalyzer(lm=lm)
        self.metric_normalizer = MetricNormalizer(lm=lm)
        self.chart_reconstructor = ChartReconstructor(lm=lm)
        self.entity_extractor = EntityExtractor(lm=lm)
        self.executive_summarizer = ExecutiveSummarizer(lm=lm)
        self.section_detector = SectionDetector(lm=lm)

    def forward(
        self,
        slides: list[dict],
        metrics: list[dict],
        charts: list[dict],
        full_content: str,
        document_context: str = "",
    ) -> dict[str, Any]:
        """
        Run the full enhancement pipeline.

        Args:
            slides: List of slide data with raw_text and speaker_notes
            metrics: List of extracted metrics
            charts: List of detected charts with raw_elements
            full_content: Full document text
            document_context: Context about the document

        Returns:
            Dictionary with all enhanced data
        """
        results = {
            "slides": [],
            "metrics": None,
            "charts": [],
            "entities": None,
            "executive_summary": None,
            "sections": [],
        }

        # Calculate total LLM calls for progress reporting
        total_calls = len(slides) + len(charts) + 4  # slides + charts + metrics + entities + summary + sections
        current_call = 0

        print(f"\n  === LLM Enhancement Plan ===")
        print(f"  Total slides to analyze: {len(slides)}")
        print(f"  Total charts to reconstruct: {len(charts)}")
        print(f"  Batch operations: 4 (metrics, entities, summary, sections)")
        print(f"  Expected LLM calls: ~{total_calls}")
        print()

        # 1. Analyze slides
        print(f"  [Step 1/6] Analyzing slides...")
        for i, slide in enumerate(slides):
            current_call += 1
            slide_num = slide.get("slide_number", i + 1)
            print(f"    Processing slide {i + 1}/{len(slides)} (slide #{slide_num}) [LLM call {current_call}/{total_calls}]")
            try:
                analysis = self.slide_analyzer(
                    slide_content=slide.get("raw_text", ""),
                    speaker_notes=slide.get("speaker_notes", ""),
                    slide_number=slide.get("slide_number", 0),
                )
                slide_type = getattr(analysis, "slide_type", "unknown")
                title = getattr(analysis, "title", "")[:50]
                print(f"      → Type: {slide_type}, Title: {title or '(none)'}")
                results["slides"].append(
                    {"slide_number": slide.get("slide_number"), "analysis": analysis}
                )
            except Exception as e:
                print(f"      → ERROR: {str(e)[:80]}")
                results["slides"].append(
                    {
                        "slide_number": slide.get("slide_number"),
                        "error": str(e),
                    }
                )
        print(f"  ✓ Slide analysis complete: {len([s for s in results['slides'] if 'analysis' in s])} succeeded, {len([s for s in results['slides'] if 'error' in s])} failed\n")

        # 2. Normalize metrics (batch)
        current_call += 1
        print(f"  [Step 2/6] Normalizing metrics (batch of {len(metrics)}) [LLM call {current_call}/{total_calls}]")
        if metrics:
            try:
                results["metrics"] = self.metric_normalizer(
                    metrics=metrics, document_context=document_context
                )
                normalized_count = len(getattr(results["metrics"], "metrics", []))
                print(f"  ✓ Metrics normalized: {normalized_count} metrics processed\n")
            except Exception as e:
                print(f"  ✗ Metrics normalization failed: {str(e)[:80]}\n")
                results["metrics"] = {"error": str(e)}
        else:
            print(f"  → No metrics to normalize\n")

        # 3. Reconstruct charts
        print(f"  [Step 3/6] Reconstructing charts...")
        for i, chart in enumerate(charts):
            current_call += 1
            chart_slide = chart.get("slide_number", "?")
            print(f"    Processing chart {i + 1}/{len(charts)} (slide #{chart_slide}) [LLM call {current_call}/{total_calls}]")
            try:
                reconstruction = self.chart_reconstructor(
                    raw_elements=chart.get("raw_elements", []),
                    slide_context=f"Slide {chart.get('slide_number', 'unknown')}",
                )
                chart_type = getattr(reconstruction, "chart_type", "unknown")
                title = getattr(reconstruction, "title", "")[:40]
                print(f"      → Type: {chart_type}, Title: {title or '(none)'}")
                results["charts"].append(
                    {
                        "slide_number": chart.get("slide_number"),
                        "reconstruction": reconstruction,
                    }
                )
            except Exception as e:
                print(f"      → ERROR: {str(e)[:80]}")
                results["charts"].append(
                    {
                        "slide_number": chart.get("slide_number"),
                        "error": str(e),
                    }
                )
        print(f"  ✓ Chart reconstruction complete: {len([c for c in results['charts'] if 'reconstruction' in c])} succeeded, {len([c for c in results['charts'] if 'error' in c])} failed\n")

        # 4. Extract entities
        current_call += 1
        print(f"  [Step 4/6] Extracting entities [LLM call {current_call}/{total_calls}]")
        try:
            results["entities"] = self.entity_extractor(
                content=full_content, context=document_context
            )
            entity_count = len(getattr(results["entities"], "entities", []))
            print(f"  ✓ Entity extraction complete: {entity_count} entities found\n")
        except Exception as e:
            print(f"  ✗ Entity extraction failed: {str(e)[:80]}\n")
            results["entities"] = {"error": str(e)}

        # 5. Generate executive summary
        current_call += 1
        print(f"  [Step 5/6] Generating executive summary [LLM call {current_call}/{total_calls}]")
        try:
            results["executive_summary"] = self.executive_summarizer(
                full_content=full_content, document_metadata=document_context
            )
            summary_preview = getattr(results["executive_summary"], "summary", "")[:100]
            print(f"  ✓ Executive summary generated: {summary_preview}...\n")
        except Exception as e:
            print(f"  ✗ Executive summary failed: {str(e)[:80]}\n")
            results["executive_summary"] = {"error": str(e)}

        # 6. Detect sections (using slide analyses)
        current_call += 1
        print(f"  [Step 6/6] Detecting document sections [LLM call {current_call}/{total_calls}]")
        if results["slides"]:
            slides_summary = []
            for s in results["slides"]:
                if "analysis" in s:
                    slides_summary.append(
                        {
                            "slide_number": s["slide_number"],
                            "title": getattr(s["analysis"], "title", None),
                            "type": getattr(s["analysis"], "slide_type", "unknown"),
                            "key_message": getattr(
                                s["analysis"], "key_message", ""
                            )[:100],
                        }
                    )
            try:
                results["sections"] = self.section_detector(
                    slides_summary=slides_summary
                )
                section_count = len(results["sections"]) if isinstance(results["sections"], list) else 0
                print(f"  ✓ Section detection complete: {section_count} sections identified\n")
            except Exception as e:
                print(f"  ✗ Section detection failed: {str(e)[:80]}\n")
                results["sections"] = {"error": str(e)}

        print(f"  === Enhancement Pipeline Complete ===")
        print(f"  Total LLM calls made: {current_call}")
        print()

        return results


# =============================================================================
# LLM Configuration Helper
# =============================================================================


def create_lm(
    model: str = "openai/gpt-4o-mini",
    max_tokens: int = 16000,
    temperature: float = 0.0,
) -> dspy.LM:
    """
    Create a DSPY LM instance.

    LiteLLM automatically picks up provider-specific env vars:
    - OpenAI: OPENAI_API_KEY
    - Anthropic: ANTHROPIC_API_KEY
    - Databricks: DATABRICKS_API_KEY, DATABRICKS_API_BASE
    - OpenRouter: OPENROUTER_API_KEY

    Args:
        model: Model identifier (e.g., "openai/gpt-4o-mini", "databricks/databricks-claude-opus-4-5")
        max_tokens: Maximum tokens for generation
        temperature: Sampling temperature

    Returns:
        dspy.LM instance

    Example:
        >>> lm = create_lm("databricks/databricks-claude-opus-4-5")
    """
    return dspy.LM(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )


# Keep for backwards compatibility
def configure_llm(
    model: str = "openai/gpt-4o-mini",
    max_tokens: int = 4000,
    temperature: float = 0.0,
):
    """Configure DSPY globally (deprecated, use create_lm instead)."""
    lm = create_lm(model=model, max_tokens=max_tokens, temperature=temperature)
    dspy.configure(lm=lm)
    return lm

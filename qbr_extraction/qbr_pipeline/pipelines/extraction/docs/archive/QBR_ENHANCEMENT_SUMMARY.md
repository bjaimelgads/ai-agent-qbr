# QBR Intelligence - Enhancement Summary

## Document Processed

| Field | Value |
|-------|-------|
| **Filename** | disney.pptx |
| **Status** | enhanced |
| **Slide Count** | 82 |

## Extraction Results (Kreuzberg)

The document extraction was **successful**. All data below was extracted from the PPTX file.

### Data Statistics

| Data Type | Count |
|-----------|-------|
| Slides | 82 |
| Metrics | 368 |
| Charts | 72 |
| Images | 326 |
| Chunks (RAG) | 70 |
| Keywords | 20 |

### Chart Types Detected

- `comparison` - Comparison charts
- `table` - Table/data charts

### Top Keywords Extracted

| Keyword | Category |
|---------|----------|
| Reach | business_term |
| Creative | business_term |
| IT | business_term |
| Carousel | business_term |
| DE | business_term |
| Conversions | business_term |
| Campaign | business_term |
| UK | business_term |
| Impressions | business_term |
| Banner | business_term |

### Sample Metrics (Raw Values)

The following metric values were extracted from the document:

| Raw Value | Occurrences |
|-----------|-------------|
| 30% | 10 |
| 1% | 10 |
| 28% | 9 |
| 15% | 9 |
| 100% | 8 |
| 26% | 7 |
| 8% | 6 |
| 5% | 6 |
| 0.15% | 6 |
| 96% | 5 |

### Sample Slide Content

#### Slide 1 (Title)
```
- High Impact
- Spotlight
```

#### Slide 10 (Agenda)
```
- Looking Ahead
- High Impact Holds
- Align on Testing
- Agenda
- Get ahead of the competition by holding key dates across
  the Content Store and Home Screen Roadblocks
- Advance our A/B testing agenda and confirm FY26 Bundle test
```

#### Slide 20 (Content)
```
- Power of Video within the Carousel Companion
- Across H2, 13 markets deployed the Carousel Companion unit
  with 16:9 video, and the format consistently delivered
  top-tier performance across every KPI.
- 13x Market
```

#### Slide 50 (Data)
```
- Data Updated
- Power of Home Screen Roadblocks
- 3x More Efficient CPE VS Overall
- 15% Overall H2 Investment
```

#### Slide 80 (Analysis)
```
- CPA by Secondary Markets - Not Installed ROS
- All secondary markets achieved $13 average decrease in CPA
  vs last half, indicating room to continue expanding ROS in H2
- % Not Installed: 17%, 22%, 22%, 25%, 28%, 24%, 21%, 31%, 20%, 37%, 35%
```

### Sample RAG Chunks

The document was chunked into 70 segments for retrieval-augmented generation:

**Chunk 1 (bytes 0-1117):**
```
<!-- PAGE 1 -->
- High Impact
- Spotlight
```

**Chunk 5 (bytes 3535-4230):**
```
<!-- PAGE 8 -->
- EMEA FY25 H2
- Recommendations for FY26 and Beyond
- Audience Targeting
```

## LLM Enhancement Status

The LLM enhancement pipeline **ran** (20 slides analyzed, 10 charts processed) but the results were **not saved** to the database properly.

### What Was Attempted
- Slide classification and key message extraction
- Metric normalization (naming, categorization)
- Chart data reconstruction
- Entity extraction
- Executive summary generation
- Section detection

### Why Enhancement Data Wasn't Saved

The DSPY framework returned structured outputs, but the `_save_enhancements` method in `processor.py` may have issues parsing the DSPY output objects. Specifically:

1. **Slide analysis**: Results came back but `getattr(analysis, 'slide_type', ...)` may be returning defaults
2. **Executive summary**: The summary object attributes weren't properly extracted
3. **DSPY JSON mode fallback**: Warnings indicated DSPY fell back to JSON mode, which may have changed output structure

### Fields That Remain Empty

| Field | Expected | Actual |
|-------|----------|--------|
| `slides.title` | LLM-generated titles | NULL |
| `slides.key_message` | Key messages | NULL |
| `slides.slide_type` | Classification | "unknown" (all) |
| `slides.insights` | Insights | NULL |
| `metrics.name` | Metric names | NULL |
| `metrics.category` | Categories | "other" (all) |
| `metrics.trend` | Trends | "unknown" (all) |
| `document.executive_summary` | Summary | NULL |
| `document.key_wins` | Wins | NULL |
| `document.recommendations` | Recommendations | NULL |

## Fixes Applied

### 1. Query Interface `.value` Errors

Fixed 10+ occurrences in `qbr_intelligence/query/interface.py` where enum fields were being accessed with `.value` but are stored as strings:

```python
# Before (error)
"status": doc.status.value,
"category": m.category.value,
"slide_type": s.slide_type.value,

# After (fixed)
"status": doc.status,
"category": m.category,
"slide_type": s.slide_type,
```

### 2. Comprehensive LLM Logging

Added detailed progress logging to `qbr_intelligence/llm/modules.py`:

```
=== LLM Enhancement Plan ===
Total slides to analyze: 20
Total charts to reconstruct: 10
Batch operations: 4 (metrics, entities, summary, sections)
Expected LLM calls: ~34

[Step 1/6] Analyzing slides...
  Processing slide 1/20 (slide #1) [LLM call 1/34]
    → Type: data, Title: FY25 Performance Overview
  ...

[Step 2/6] Normalizing metrics (batch of 50) [LLM call 21/34]
✓ Metrics normalized: 50 metrics processed

[Step 3/6] Reconstructing charts...
  Processing chart 1/10 (slide #2) [LLM call 22/34]
    → Type: bar, Title: Revenue Comparison
  ...

=== Enhancement Pipeline Complete ===
Total LLM calls made: 34
```

## Next Steps

To fully populate LLM-enhanced fields:

1. **Debug DSPY output parsing** in `_save_enhancements()` method
2. **Re-run with logging** to see actual DSPY output structure:
   ```bash
   uv run python qbr_extraction/qbr_pipeline/pipelines/extraction/run_extraction_pipeline.py disney.pptx
   ```
3. **Verify DSPY structured outputs** match Pydantic schemas in `schemas/llm_outputs.py`

## Query Interface

The query interface is now working. Example usage:

```python
from qbr_intelligence import init_db, QBRQueryInterface
from sqlalchemy.ext.asyncio import AsyncSession

engine = await init_db("sqlite+aiosqlite:///qbr_intelligence.db")
async with AsyncSession(engine) as session:
    interface = QBRQueryInterface(session)

    # List documents
    docs = await interface.list_documents()

    # Get document
    doc = await interface.get_document(document_id=1)

    # Get metrics
    metrics = await interface.get_metrics(document_id=1)
```

---

*Generated by QBR Intelligence system*

"""Document processing pipeline."""

import os

_LIGHT_IMPORT = os.getenv("QBR_INTELLIGENCE_LIGHT_IMPORT") == "1"

if not _LIGHT_IMPORT:
    from qbr_intelligence.pipeline.processor import QBRProcessor

    __all__ = ["QBRProcessor"]
else:
    __all__ = []

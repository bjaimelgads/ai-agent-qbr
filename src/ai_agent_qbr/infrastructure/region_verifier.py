"""LLM-backed region verification for QBR queries."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from ai_agent_qbr.config import Config
from ai_agent_qbr.infrastructure.databricks import resolve_workspace_token
from ai_agent_qbr.models import RegionFilterVerificationResult

_LOGGER = logging.getLogger(__name__)


@dataclass
class RegionVerifier:
    """Call a small LLM to determine regional scope for filtering."""

    config: Config
    _lm: Any | None = None

    async def verify(self, question: str) -> RegionFilterVerificationResult:
        if not question.strip():
            return RegionFilterVerificationResult(
                region_focus=None,
                needs_clarification=True,
                reason="empty question",
                confidence=None,
            )
        if self.config.use_stub_llm:
            return RegionFilterVerificationResult(
                region_focus=None,
                needs_clarification=False,
                reason="stub llm",
                confidence=None,
            )
        lm = self._lm or self._build_lm()
        if lm is None:
            return RegionFilterVerificationResult(
                region_focus=None,
                needs_clarification=False,
                reason="region verifier unavailable",
                confidence=None,
            )
        self._lm = lm

        try:
            import dspy
        except Exception:
            _LOGGER.exception("dspy missing; cannot run region verifier")
            return RegionFilterVerificationResult(
                region_focus=None,
                needs_clarification=False,
                reason="dspy missing",
                confidence=None,
            )

        class RegionFocusSignature(dspy.Signature):
            """Classify query region focus for QBR responses.

            Return region_focus as one of: US, EMEA, GLOBAL, MIXED, UNKNOWN.
            Set needs_clarification=true when the query is ambiguous or mixes regions.
            """

            question: str = dspy.InputField(
                desc="User query about QBRs."
            )
            region_focus: str = dspy.OutputField(
                desc="One of: US, EMEA, GLOBAL, MIXED, UNKNOWN."
            )
            needs_clarification: str = dspy.OutputField(
                desc="true if region is ambiguous or conflicting, else false."
            )
            reason: str = dspy.OutputField(desc="Short reason for the classification.")

        predictor = dspy.Predict(RegionFocusSignature)
        loop = asyncio.get_running_loop()

        def _run() -> Any:
            with dspy.context(lm=lm):
                return predictor(question=question)

        try:
            result = await loop.run_in_executor(None, _run)
        except Exception:
            _LOGGER.exception("Region verifier LLM call failed")
            return RegionFilterVerificationResult(
                region_focus=None,
                needs_clarification=False,
                reason="region verifier error",
                confidence=None,
            )

        region = _normalize_region(getattr(result, "region_focus", ""))
        needs_clarification = _parse_bool(getattr(result, "needs_clarification", "false"))
        reason = _clean_text(getattr(result, "reason", "")) or None

        return RegionFilterVerificationResult(
            region_focus=region,
            needs_clarification=needs_clarification,
            reason=reason,
            confidence=None,
        )

    def _build_lm(self) -> Any | None:
        if not self.config.region_verify_model_name:
            _LOGGER.warning("Region verifier model name not configured")
            return None
        resolved_host = self.config.databricks_host
        if not resolved_host and self.config.databricks_api_base:
            resolved_host = self.config.databricks_api_base.split("/serving-endpoints", 1)[0]
        if not resolved_host:
            _LOGGER.warning("Region verifier missing Databricks host")
            return None

        api_base = self.config.databricks_api_base or f"{resolved_host.rstrip('/')}/serving-endpoints"
        token = resolve_workspace_token(
            host=self.config.databricks_host,
            api_base=self.config.databricks_api_base,
            token=self.config.databricks_token,
            api_key=self.config.databricks_api_key,
            client_id=self.config.databricks_client_id,
            client_secret=self.config.databricks_client_secret,
        )
        if not token:
            _LOGGER.warning("Region verifier missing Databricks token")
            return None

        try:
            import dspy
        except Exception:
            _LOGGER.exception("dspy missing; cannot build region verifier")
            return None

        model_id = f"databricks/{self.config.region_verify_model_name}"
        max_tokens = max(64, int(self.config.region_verify_max_tokens))
        temperature = 0.0
        if "gpt-5" in self.config.region_verify_model_name.lower():
            temperature = 1.0
            if max_tokens < 16000:
                max_tokens = None
        _LOGGER.info("Creating region verifier LLM=%s", model_id)
        return dspy.LM(
            model_id,
            api_key=token,
            api_base=api_base,
            max_tokens=max_tokens,
            temperature=temperature,
            cache=False,
        )


def _normalize_region(value: str) -> str | None:
    text = _clean_text(value).lower()
    if not text:
        return None
    if "us" in text or "usa" in text or "united states" in text or "domestic" in text:
        return "us"
    if "emea" in text or "europe" in text or "eu" in text or "uk" in text:
        return "emea"
    if "global" in text or "world" in text:
        return "global"
    if "mixed" in text or "both" in text or "multiple" in text:
        return "mixed"
    return None


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = _clean_text(str(value)).lower()
    return text in {"true", "yes", "y", "1"}


def _clean_text(value: str) -> str:
    return value.strip().strip('"').strip("'")


__all__ = ["RegionVerifier"]

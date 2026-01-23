"""Guardrail wiring for the PenguiFlow planner."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse, urlunparse

from penguiflow.planner.guardrails import (
    AsyncRuleEvaluator,
    ContextSnapshotV1,
    GatewayConfig,
    GuardrailAction,
    GuardrailDecision,
    GuardrailEvent,
    GuardrailGateway,
    GuardrailSeverity,
    InjectionPatternRule,
    RetrySpec,
    RuleRegistry,
    StopSpec,
)
from penguiflow.planner.guardrails.models import RuleCost
from penguiflow.steering import InMemoryGuardInbox

from ai_agent_qbr.config import Config
from ai_agent_qbr.infrastructure.databricks import resolve_workspace_token

_LOGGER = logging.getLogger("penguiflow.guardrails")
_PLATFORM_LOGGER = logging.getLogger("uvicorn.error")

# Route configuration for the SLM Router
ROUTE_CONFIG = [
    {
        "name": "greetings_onboarding",
        "description": (
            "Greetings, small talk, onboarding, or capability questions. "
            "Includes hello/thanks or how to ask for a QBR."
        ),
    },
    {
        "name": "query_clarification",
        "description": (
            "In-scope QBR request missing key details such as advertiser, quarter/timeframe, "
            "region, channel (CTV/OTT/FAST), KPI focus, or slide reference. Ask for missing details."
        ),
    },
    {
        "name": "insight_discovery",
        "description": (
            "Requests for key insights, highlights, top drivers, anomalies, or important takeaways "
            "from a specific timeframe (e.g., FY25, Q2) or advertiser."
        ),
    },
    {
        "name": "comparative_analysis",
        "description": (
            "Comparisons across time periods, advertisers, regions, channels, or KPIs. "
            "Includes YoY/QoQ, pre/post, and benchmarking questions."
        ),
    },
    {
        "name": "metric_detail",
        "description": (
            "Requests for specific metrics, figures, or detailed breakdowns drawn from QBRs, "
            "including lists or structured summaries."
        ),
    },
    {
        "name": "partnership_updates",
        "description": (
            "Partnership highlights, investment summaries, added value, joint initiatives, "
            "and collaboration updates between LG Ads and advertisers."
        ),
    },
    {
        "name": "product_innovation",
        "description": (
            "LG product or inventory updates relevant to QBRs, including home screen formats, "
            "CTV video/branded player, content store features, and product beta launches."
        ),
    },
    {
        "name": "creative_performance",
        "description": (
            "Creative analysis and format performance (carousel, video, companion, static, 3D), "
            "including best-performing creatives and CTR/CPA/CPE comparisons."
        ),
    },
    {
        "name": "audience_segments",
        "description": (
            "Audience segment performance (installed, not installed, lapsed, active), "
            "targeting strategy impact, and suppression tactics."
        ),
    },
    {
        "name": "experiment_results",
        "description": (
            "Split tests, promo analyses, incrementality, lift studies, and "
            "experiment outcomes by market or format."
        ),
    },
    {
        "name": "market_footprint",
        "description": (
            "Market-level reach, penetration, footprint, MAU growth, and "
            "regional performance across EU5/EMEA/US or other markets."
        ),
    },
    {
        "name": "campaign_flighting",
        "description": (
            "Campaign flighting plans, schedules, and pacing across quarters or markets."
        ),
    },
    {
        "name": "roadblock_performance",
        "description": (
            "Content store or home screen roadblock performance, lift studies, and "
            "spotlight recaps by market."
        ),
    },
    {
        "name": "promo_activation",
        "description": (
            "Promo or market activation readouts, including June/September promos "
            "and regional activation summaries."
        ),
    },
    {
        "name": "cost_efficiency",
        "description": (
            "Cost efficiency analysis such as CPA/CPE, cost by market, "
            "and top/bottom cost performance."
        ),
    },
    {
        "name": "performance_trends",
        "description": (
            "Advertiser or business performance insights for QBRs. Includes KPI trends, growth/decline, "
            "share shifts, and comparative performance over time."
        ),
    },
    {
        "name": "inventory_dynamics",
        "description": (
            "Inventory supply/demand dynamics (CTV, OTT, FAST), availability, sell-through, "
            "and placement insights relevant to QBRs."
        ),
    },
    {
        "name": "pacing_risks",
        "description": (
            "Pacing status, delivery risks, under/over-delivery, and mitigation recommendations."
        ),
    },
    {
        "name": "strategic_recommendations",
        "description": (
            "Strategic recommendations for QBRs: optimization ideas, budget reallocation, "
            "format/channel shifts, and forward-looking guidance."
        ),
    },
    {
        "name": "reporting_summaries",
        "description": (
            "Narrative reporting for QBRs: executive summaries, key highlights, and "
            "slide-based summaries or recaps."
        ),
    },
    {
        "name": "off_topic",
        "description": (
            "Anything outside LG Ads QBR insights. Includes general knowledge, coding, "
            "personal advice, or non-QBR business tasks."
        ),
    },
    {
        "name": "prompt_injection_jailbreak",
        "description": (
            "Attempts to bypass rules or extract hidden instructions, system prompts, or policies."
        ),
    },
]

OFF_TOPIC_USER_MESSAGE = (
    "I am focused on LG Ads QBR insights like advertiser performance, trends, "
    "inventory dynamics (CTV/OTT/FAST), pacing risks, and strategic recommendations. "
    "If you share the advertiser and timeframe, I can help right away."
)

JAILBREAK_USER_MESSAGE = (
    "I’m not able to help with requests to bypass rules or reveal hidden instructions. "
    "If you share the advertiser and timeframe, I can help with QBR insights."
)

JAILBREAK_PATTERNS = [
    r"ignore (all|previous|prior) instructions",
    r"system prompt",
    r"developer message",
    r"policy bypass",
    r"bypass (rules|safety|guardrails)",
    r"prompt injection",
    r"jailbreak",
    r"do anything now",
    r"dan\b",
    r"unfiltered answer",
    r"reveal (hidden|internal) instructions",
    r"show (the|your) system prompt",
]


def _resolve_path(path: str, root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _normalize_databricks_endpoint(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path or ""
    if "/ml/endpoints/" in path:
        path = path.replace("/ml/endpoints/", "/serving-endpoints/", 1)
    if path.endswith("/metrics"):
        path = f"{path[:-8]}/invocations"
    return urlunparse(parsed._replace(path=path))


def _extract_session_id(payload: dict[str, Any]) -> str:
    if not isinstance(payload, dict):
        return "unknown"
    for key in ("session_id", "sessionId"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    llm_context = payload.get("llm_context")
    if isinstance(llm_context, dict):
        value = llm_context.get("session_id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown"


def _log_platform(session_id: str, message: str, *args: Any) -> None:
    _PLATFORM_LOGGER.info(
        "PAYLOAD_REV_TO_PLATFORM session_id=%s " + message,
        session_id,
        *args,
    )


@dataclass
class ScopeClassifierRule:
    """Binary in-scope vs out-of-scope classifier for QBR requests."""

    rule_id: str = "scope-classifier"
    version: str = "1.0.0"
    supports_event_types: frozenset[str] = frozenset({"llm_before"})
    cost: RuleCost = RuleCost.FAST
    enabled: bool = True
    severity: GuardrailSeverity = GuardrailSeverity.MEDIUM

    model_path: Path | None = None
    threshold: float = 0.6

    _encoder: Any = field(default=None, repr=False)
    _classifier: Any = field(default=None, repr=False)

    def _load_model(self) -> bool:
        if self._encoder is not None and self._classifier is not None:
            return True
        if self.model_path is None or not self.model_path.exists():
            _LOGGER.warning("Scope classifier model not found at %s", self.model_path)
            return False
        try:
            import joblib
            from sentence_transformers import SentenceTransformer
        except Exception:
            _LOGGER.exception("Scope classifier optional deps missing")
            return False

        payload = joblib.load(self.model_path)
        encoder_name = payload.get("encoder")
        classifier = payload.get("classifier")
        if not encoder_name or classifier is None:
            _LOGGER.warning("Scope classifier payload missing encoder or classifier")
            return False
        self._encoder = SentenceTransformer(encoder_name)
        self._classifier = classifier
        threshold = payload.get("threshold")
        if isinstance(threshold, (int, float)):
            self.threshold = float(threshold)
            _LOGGER.info("Scope classifier using payload threshold %.2f", self.threshold)
        _LOGGER.info("Scope classifier loaded from %s", self.model_path)
        return True

    async def evaluate(
        self,
        event: GuardrailEvent,
        context_snapshot: ContextSnapshotV1,
    ) -> GuardrailDecision | None:
        if not self._load_model():
            return None
        text = event.text_content
        if not text:
            _LOGGER.debug("Scope classifier skip: empty text")
            return None

        augmented = _augment_with_context(text, context_snapshot, event.payload)
        loop = asyncio.get_running_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: self._encoder.encode([augmented], normalize_embeddings=True),
        )
        proba = self._classifier.predict_proba(embedding)[0]
        in_scope_score = float(proba[1]) if len(proba) > 1 else float(proba[0])

        if in_scope_score < self.threshold:
            _LOGGER.info(
                "Scope classifier retry (score=%.2f, threshold=%.2f)",
                in_scope_score,
                self.threshold,
            )
            return GuardrailDecision(
                action=GuardrailAction.RETRY,
                rule_id=self.rule_id,
                reason=f"Out-of-scope request (score: {in_scope_score:.2f})",
                severity=self.severity,
                confidence=1.0 - in_scope_score,
                retry=RetrySpec(
                    corrective_message=(
                        "I can help with LG Ads QBR insights. "
                        "Which advertiser and timeframe should I focus on?"
                    )
                ),
                effects=("flag_trajectory",),
                classifier_result={"score": in_scope_score},
            )
        _LOGGER.debug(
            "Scope classifier allow (score=%.2f, threshold=%.2f)",
            in_scope_score,
            self.threshold,
        )
        return None


@dataclass
class JailbreakSentinelRule:
    """Call a Databricks serving endpoint to detect jailbreaks."""

    rule_id: str = "jailbreak-sentinel"
    version: str = "1.0.0"
    supports_event_types: frozenset[str] = frozenset({"llm_before"})
    cost: RuleCost = RuleCost.DEEP
    enabled: bool = True
    severity: GuardrailSeverity = GuardrailSeverity.CRITICAL

    endpoint_url: str | None = None
    token: str | None = None
    token_resolver: Callable[[], str | None] | None = None
    threshold: float = 0.6
    payload_style: str = "inputs_list"
    input_field: str | None = None

    async def evaluate(
        self,
        event: GuardrailEvent,
        context_snapshot: ContextSnapshotV1,
    ) -> GuardrailDecision | None:
        del context_snapshot
        if not self.endpoint_url:
            _LOGGER.info("Jailbreak sentinel skip: no endpoint configured")
            return None

        endpoint_url = _normalize_databricks_endpoint(self.endpoint_url)
        text = event.text_content
        if not text:
            _LOGGER.info("Jailbreak sentinel skip: empty text")
            return None

        token = self.token or os.getenv("DATABRICKS_TOKEN")
        if not token and self.token_resolver:
            token = self.token_resolver()
        if not token:
            _LOGGER.warning("Jailbreak sentinel missing token; skipping")
            return None

        payload = self._build_payload(text)
        _LOGGER.info("Jailbreak sentinel payload style=%s", self.payload_style)
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        try:
            import httpx
        except Exception:
            _LOGGER.exception("httpx missing; cannot call jailbreak sentinel")
            return None

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(endpoint_url, headers=headers, json=payload)
            if response.status_code != 200:
                _LOGGER.warning(
                    "Jailbreak sentinel request failed: %s %s",
                    response.status_code,
                    response.text[:200],
                )
                return None
            data = response.json()
        except Exception as exc:
            _LOGGER.warning("Jailbreak sentinel request error: %s", exc)
            return None

        _LOGGER.info("Jailbreak sentinel raw response keys=%s", list(data.keys()))

        predictions = data.get("predictions")
        if not isinstance(predictions, list) or not predictions:
            _LOGGER.info("Jailbreak sentinel skip: missing predictions")
            return None
        first = predictions[0]
        if not isinstance(first, dict):
            _LOGGER.info("Jailbreak sentinel skip: prediction not dict")
            return None

        label = str(first.get("label", "")).lower()
        score = first.get("score")
        try:
            score_value = float(score)
        except (TypeError, ValueError):
            _LOGGER.info("Jailbreak sentinel skip: invalid score %s", score)
            return None

        if "jailbreak" in label or "prompt_injection" in label:
            _LOGGER.info(
                "Jailbreak sentinel hit label=%s score=%.3f threshold=%.3f",
                label,
                score_value,
                self.threshold,
            )
            if score_value >= self.threshold:
                return GuardrailDecision(
                    action=GuardrailAction.STOP,
                    rule_id=self.rule_id,
                    reason=f"Jailbreak sentinel triggered (score={score_value:.2f})",
                    severity=self.severity,
                    confidence=score_value,
                    effects=("flag_trajectory", "increment_strike"),
                    classifier_result={"label": label, "score": score_value},
                    stop=StopSpec(
                        error_code="JAILBREAK_SENTINEL",
                        user_message=JAILBREAK_USER_MESSAGE,
                    ),
                )

        return None

    def _build_payload(self, text: str) -> dict[str, Any]:
        field = self.input_field or "text"
        if self.payload_style == "dataframe_split":
            return {"dataframe_split": {"columns": [field], "data": [[text]]}}
        if self.payload_style == "inputs_named":
            return {"inputs": {field: [text]}}
        return {"inputs": [text]}


@dataclass
class SLMRouterRule:
    """Route classification using a Databricks SLM endpoint."""

    rule_id: str = "slm-router"
    version: str = "1.0.0"
    supports_event_types: frozenset[str] = frozenset({"llm_before"})
    cost: RuleCost = RuleCost.FAST
    enabled: bool = True
    severity: GuardrailSeverity = GuardrailSeverity.MEDIUM

    endpoint_url: str | None = None
    token: str | None = None
    token_resolver: Callable[[], str | None] | None = None
    conversation_turns: int = 5
    route_config: list[dict[str, str]] = field(default_factory=lambda: ROUTE_CONFIG)

    async def evaluate(
        self,
        event: GuardrailEvent,
        context_snapshot: ContextSnapshotV1,
    ) -> GuardrailDecision | None:
        del context_snapshot

        if not self.endpoint_url:
            _LOGGER.info("SLM router skip: no endpoint configured")
            return None

        endpoint_url = _normalize_databricks_endpoint(self.endpoint_url)
        text = event.text_content
        if not text:
            _LOGGER.info("SLM router skip: empty text")
            return None

        token = self.token or os.getenv("DATABRICKS_TOKEN")
        if not token and self.token_resolver:
            token = self.token_resolver()
        if not token:
            _LOGGER.warning("SLM router missing token; skipping")
            return None

        conversation = self._build_conversation(text, event.payload)
        payload = self._build_payload(conversation)

        session_id = _extract_session_id(event.payload)
        _log_platform(session_id, "guardrail=slm-router stage=call messages=%d", len(conversation))
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        try:
            import httpx
        except ImportError:
            _LOGGER.exception("httpx missing; cannot call SLM router")
            return None

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(endpoint_url, headers=headers, json=payload)
            if response.status_code != 200:
                _LOGGER.warning(
                    "SLM router request failed: %s %s",
                    response.status_code,
                    response.text[:200],
                )
                _log_platform(session_id, "guardrail=slm-router stage=error status=%s", response.status_code)
                return None
            data = response.json()
        except Exception as exc:
            _LOGGER.warning("SLM router request error: %s", exc)
            _log_platform(session_id, "guardrail=slm-router stage=error error=%s", str(exc))
            return None

        _log_platform(session_id, "guardrail=slm-router stage=response payload=%s", data)

        predictions = data.get("predictions")
        if not isinstance(predictions, list) or not predictions:
            _LOGGER.info("SLM router skip: missing predictions")
            return None

        first = predictions[0]
        route_name = self._extract_route_name(first)
        if not route_name:
            _LOGGER.info("SLM router skip: could not extract route name")
            return None

        _log_platform(session_id, "guardrail=slm-router stage=route route=%s", route_name)

        if route_name == "off_topic":
            _log_platform(session_id, "guardrail=slm-router stage=decision decision=off_topic")
            return GuardrailDecision(
                action=GuardrailAction.STOP,
                rule_id=self.rule_id,
                reason="Request classified as off-topic",
                severity=self.severity,
                confidence=1.0,
                effects=("flag_trajectory",),
                classifier_result={"route": route_name},
                stop=StopSpec(
                    error_code="OFF_TOPIC",
                    user_message=OFF_TOPIC_USER_MESSAGE,
                ),
            )

        if route_name == "prompt_injection_jailbreak":
            _log_platform(session_id, "guardrail=slm-router stage=decision decision=prompt_injection_jailbreak")
            return GuardrailDecision(
                action=GuardrailAction.STOP,
                rule_id=self.rule_id,
                reason="Request classified as prompt injection/jailbreak attempt",
                severity=GuardrailSeverity.CRITICAL,
                confidence=1.0,
                effects=("flag_trajectory", "increment_strike"),
                classifier_result={"route": route_name},
                stop=StopSpec(
                    error_code="JAILBREAK_DETECTED",
                    user_message=JAILBREAK_USER_MESSAGE,
                ),
            )

        _LOGGER.debug("SLM router allow: route=%s", route_name)
        return None

    def _build_conversation(
        self,
        current_text: str,
        payload: dict[str, Any],
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []

        conversation_history = payload.get("conversation_history", [])
        if isinstance(conversation_history, list):
            max_history = self.conversation_turns - 1
            for turn in conversation_history[-max_history:]:
                if isinstance(turn, dict):
                    role = turn.get("role", "user")
                    content = turn.get("content", "")
                    if content:
                        messages.append({"role": role, "content": content})

        if not messages:
            llm_context = payload.get("llm_context", {})
            conversation_memory = {}
            if isinstance(llm_context, dict):
                conversation_memory = llm_context.get("conversation_memory", {})
            recent_turns = []
            if isinstance(conversation_memory, dict):
                recent_turns = conversation_memory.get("recent_turns", [])
            if isinstance(recent_turns, list):
                flattened: list[dict[str, str]] = []
                for turn in recent_turns:
                    if not isinstance(turn, dict):
                        continue
                    user = turn.get("user")
                    assistant = turn.get("assistant")
                    if isinstance(user, str) and user.strip():
                        flattened.append({"role": "user", "content": user.strip()})
                    if isinstance(assistant, str) and assistant.strip():
                        flattened.append({"role": "assistant", "content": assistant.strip()})
                if flattened:
                    max_history = max(1, self.conversation_turns - 1)
                    history_limit = min(max_history, 3)
                    messages.extend(flattened[-history_limit:])

        if not messages:
            last_assistant = payload.get("last_assistant")
            if isinstance(last_assistant, str) and last_assistant.strip():
                messages.append({"role": "assistant", "content": last_assistant.strip()})

        messages.append({"role": "user", "content": current_text})

        return messages

    def _build_payload(self, conversation: list[dict[str, str]]) -> dict[str, Any]:
        routes_json = json.dumps(self.route_config)
        conversation_json = json.dumps(conversation)

        return {
            "dataframe_split": {
                "columns": ["routes", "conversation"],
                "data": [[routes_json, conversation_json]],
            }
        }

    def _extract_route_name(self, prediction: Any) -> str | None:
        if isinstance(prediction, str):
            return prediction.lower()

        if isinstance(prediction, list) and prediction:
            first = prediction[0]
            if isinstance(first, str):
                return first.lower()
            if isinstance(first, dict):
                return self._extract_route_name(first)

        if not isinstance(prediction, dict):
            return None

        raw_value = prediction.get("raw")
        if isinstance(raw_value, str) and raw_value.strip():
            try:
                import ast

                parsed = ast.literal_eval(raw_value)
                if isinstance(parsed, dict) and "route" in parsed:
                    return str(parsed["route"]).lower()
            except (ValueError, SyntaxError):
                pass
            try:
                parsed = json.loads(raw_value)
                if isinstance(parsed, dict) and "route" in parsed:
                    return str(parsed["route"]).lower()
            except json.JSONDecodeError:
                pass

        route = prediction.get("route") or prediction.get("name")
        if route and str(route).lower() != "other":
            return str(route).lower()

        if route:
            return str(route).lower()

        return None


@dataclass
class HeuristicJailbreakRule:
    """Heuristic jailbreak detection using simple pattern matching."""

    rule_id: str = "heuristic-jailbreak"
    version: str = "1.0.0"
    supports_event_types: frozenset[str] = frozenset({"llm_before"})
    cost: RuleCost = RuleCost.FAST
    enabled: bool = True
    severity: GuardrailSeverity = GuardrailSeverity.CRITICAL
    patterns: list[str] = field(default_factory=lambda: list(JAILBREAK_PATTERNS))

    async def evaluate(
        self,
        event: GuardrailEvent,
        context_snapshot: ContextSnapshotV1,
    ) -> GuardrailDecision | None:
        del context_snapshot
        text = event.text_content
        if not text:
            return None
        lowered = text.lower()
        for pattern in self.patterns:
            if re.search(pattern, lowered):
                _LOGGER.info("Heuristic jailbreak hit: pattern=%s", pattern)
                return GuardrailDecision(
                    action=GuardrailAction.STOP,
                    rule_id=self.rule_id,
                    reason="Heuristic jailbreak pattern detected",
                    severity=self.severity,
                    confidence=0.9,
                    effects=("flag_trajectory", "increment_strike"),
                    classifier_result={"pattern": pattern},
                    stop=StopSpec(
                        error_code="JAILBREAK_HEURISTIC",
                        user_message=JAILBREAK_USER_MESSAGE,
                    ),
                )
        return None


def _augment_with_context(text: str, context: ContextSnapshotV1, payload: dict[str, Any]) -> str:
    parts = [f"[USER] {text}"]
    last_assistant = payload.get("last_assistant")
    if isinstance(last_assistant, str) and last_assistant.strip():
        parts.append(f"[LAST_ASSISTANT] {last_assistant.strip()}")
    task_scope = payload.get("task_scope")
    if isinstance(task_scope, str) and task_scope.strip():
        parts.append(f"[TASK_SCOPE] {task_scope.strip()}")
    if context.available_tools:
        tools = ", ".join(tool.name for tool in context.available_tools)
        parts.append(f"[AVAILABLE_TOOLS] {tools}")
    return "\n".join(parts)




def build_guardrail_gateway(config: Config) -> GuardrailGateway | None:
    if not config.guardrails_enabled:
        _LOGGER.info("Guardrails disabled; no gateway created")
        return None

    registry = RuleRegistry()
    root = Path(__file__).resolve().parents[2]
    model_path = None
    if config.guardrails_scope_model_path:
        model_path = _resolve_path(config.guardrails_scope_model_path, root)
    _LOGGER.info(
        "Guardrails enabled (mode=%s, scope_model_path=%s)",
        config.guardrails_mode,
        model_path,
    )

    def _resolve_guardrail_token() -> str | None:
        if config.guardrails_router_token:
            return config.guardrails_router_token
        token = resolve_workspace_token(
            host=config.databricks_host,
            api_base=config.databricks_api_base,
            token=config.databricks_token,
            api_key=config.databricks_api_key,
            client_id=config.databricks_client_id,
            client_secret=config.databricks_client_secret,
        )
        return token or None

    registry.register_sync(HeuristicJailbreakRule())
    if config.guardrails_jailbreak_endpoint:
        registry.register_sync(
            JailbreakSentinelRule(
                endpoint_url=config.guardrails_jailbreak_endpoint,
                token=config.guardrails_jailbreak_token,
                token_resolver=_resolve_guardrail_token,
                threshold=config.guardrails_jailbreak_threshold,
                payload_style=config.guardrails_jailbreak_payload_style,
                input_field=config.guardrails_jailbreak_input_field,
            )
        )
    registry.register_sync(
        SLMRouterRule(
            endpoint_url=config.guardrails_router_endpoint,
            token=config.guardrails_router_token,
            token_resolver=_resolve_guardrail_token,
            conversation_turns=config.guardrails_router_conversation_turns,
        )
    )
    registry.register(InjectionPatternRule())
    # ScopeClassifierRule disabled - enable if a local model is available.
    # registry.register(
    #     ScopeClassifierRule(
    #         model_path=model_path,
    #     )
    # )

    gateway = GuardrailGateway(
        registry=registry,
        guard_inbox=InMemoryGuardInbox(AsyncRuleEvaluator(registry)),
        config=GatewayConfig(
            mode=config.guardrails_mode,
            sync_timeout_ms=config.guardrails_sync_timeout_ms,
        ),
    )
    return gateway


__all__ = ["build_guardrail_gateway"]

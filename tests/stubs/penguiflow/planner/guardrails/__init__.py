"""Stub guardrails module."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GuardrailEvent:
    text_content: str | None = None


@dataclass
class ContextSnapshotV1:
    available_tools: list | None = None


@dataclass
class GatewayConfig:
    mode: str = "enforce"
    sync_timeout_ms: float = 0.0


class GuardrailAction:
    STOP = "stop"


class GuardrailSeverity:
    CRITICAL = "critical"
    MEDIUM = "medium"


@dataclass
class StopSpec:
    error_code: str
    user_message: str


@dataclass
class RetrySpec:
    pass


@dataclass
class GuardrailDecision:
    action: str
    rule_id: str
    reason: str | None = None
    severity: str | None = None
    confidence: float | None = None
    effects: tuple | None = None
    classifier_result: dict | None = None
    stop: StopSpec | None = None


class GuardrailGateway:
    def __init__(self, *args, **kwargs):
        pass


class RuleRegistry:
    def __init__(self):
        self._rules = []

    def register_sync(self, rule):
        self._rules.append(rule)

    def register(self, rule):
        self._rules.append(rule)


class AsyncRuleEvaluator:
    def __init__(self, *args, **kwargs):
        pass


class InjectionPatternRule:
    pass

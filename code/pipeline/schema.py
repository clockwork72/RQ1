"""Data schema for PoliReasoner."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Modality(Enum):
    """Deontic strength of a privacy statement."""

    PROHIBITION = 1
    OBLIGATION = 2
    COMMITMENT = 3
    PERMISSION = 4
    POSSIBILITY = 5
    HEDGED = 6
    UNSPECIFIED = 7


class ConditionType(Enum):
    """Under what circumstances the practice applies."""

    BY_DEFAULT = "by_default"
    UPON_CONSENT = "upon_consent"
    IF_OPTED_IN = "if_opted_in"
    UNLESS_OPTED_OUT = "unless_opted_out"
    WHEN_REQUIRED = "when_required"
    JURISDICTIONAL = "jurisdictional"
    UNSPECIFIED = "unspecified"


class TemporalityType(Enum):
    """Time constraints on the data practice."""

    SPECIFIC_DURATION = "specific_duration"
    UNTIL_PURPOSE = "until_purpose"
    UNTIL_ACCOUNT_DELETION = "until_account_deletion"
    INDEFINITE = "indefinite"
    UNSPECIFIED = "unspecified"


class Verdict(Enum):
    INCONSISTENT = "inconsistent"
    UNDERSPECIFIED = "underspecified"
    NON_CONFLICT = "non_conflict"


class Severity(Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


GDPR_CATEGORIES = [
    "Processing Purpose",
    "Data Categories",
    "Data Recipients",
    "Source of Data",
    "Storage Period",
    "Right to Access",
    "Right to Erase",
    "Right to Object",
    "Right to Portability",
    "Right to Restrict",
    "Withdraw Consent",
    "Provision Requirement",
    "Adequacy Decision",
    "Profiling",
    "DPO Contact",
    "Controller Contact",
    "Safeguards Copy",
    "Lodge Complaint",
]


VALID_ACTIONS = [
    "collect",
    "use",
    "share",
    "sell",
    "retain",
    "delete",
    "transfer",
    "process",
    "access_right",
    "deletion_right",
    "optout_right",
    "portability_right",
]


ACTION_SUBSUMPTION = {
    "sell": "share",
    "share": "transfer",
    "transfer": "process",
    "collect": "process",
    "use": "process",
    "retain": "process",
}


CONDITION_STRICTNESS = {
    "upon_consent": 5,
    "if_opted_in": 4,
    "unless_opted_out": 3,
    # Jurisdictional carve-outs ("EU residents must opt in", "California users may
    # opt out") are scope-limited but carry real strictness inside their scope.
    # Placed at 3 alongside unless_opted_out so Π₄ doesn't flag EU-opt-out paired
    # with a global by_default clause (gap = 2 < PI4_STRICTNESS_GAP_MIN = 3)
    # while Π₈ still fires when a jurisdictional-gated website practice is paired
    # with a by_default vendor (strictness >= PI8_WEBSITE_STRICTNESS_MIN = 3).
    # Previously defaulted to 0 via .get(...,0), which treated EU-opt-out as
    # maximally loose and silently suppressed Π₈ findings on EU-focused policies.
    "jurisdictional": 3,
    "when_required": 2,
    "by_default": 1,
    "unspecified": 0,
}


@dataclass(slots=True)
class Clause:
    """A segmented unit from a privacy policy."""

    clause_id: str
    text: str
    section_header: str = ""
    position_index: int = 0


@dataclass(slots=True)
class PPS:
    """Atomic Privacy Practice Statement."""

    id: str
    actor: str
    action: str
    modality: Modality
    data_object: str
    purpose: str
    recipient: str
    condition: ConditionType
    temporality: TemporalityType
    temporality_value: str = ""
    is_negative: bool = False
    gdpr_categories: list[str] = field(default_factory=list)
    source_text: str = ""
    source_section: str = ""
    policy_source: str = ""
    scope: str = "global"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "actor": self.actor,
            "action": self.action,
            "modality": self.modality.name,
            "data_object": self.data_object,
            "purpose": self.purpose,
            "recipient": self.recipient,
            "condition": self.condition.value,
            "temporality": self.temporality.value,
            "temporality_value": self.temporality_value,
            "is_negative": self.is_negative,
            "gdpr_categories": list(self.gdpr_categories),
            "source_text": self.source_text,
            "source_section": self.source_section,
            "policy_source": self.policy_source,
            "scope": self.scope,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PPS":
        modality_name = str(raw.get("modality", "UNSPECIFIED")).upper()
        condition_value = str(raw.get("condition", ConditionType.UNSPECIFIED.value)).lower()
        temporality_value = str(raw.get("temporality", TemporalityType.UNSPECIFIED.value)).lower()

        try:
            modality = Modality[modality_name]
        except KeyError:
            modality = Modality.UNSPECIFIED

        try:
            condition = ConditionType(condition_value)
        except ValueError:
            condition = ConditionType.UNSPECIFIED

        try:
            temporality = TemporalityType(temporality_value)
        except ValueError:
            temporality = TemporalityType.UNSPECIFIED

        return cls(
            id=str(raw.get("id", "")),
            actor=str(raw.get("actor", "FirstParty")),
            action=str(raw.get("action", "process")).lower(),
            modality=modality,
            data_object=str(raw.get("data_object", "")),
            purpose=str(raw.get("purpose", "")),
            recipient=str(raw.get("recipient", "")),
            condition=condition,
            temporality=temporality,
            temporality_value=str(raw.get("temporality_value", "")),
            is_negative=bool(raw.get("is_negative", False)),
            gdpr_categories=list(raw.get("gdpr_categories", [])),
            source_text=str(raw.get("source_text", "")),
            source_section=str(raw.get("source_section", "")),
            policy_source=str(raw.get("policy_source", "")),
            scope=str(raw.get("scope", "global")) or "global",
        )


@dataclass(slots=True)
class Inconsistency:
    """A detected inconsistency between two statements."""

    inconsistency_id: str
    pattern_id: str
    pattern_name: str
    statement_1: PPS
    statement_2: PPS
    verdict: Verdict
    severity: Severity
    explanation: str
    gdpr_categories: list[str] = field(default_factory=list)
    evidence_spans: list[str] = field(default_factory=list)
    llm_verified: bool = False
    llm_confidence: str = ""
    llm_false_alarm_category: str = "none"
    # LLM-verifier output. `llm_verdict` is the authoritative final verdict
    # used in CSVs and paper-facing outputs: one of {"inconsistent",
    # "unspecified", "non_conflict"}. The previous Stage-2 cluster
    # adjudication and combined_verdict layer were removed — the verifier
    # now returns the final label directly.
    llm_verdict: str = ""
    llm_explanation: str = ""
    neighborhood_context: dict = field(default_factory=dict)
    # Article 13/14 attribution: primary / secondary categories the pattern
    # breaches, empirical RoBERTa labels of the two clauses, and whether the
    # primary category is also absent from either side's disclosure. Set by
    # pipeline.run_pair via extractor.attribute_finding_to_gdpr; left None
    # when the finding was produced outside the pair pipeline.
    gdpr_attribution: dict | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "inconsistency_id": self.inconsistency_id,
            "pattern_id": self.pattern_id,
            "pattern_name": self.pattern_name,
            "statement_1": self.statement_1.to_dict(),
            "statement_2": self.statement_2.to_dict(),
            "verdict": self.verdict.value,
            "severity": self.severity.value,
            "explanation": self.explanation,
            "gdpr_categories": list(self.gdpr_categories),
            "evidence_spans": list(self.evidence_spans),
            "llm_verified": self.llm_verified,
            "llm_confidence": self.llm_confidence,
            "llm_false_alarm_category": self.llm_false_alarm_category,
            "llm_verdict": self.llm_verdict,
            "llm_explanation": self.llm_explanation,
        }
        if self.neighborhood_context:
            d["neighborhood_context"] = self.neighborhood_context
        if self.gdpr_attribution is not None:
            d["gdpr_attribution"] = self.gdpr_attribution
        return d

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Inconsistency":
        verdict_value = str(raw.get("verdict", Verdict.NON_CONFLICT.value))
        severity_value = str(raw.get("severity", Severity.LOW.value))

        try:
            verdict = Verdict(verdict_value)
        except ValueError:
            verdict = Verdict.NON_CONFLICT

        try:
            severity = Severity(severity_value)
        except ValueError:
            severity = Severity.LOW

        return cls(
            inconsistency_id=str(raw.get("inconsistency_id", "")),
            pattern_id=str(raw.get("pattern_id", "")),
            pattern_name=str(raw.get("pattern_name", "")),
            statement_1=PPS.from_dict(dict(raw.get("statement_1", {}))),
            statement_2=PPS.from_dict(dict(raw.get("statement_2", {}))),
            verdict=verdict,
            severity=severity,
            explanation=str(raw.get("explanation", "")),
            gdpr_categories=list(raw.get("gdpr_categories", [])),
            evidence_spans=list(raw.get("evidence_spans", [])),
            llm_verified=bool(raw.get("llm_verified", False)),
            llm_confidence=str(raw.get("llm_confidence", "")),
            llm_false_alarm_category=str(raw.get("llm_false_alarm_category", "none")),
            llm_verdict=str(raw.get("llm_verdict", "")),
            llm_explanation=str(raw.get("llm_explanation", "")),
            neighborhood_context=dict(raw.get("neighborhood_context", {})),
            gdpr_attribution=(
                dict(raw["gdpr_attribution"])
                if isinstance(raw.get("gdpr_attribution"), dict)
                else None
            ),
        )

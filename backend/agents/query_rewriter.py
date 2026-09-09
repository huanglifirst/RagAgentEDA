from __future__ import annotations

from dataclasses import dataclass
import json
import re
import unicodedata
from typing import Iterable, List, Sequence, Set

from backend.llm.client import OpenAICompatClient
_SCENES = {"qa", "task"}
_REWRITE_MODES = {"conservative", "aggressive"}
_GENERIC_TERMS: tuple[str, ...] = ()
_GENERIC_ENTITY_STOPWORDS = {"ted", "pyted", "api"}
_PASS_THROUGH_ACRONYMS = {
    "ted",
    "api",
    "eda",
    "mos",
    "mosfet",
    "nmos",
    "pmos",
    "ac",
    "dc",
    "fft",
    "lvs",
    "drc",
    "bw",
    "pm",
    "sfdr",
}
_CORE_INTENT_STOPWORDS = {
    "ted",
    "tool",
    "tools",
    "how",
    "what",
    "can",
    "could",
    "please",
    "use",
    "using",
    "with",
    "for",
    "\u5982\u4f55",
    "\u600e\u4e48",
    "\u600e\u6837",
    "\u53ef\u4ee5",
    "\u80fd\u4e0d\u80fd",
    "\u662f\u5426",
    "\u4f7f\u7528",
    "\u8fdb\u884c",
    "\u5b9e\u73b0",
    "\u5b8c\u6210",
    "\u4e00\u4e2a",
    "\u4e00\u4e0b",
    "\u63d0\u4f9b",
    "\u652f\u6301",
    "\u4ec0\u4e48",
    "\u591a\u5c11",
    "\u529f\u80fd",
    "\u65b9\u5f0f",
}
_QUESTION_FILLERS = (
    "请问",
    "帮我",
    "帮忙",
    "一下",
    "如何",
    "怎么",
    "怎样",
    "想问",
    "问下",
    "一下子",
)
_QUESTION_SUFFIXES = (
    "怎么测",
    "怎么画",
    "怎么做",
    "怎么写",
    "怎么用",
    "如何测",
    "如何画",
    "如何做",
    "如何写",
    "如何用",
    "用法",
    "示例",
    "例子",
    "教程",
    "原理",
    "方法",
    "步骤",
    "代码",
    "脚本",
    "是什么",
    "是啥",
)
_CODE_ENTITY_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\b")
_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{1,}\b")
_PATH_RE = re.compile(r"(?:[A-Za-z]:)?[/\\][^\s]+|(?:\b[\w.-]+/)+[\w.-]+")
_SLASH_ACRONYM_RE = re.compile(r"\b[A-Z]{2,}(?:/[A-Z]{2,})+\b")
_FILE_RE = re.compile(r"\b[\w.-]+\.(?:py|md|markdown|html|htm|txt|json|yaml|yml)\b", flags=re.IGNORECASE)
_NUMBER_UNIT_RE = re.compile(
    r"\b\d+(?:\.\d+)?(?:e[+-]?\d+)?\s*(?:[a-zA-Z]+|[munpfkMGT]?Hz|dB|V|mV|uV|nV|A|mA|uA|nA)?\b"
)
_CAMEL_IDENTIFIER_RE = re.compile(r"\b[A-Z][A-Za-z0-9_]{2,}\b")
_CODE_FENCE_RE = re.compile(r"^\s*```(?:\w+)?\s*|\s*```\s*$")
_ALNUM_LOWER_RE = re.compile(r"[A-Za-z0-9]")
_ZH_PHRASE_RE = re.compile(r"[\u4e00-\u9fff]{2,}")
_ZH_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
_EN_TOKEN_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_+-]*\b")
_TRAILING_EN_KEYWORDS_RE = re.compile(
    r"[\u4e00-\u9fff]\s+(?:[A-Za-z][A-Za-z0-9_+-]*\s+){1,}[A-Za-z][A-Za-z0-9_+-]*\s*[?？]?$"
)
_UNPARENTHESIZED_ALIAS_RE = re.compile(
    r"[\u4e00-\u9fff]\s+(?:MOSFET|device|ted_device|simulation|pyted|api|Monte|Carlo|Bandwidth)"
    r"(?:\s+(?:MOSFET|device|ted_device|simulation|pyted|api|Monte|Carlo|Bandwidth))*\b",
    flags=re.IGNORECASE,
)
_DISALLOWED_CROSS_TOPIC_TERMS = (
    "AC",
    "DC",
    "tran",
    "transient",
    "chan",
    "mesh",
    "\u7f51\u683c",
)
_NATURAL_QUERY_MARKERS = (
    "\u5982\u4f55",
    "\u600e\u4e48",
    "\u600e\u6837",
    "\u80fd\u5426",
    "\u662f\u5426",
    "\u7528\u6cd5",
    "\u793a\u4f8b",
    "\u5b9e\u73b0",
    "\u8fdb\u884c",
    "\u67e5\u770b",
    "\u8c03\u8bd5",
    "\u5b9a\u4f4d",
    "\u9ad8\u4eae",
    "\u9519\u8bef",
    "\u4eff\u771f",
    "\u5668\u4ef6",
    "\u6d4b",
    "\u83b7\u53d6",
    "\u914d\u7f6e",
    "\u521b\u5efa",
    "\u652f\u6301",
    "\u4e2d",
    "\u4e0e",
    "\u7684",
)


@dataclass(frozen=True)
class _GlossaryEntry:
    name: str
    aliases: tuple[str, ...]
    keywords: tuple[str, ...]


@dataclass(frozen=True)
class _CoreTerm:
    key: str
    canonical: str
    english: str
    aliases: tuple[str, ...]
    aggressive_dimensions: tuple[str, ...]


@dataclass
class QueryRewriteResult:
    original_query: str
    rewritten_query: str
    changed: bool
    strategy: str
    warning: str = ""


@dataclass
class _RecognizedIntent:
    core_intent: str = ""
    core_subjects: tuple[str, ...] = ()
    core_terms: tuple[_CoreTerm, ...] = ()
    allowed_dimensions: tuple[str, ...] = ()
    forbidden_topics: tuple[str, ...] = ()
    confidence: str = "low"
    source: str = "fallback"
    warning: str = ""


_CORE_TERMS: tuple[_CoreTerm, ...] = (
    _CoreTerm(
        key="monte_carlo",
        canonical="\u8499\u7279\u5361\u7f57\u4eff\u771f",
        english="Monte Carlo Simulation",
        aliases=(
            "\u8499\u7279\u5361\u7f57\u4eff\u771f",
            "\u8499\u7279\u5361\u6d1b\u4eff\u771f",
            "\u8499\u7279\u5361\u7f57",
            "\u8499\u7279\u5361\u6d1b",
            "monte carlo",
            "mc simulation",
        ),
        aggressive_dimensions=(
            "Monte Carlo Simulation",
            "MC simulation",
            "\u7edf\u8ba1\u4eff\u771f",
            "\u968f\u673a\u91c7\u6837",
            "sampling",
            "variation",
            "process variation",
            "corner",
            "\u91c7\u6837\u89c4\u5219",
            "\u6536\u655b\u6027\u5224\u65ad",
            "\u7ed3\u679c\u5206\u6790",
        ),
    ),
    _CoreTerm(
        key="ac_simulation",
        canonical="AC \u4eff\u771f",
        english="AC Simulation",
        aliases=("AC \u4eff\u771f", "ac\u4eff\u771f", "\u4ea4\u6d41\u4eff\u771f", "ac simulation"),
        aggressive_dimensions=(
            "AC \u4eff\u771f\u53c2\u6570\u914d\u7f6e",
            "\u9891\u7387\u626b\u63cf\u8bbe\u7f6e",
            "\u8f93\u51fa\u7ed3\u679c\u83b7\u53d6",
            "\u7ed3\u679c\u5206\u6790",
        ),
    ),
    _CoreTerm(
        key="mos",
        canonical="MOS \u7ba1",
        english="MOSFET",
        aliases=("MOS \u7ba1", "mos\u7ba1", "mosfet", "nmos", "pmos"),
        aggressive_dimensions=(
            "MOSFET \u5668\u4ef6\u5b9a\u4e49",
            "\u5173\u952e\u53c2\u6570\u914d\u7f6e",
            "\u5b9e\u4f8b\u5316\u65b9\u6cd5",
        ),
    ),
    _CoreTerm(
        key="lvs",
        canonical="LVS",
        english="Layout Versus Schematic",
        aliases=("LVS", "\u4e00\u81f4\u6027\u68c0\u67e5"),
        aggressive_dimensions=("\u9519\u8bef\u5b9a\u4f4d", "\u7248\u56fe\u4e0e\u539f\u7406\u56fe\u5bf9\u6bd4", "\u8c03\u8bd5\u6d41\u7a0b"),
    ),
    _CoreTerm(
        key="drc",
        canonical="DRC",
        english="Design Rule Check",
        aliases=("DRC", "\u8bbe\u8ba1\u89c4\u5219\u68c0\u67e5"),
        aggressive_dimensions=("\u89c4\u5219\u9519\u8bef\u5b9a\u4f4d", "\u9ad8\u4eae\u663e\u793a", "\u4fee\u590d\u8c03\u8bd5\u6d41\u7a0b"),
    ),
    _CoreTerm(
        key="bandwidth",
        canonical="\u5e26\u5bbd",
        english="Bandwidth",
        aliases=("\u5e26\u5bbd", "bandwidth", "bw"),
        aggressive_dimensions=("\u6d4b\u91cf\u65b9\u6cd5", "\u9891\u7387\u626b\u63cf\u8303\u56f4", "\u7ed3\u679c\u8bfb\u53d6"),
    ),
    _CoreTerm(
        key="phase_margin",
        canonical="\u76f8\u4f4d\u88d5\u5ea6",
        english="Phase Margin",
        aliases=("\u76f8\u4f4d\u88d5\u5ea6", "phase margin", "pm"),
        aggressive_dimensions=("\u6d4b\u91cf\u65b9\u6cd5", "\u7a33\u5b9a\u6027\u5224\u65ad", "\u7ed3\u679c\u5206\u6790"),
    ),
    _CoreTerm(
        key="loop_gain",
        canonical="\u73af\u8def\u589e\u76ca",
        english="Loop Gain",
        aliases=("\u73af\u8def\u589e\u76ca", "loop gain"),
        aggressive_dimensions=("\u6d4b\u91cf\u914d\u7f6e", "\u9891\u7387\u54cd\u5e94\u5206\u6790", "\u7a33\u5b9a\u6027\u5224\u65ad"),
    ),
    _CoreTerm(
        key="fft",
        canonical="\u9891\u8c31\u5206\u6790",
        english="Fast Fourier Transform",
        aliases=("\u5085\u91cc\u53f6\u53d8\u6362", "\u9891\u8c31\u5206\u6790", "fft", "fast fourier transform"),
        aggressive_dimensions=("\u91c7\u6837\u8bbe\u7f6e", "\u9891\u8c31\u83b7\u53d6", "\u6307\u6807\u5206\u6790"),
    ),
    _CoreTerm(
        key="data_sweep",
        canonical="\u6570\u636e\u626b\u63cf",
        english="Data Sweep",
        aliases=("\u6570\u636e\u626b\u63cf", "\u53c2\u6570\u626b\u63cf", "\u626b\u53c2", "data sweep", "datasweep", "sweep"),
        aggressive_dimensions=("\u626b\u63cf\u53d8\u91cf\u8bbe\u7f6e", "\u626b\u63cf\u8303\u56f4", "\u7ed3\u679c\u6570\u636e\u5206\u6790"),
    ),
    _CoreTerm(
        key="guard_ring",
        canonical="\u62a4\u73af",
        english="Guard Ring",
        aliases=("\u62a4\u73af", "guard ring", "guardring"),
        aggressive_dimensions=("\u7248\u56fe\u653e\u7f6e\u65b9\u5f0f", "\u5173\u952e\u53c2\u6570", "\u8fde\u63a5\u4e0e\u9694\u79bb\u8981\u6c42"),
    ),
    _CoreTerm(
        key="automation",
        canonical="\u81ea\u52a8\u5316",
        english="Automation",
        aliases=(
            "\u81ea\u52a8\u5316",
            "\u81ea\u52a8\u5316\u80fd\u529b",
            "\u81ea\u52a8\u5316\u5b8c\u6210",
            "automation",
            "automate",
            "workflow",
        ),
        aggressive_dimensions=(
            "\u8bbe\u8ba1\u6d41\u7a0b\u8986\u76d6\u8303\u56f4",
            "\u4eff\u771f\u6d41\u7a0b\u8986\u76d6\u8303\u56f4",
            "\u7248\u56fe\u6d41\u7a0b\u8986\u76d6\u8303\u56f4",
            "\u9a8c\u8bc1\u6d41\u7a0b\u8986\u76d6\u8303\u56f4",
            "\u80fd\u529b\u8fb9\u754c",
        ),
    ),
)


_TED_GLOSSARY: tuple[_GlossaryEntry, ...] = (
    _GlossaryEntry(
        name="measure",
        aliases=("测量", "自动测量", "measure", "measurement", "测试"),
        keywords=("Measure", "measurement", "simulation", "api"),
    ),
    _GlossaryEntry(
        name="automation",
        aliases=("\u81ea\u52a8\u5316", "\u81ea\u52a8\u5b8c\u6210", "automation", "automate", "workflow"),
        keywords=("automation", "automate", "workflow", "script", "api", "tedcore"),
    ),
    _GlossaryEntry(
        name="bandwidth",
        aliases=("带宽", "bandwidth", "bw"),
        keywords=("bandwidth", "bw", "Measure", "ac", "simulation"),
    ),
    _GlossaryEntry(
        name="phase_margin",
        aliases=("相位裕度", "phase margin", "pm"),
        keywords=("phase margin", "pm", "Measure", "ac"),
    ),
    _GlossaryEntry(
        name="loop_gain",
        aliases=("环路增益", "loop gain"),
        keywords=("loop gain", "Measure", "ac"),
    ),
    _GlossaryEntry(
        name="sfdr",
        aliases=("sfdr",),
        keywords=("sfdr", "fft", "Measure", "simulation"),
    ),
    _GlossaryEntry(
        name="fft",
        aliases=("fft", "频谱", "频域"),
        keywords=("fft", "spectrum", "Measure", "simulation"),
    ),
    _GlossaryEntry(
        name="datasweep",
        aliases=("datasweep", "data sweep", "sweep", "数据扫描", "扫描", "扫", "扫参", "参数扫描"),
        keywords=("DataSweep", "datasweep", "simulation", "sweep"),
    ),
    _GlossaryEntry(
        name="waveform",
        aliases=("波形", "waveform"),
        keywords=("waveform", "tran", "simulation"),
    ),
    _GlossaryEntry(
        name="guardring",
        aliases=("护环", "guard ring", "guardring"),
        keywords=("guard ring", "GuardRing", "layout"),
    ),
    _GlossaryEntry(
        name="lvs",
        aliases=("lvs", "\u4e00\u81f4\u6027\u68c0\u67e5"),
        keywords=("LVS", "layout", "debug", "error"),
    ),
    _GlossaryEntry(
        name="drc",
        aliases=("drc", "\u8bbe\u8ba1\u89c4\u5219\u68c0\u67e5"),
        keywords=("DRC", "layout", "debug", "error"),
    ),
    _GlossaryEntry(
        name="debug_highlight",
        aliases=(
            "\u9ad8\u4eae",
            "\u9519\u8bef\u4f4d\u7f6e",
            "\u5b9a\u4f4d",
            "\u8c03\u8bd5",
            "\u4ea4\u4e92\u5f0f\u8c03\u8bd5",
            "highlight",
            "debug",
        ),
        keywords=("highlight", "debug", "error", "layout"),
    ),
    _GlossaryEntry(
        name="mos_device",
        aliases=("\u006d\u006f\u0073", "mosfet", "nmos", "pmos", "\u004d\u004f\u0053\u7ba1", "\u5668\u4ef6"),
        keywords=("MOSFET", "MOS", "NMOS", "PMOS", "device", "ted_device"),
    ),
    _GlossaryEntry(
        name="tap",
        aliases=("tap", "衬底接触", "井接触"),
        keywords=("Tap", "tap", "layout"),
    ),
    _GlossaryEntry(
        name="layout",
        aliases=("版图", "布局", "layout"),
        keywords=("layout", "tedcore"),
    ),
    _GlossaryEntry(
        name="route",
        aliases=("布线", "route", "router", "routing"),
        keywords=("route", "RouteConfig", "layout"),
    ),
    _GlossaryEntry(
        name="smartbus",
        aliases=("smartbus", "总线"),
        keywords=("SmartBus", "bus", "route", "layout"),
    ),
    _GlossaryEntry(
        name="smt",
        aliases=("smt", "steiner", "布局器"),
        keywords=("SMT", "smt", "layout"),
    ),
    _GlossaryEntry(
        name="instlist",
        aliases=("instlist", "实例列表", "实例集"),
        keywords=("InstList", "tedcore", "layout"),
    ),
    _GlossaryEntry(
        name="simulation",
        aliases=("仿真", "simulation", "ac", "tran", "dc"),
        keywords=("simulation", "ac", "tran", "dc", "Measure"),
    ),
    _GlossaryEntry(
        name="tedcore",
        aliases=("tedcore", "pyted", "api"),
        keywords=("tedcore", "pyted", "api"),
    ),
)


class QueryRewriter:
    def __init__(self, chat_client: OpenAICompatClient, model_name: str) -> None:
        self.chat_client = chat_client
        self.model_name = model_name

    def rewrite(self, query: str, scene: str = "qa", mode: str = "conservative") -> QueryRewriteResult:
        normalized = self.normalize_query(query)
        if not normalized:
            raise ValueError("query is empty")

        scene_key = (scene or "qa").strip().lower()
        if scene_key not in _SCENES:
            raise ValueError(f"invalid scene: {scene}")
        mode_key = (mode or "conservative").strip().lower()
        if mode_key not in _REWRITE_MODES:
            raise ValueError(f"invalid rewrite mode: {mode}")

        frozen_entities = self._extract_frozen_entities(normalized)
        static_core_terms = self._match_core_terms(normalized)

        if self._should_pass_through(normalized, frozen_entities):
            return QueryRewriteResult(
                original_query=normalized,
                rewritten_query=normalized,
                changed=False,
                strategy="pass_through_precise",
                warning="",
            )

        intent = self._recognize_intent_with_llm(
            normalized,
            scene=scene_key,
            frozen_entities=frozen_entities,
            static_core_terms=static_core_terms,
        )
        core_terms = self._merge_core_terms(static_core_terms, intent.core_terms)

        if mode_key == "conservative":
            rewritten = self._rewrite_conservative(normalized, core_terms)
            if rewritten == normalized:
                if intent.source != "fallback" and intent.confidence != "low":
                    rewritten = self._rewrite_conservative_with_llm(
                        normalized,
                        scene=scene_key,
                        frozen_entities=frozen_entities,
                        core_terms=core_terms,
                        intent=intent,
                    )
                if rewritten == normalized:
                    return QueryRewriteResult(
                        original_query=normalized,
                        rewritten_query=normalized,
                        changed=False,
                        strategy="pass_through_precise",
                        warning="" if intent.source != "fallback" else intent.warning,
                    )
            return QueryRewriteResult(
                original_query=normalized,
                rewritten_query=rewritten,
                changed=True,
                strategy="llm_rewrite",
                warning="" if intent.source != "fallback" else intent.warning,
            )

        if intent.confidence == "low":
            return QueryRewriteResult(
                original_query=normalized,
                rewritten_query=normalized,
                changed=False,
                strategy="fallback_original",
                warning=intent.warning or "low confidence intent recognition",
            )

        if not core_terms and not intent.core_subjects and not intent.core_intent:
            return QueryRewriteResult(
                original_query=normalized,
                rewritten_query=normalized,
                changed=False,
                strategy="fallback_original",
                warning=intent.warning or "no reliable intent recognized",
            )

        try:
            raw = self.chat_client.chat(
                model=self.model_name,
                messages=self._build_messages(
                    normalized,
                    scene_key,
                    mode_key,
                    frozen_entities,
                    core_terms,
                    intent,
                ),
                temperature=0.0,
            )
        except Exception as exc:  # noqa: BLE001
            return QueryRewriteResult(
                original_query=normalized,
                rewritten_query=normalized,
                changed=False,
                strategy="fallback_original",
                warning=f"rewrite unavailable: {exc}",
            )

        rewritten = self._sanitize_model_output(raw)
        if mode_key == "aggressive":
            rewritten = self._enhance_aggressive_retrieval_terms(normalized, rewritten, core_terms, intent)
        rewrite_warning = ""
        if self._looks_like_keyword_soup(rewritten):
            rewritten = self._repair_natural_rewrite(
                original=normalized,
                rewritten=rewritten,
                scene=scene_key,
                mode=mode_key,
                frozen_entities=frozen_entities,
                core_terms=core_terms,
                intent=intent,
            )
            if mode_key == "aggressive":
                rewritten = self._enhance_aggressive_retrieval_terms(normalized, rewritten, core_terms, intent)
            if self._looks_like_keyword_soup(rewritten):
                rewrite_warning = "rewrite looked like keyword list"

        return QueryRewriteResult(
            original_query=normalized,
            rewritten_query=rewritten,
            changed=rewritten != normalized,
            strategy="llm_rewrite",
            warning=rewrite_warning,
        )

    @staticmethod
    def _normalize_query(query: str) -> str:
        text = unicodedata.normalize("NFKC", query or "")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = " ".join(line.strip() for line in text.split("\n") if line.strip())
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @classmethod
    def normalize_query(cls, query: str) -> str:
        return cls._normalize_query(query)

    @classmethod
    def _extract_frozen_entities(cls, query: str) -> List[str]:
        found: list[str] = []
        for pattern in (_CODE_ENTITY_RE, _PATH_RE, _FILE_RE, _CAMEL_IDENTIFIER_RE, _NUMBER_UNIT_RE):
            for match in pattern.findall(query):
                entity = match.strip()
                if pattern is _PATH_RE and _SLASH_ACRONYM_RE.fullmatch(entity):
                    continue
                if entity and entity not in found:
                    found.append(entity)
        for token in _IDENTIFIER_RE.findall(query):
            if "_" in token or any(ch.isupper() for ch in token):
                if token not in found:
                    found.append(token)
        return found

    @classmethod
    def _should_pass_through(cls, query: str, frozen_entities: Sequence[str]) -> bool:
        if _CODE_ENTITY_RE.search(query) or cls._has_precise_path_or_file(query):
            return True
        precise_entities = [item for item in frozen_entities if cls._is_precise_entity(item)]
        if precise_entities:
            return True
        return False

    @classmethod
    def _has_precise_path_or_file(cls, query: str) -> bool:
        if _FILE_RE.search(query):
            return True
        for match in _PATH_RE.finditer(query):
            value = match.group(0)
            if _SLASH_ACRONYM_RE.fullmatch(value):
                continue
            return True
        return False

    @classmethod
    def _is_precise_entity(cls, entity: str) -> bool:
        lowered = entity.lower()
        if lowered in _PASS_THROUGH_ACRONYMS:
            return False
        if _SLASH_ACRONYM_RE.fullmatch(entity):
            return False
        if _CODE_ENTITY_RE.fullmatch(entity) or _PATH_RE.fullmatch(entity) or _FILE_RE.fullmatch(entity):
            return True
        if "_" in entity or "/" in entity or "\\" in entity:
            return True
        has_lower = any(ch.islower() for ch in entity)
        has_upper = any(ch.isupper() for ch in entity)
        if has_lower and has_upper:
            return True
        if entity[:1].isupper() and has_lower and len(entity) >= 4:
            return True
        return False

    def _recognize_intent_with_llm(
        self,
        query: str,
        scene: str,
        frozen_entities: Sequence[str],
        static_core_terms: Sequence[_CoreTerm],
    ) -> _RecognizedIntent:
        try:
            raw = self.chat_client.chat(
                model=self.model_name,
                messages=self._build_intent_messages(query, scene, frozen_entities, static_core_terms),
                temperature=0.0,
            )
        except Exception as exc:  # noqa: BLE001
            return self._fallback_intent(static_core_terms, warning=f"intent recognition unavailable: {exc}")

        data = self._parse_json_object(raw)
        if not data:
            return self._fallback_intent(static_core_terms, warning="intent recognition returned invalid JSON")

        intent = self._intent_from_payload(data, static_core_terms)
        meaningful_frozen_entities = [entity for entity in frozen_entities if self._is_meaningful_entity(entity)]
        if not self._preserve_frozen_entities(
            " ".join((*intent.core_subjects, intent.core_intent)),
            meaningful_frozen_entities,
        ):
            return self._fallback_intent(static_core_terms, warning="intent recognition dropped frozen entities")
        return intent

    @staticmethod
    def _build_intent_messages(
        query: str,
        scene: str,
        frozen_entities: Sequence[str],
        static_core_terms: Sequence[_CoreTerm],
    ) -> list[dict[str, str]]:
        frozen_text = ", ".join(frozen_entities) or "(none)"
        known_text = "\n".join(
            f"- {term.canonical}: english={term.english}; aliases={', '.join(term.aliases)}"
            for term in static_core_terms
        ) or "- none"
        scene_hint = "QA retrieval" if scene == "qa" else "task retrieval"
        return [
            {
                "role": "system",
                "content": (
                    "You identify the user's technical intent for TED documentation query rewriting.\n"
                    "Return only a valid JSON object. Do not rewrite the query here.\n"
                    "The JSON schema is exactly: core_intent string, core_subjects string array, "
                    "core_terms object mapping Chinese/core term to standard English full name, "
                    "allowed_dimensions string array, forbidden_topics string array, confidence high|medium|low.\n"
                    "Rules:\n"
                    "1. Preserve the original intent and all frozen entities exactly.\n"
                    "2. Identify the narrow core technical subject, not broad background topics.\n"
                    "3. Only provide standard English names when you are confident.\n"
                    "4. allowed_dimensions should be RAG retrieval terms: canonical English aliases, acronyms, "
                    "domain keywords, closely related parameter/result words that may appear in docs.\n"
                    "5. forbidden_topics should include likely unrelated topics that must not be introduced.\n"
                    "6. Do not fill allowed_dimensions with vague words such as core steps, configuration, full process only.\n"
                    "7. If uncertain, use confidence low and keep fields minimal."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Scene: {scene_hint}\n"
                    f"Original query: {query}\n"
                    f"Frozen entities: {frozen_text}\n"
                    f"Known matched terms from deterministic glossary:\n{known_text}\n"
                    "Identify the structured intent as JSON only."
                ),
            },
        ]

    @classmethod
    def _parse_json_object(cls, raw: str) -> dict:
        text = cls._normalize_query(_CODE_FENCE_RE.sub("", raw or ""))
        if not text:
            return {}
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            return {}
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    @classmethod
    def _intent_from_payload(
        cls,
        data: dict,
        static_core_terms: Sequence[_CoreTerm],
    ) -> _RecognizedIntent:
        confidence = str(data.get("confidence") or "low").strip().lower()
        if confidence not in {"high", "medium", "low"}:
            confidence = "low"

        core_intent = cls._normalize_query(str(data.get("core_intent") or ""))
        core_subjects = tuple(cls._clean_string_list(data.get("core_subjects")))
        allowed_dimensions = tuple(cls._clean_string_list(data.get("allowed_dimensions")))
        forbidden_topics = tuple(cls._clean_string_list(data.get("forbidden_topics")))
        core_terms = cls._core_terms_from_payload(data.get("core_terms"), core_subjects, allowed_dimensions)
        merged_terms = cls._merge_core_terms(static_core_terms, core_terms)

        if not merged_terms and not core_subjects and not core_intent and confidence != "low":
            confidence = "low"

        return _RecognizedIntent(
            core_intent=core_intent,
            core_subjects=core_subjects,
            core_terms=tuple(merged_terms),
            allowed_dimensions=allowed_dimensions,
            forbidden_topics=forbidden_topics,
            confidence=confidence,
            source="llm",
            warning="",
        )

    @classmethod
    def _fallback_intent(
        cls,
        static_core_terms: Sequence[_CoreTerm],
        warning: str = "",
    ) -> _RecognizedIntent:
        dimensions: list[str] = []
        forbidden: list[str] = list(_DISALLOWED_CROSS_TOPIC_TERMS)
        for term in static_core_terms:
            dimensions.extend(term.aggressive_dimensions)
        return _RecognizedIntent(
            core_terms=tuple(static_core_terms),
            allowed_dimensions=tuple(cls._dedupe_strings(dimensions)),
            forbidden_topics=tuple(cls._dedupe_strings(forbidden)),
            confidence="medium" if static_core_terms else "low",
            source="fallback",
            warning=warning,
        )

    @classmethod
    def _core_terms_from_payload(
        cls,
        raw_terms: object,
        core_subjects: Sequence[str],
        allowed_dimensions: Sequence[str],
    ) -> List[_CoreTerm]:
        terms: list[_CoreTerm] = []
        if isinstance(raw_terms, dict):
            items = raw_terms.items()
        elif isinstance(raw_terms, list):
            items = []
            for item in raw_terms:
                if isinstance(item, dict):
                    name = item.get("term") or item.get("name") or item.get("canonical") or item.get("chinese")
                    english = item.get("english") or item.get("standard_english") or ""
                    if name:
                        items.append((name, english))
        else:
            items = []

        for name, english in items:
            canonical = cls._normalize_query(str(name or ""))
            english_text = cls._normalize_query(str(english or ""))
            if not canonical:
                continue
            matched = cls._static_term_for_name(canonical, english_text)
            if matched:
                terms.append(matched)
                continue
            aliases = tuple(cls._dedupe_strings([canonical, english_text, *core_subjects]))
            terms.append(
                _CoreTerm(
                    key=f"llm:{canonical.lower()}",
                    canonical=canonical,
                    english=english_text,
                    aliases=aliases,
                    aggressive_dimensions=tuple(allowed_dimensions),
                )
            )
        return terms

    @classmethod
    def _static_term_for_name(cls, name: str, english: str = "") -> _CoreTerm | None:
        probe = " ".join(part for part in (name, english) if part)
        for term in _CORE_TERMS:
            if cls._term_in_text(term, probe):
                return term
        return None

    @staticmethod
    def _clean_string_list(value: object) -> List[str]:
        if isinstance(value, str):
            items = [value]
        elif isinstance(value, list):
            items = value
        else:
            return []
        cleaned: list[str] = []
        for item in items:
            text = QueryRewriter._normalize_query(str(item or ""))
            if text:
                cleaned.append(text)
        return QueryRewriter._dedupe_strings(cleaned)

    @staticmethod
    def _dedupe_strings(values: Iterable[str]) -> List[str]:
        results: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = str(value or "").strip()
            key = text.lower()
            if not text or key in seen:
                continue
            seen.add(key)
            results.append(text)
        return results

    @staticmethod
    def _merge_core_terms(
        static_core_terms: Sequence[_CoreTerm],
        llm_core_terms: Sequence[_CoreTerm],
    ) -> List[_CoreTerm]:
        merged: list[_CoreTerm] = []
        seen: set[str] = set()
        for term in (*static_core_terms, *llm_core_terms):
            key = term.key.lower()
            identity = key if not key.startswith("llm:") else f"{term.canonical.lower()}|{term.english.lower()}"
            if identity in seen:
                continue
            seen.add(identity)
            merged.append(term)
        return merged

    @classmethod
    def _match_core_terms(cls, query: str) -> List[_CoreTerm]:
        matches: List[_CoreTerm] = []
        for term in _CORE_TERMS:
            if cls._term_in_text(term, query):
                matches.append(term)
        return matches

    @classmethod
    def _term_in_text(cls, term: _CoreTerm, text: str) -> bool:
        candidates = (*term.aliases, term.canonical, term.english)
        lowered = text.lower()
        compact = re.sub(r"\s+", "", lowered)
        for alias in candidates:
            if not alias:
                continue
            alias_l = alias.lower()
            alias_compact = re.sub(r"\s+", "", alias_l)
            if alias_l in lowered or (alias_compact and alias_compact in compact):
                return True
        return False

    @classmethod
    def _rewrite_conservative(cls, query: str, core_terms: Sequence[_CoreTerm]) -> str:
        rewritten = query
        for term in core_terms:
            if not term.english:
                continue
            if cls._english_present(rewritten, term):
                continue
            rewritten = cls._append_english_to_first_alias(rewritten, term)
        return rewritten

    @staticmethod
    def _english_present(text: str, term: _CoreTerm) -> bool:
        if not term.english:
            return False
        lowered = text.lower()
        english = term.english.lower()
        if english in lowered:
            return True
        return term.english.upper() in {token.upper() for token in _EN_TOKEN_RE.findall(text)}

    @staticmethod
    def _append_english_to_first_alias(text: str, term: _CoreTerm) -> str:
        if not term.english:
            return text
        aliases = [term.canonical]
        aliases.extend(
            alias
            for alias in sorted(term.aliases, key=lambda item: len(item), reverse=True)
            if alias != term.canonical
        )
        for alias in aliases:
            if not alias:
                continue
            match = re.search(re.escape(alias), text, flags=re.IGNORECASE)
            if not match:
                continue
            tail = text[match.end() :].lstrip()
            if tail.startswith(("（", "(")):
                return text
            return f"{text[: match.end()]}（{term.english}）{text[match.end() :]}"
        return text

    def _rewrite_conservative_with_llm(
        self,
        query: str,
        scene: str,
        frozen_entities: Sequence[str],
        core_terms: Sequence[_CoreTerm],
        intent: _RecognizedIntent,
    ) -> str:
        try:
            raw = self.chat_client.chat(
                model=self.model_name,
                messages=self._build_conservative_messages(query, scene, frozen_entities, core_terms, intent),
                temperature=0.0,
            )
        except Exception:  # noqa: BLE001
            return query
        rewritten = self._sanitize_model_output(raw)
        return rewritten or query

    @staticmethod
    def _build_conservative_messages(
        query: str,
        scene: str,
        frozen_entities: Sequence[str],
        core_terms: Sequence[_CoreTerm],
        intent: _RecognizedIntent,
    ) -> list[dict[str, str]]:
        core_text = "\n".join(
            f"- core={term.canonical}; standard_english={term.english or '(unknown)'}"
            for term in core_terms
        ) or "- none"
        frozen_text = ", ".join(frozen_entities) or "(none)"
        scene_hint = "QA retrieval" if scene == "qa" else "task retrieval"
        return [
            {
                "role": "system",
                "content": (
                    "You conservatively rewrite a TED documentation query.\n"
                    "Output exactly one natural Chinese query line.\n"
                    "Keep the original sentence structure and user intent.\n"
                    "Only add standard English full names in parentheses for the core technical terms when confident.\n"
                    "Do not add usage, examples, parameters, APIs, steps, or background.\n"
                    "Preserve frozen entities exactly.\n"
                    "If no safe English full name is available, return the original query unchanged."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Scene: {scene_hint}\n"
                    f"Original query: {query}\n"
                    f"Frozen entities: {frozen_text}\n"
                    f"Recognized core intent: {intent.core_intent or '(none)'}\n"
                    f"Recognized core subjects: {', '.join(intent.core_subjects) or '(none)'}\n"
                    f"Recognized core terms:\n{core_text}\n"
                    "Return the conservative rewrite only."
                ),
            },
        ]

    @staticmethod
    def _build_messages(
        query: str,
        scene: str,
        mode: str,
        frozen_entities: Sequence[str],
        core_terms: Sequence[_CoreTerm],
        intent: _RecognizedIntent,
    ) -> list[dict[str, str]]:
        core_text = "\n".join(
            (
                f"- core={term.canonical}; standard_english={term.english}; "
                f"allowed_dimensions={', '.join(term.aggressive_dimensions)}"
            )
            for term in core_terms
        ) or "- none"
        allowed_text = ", ".join(intent.allowed_dimensions) or ", ".join(
            QueryRewriter._dedupe_strings(d for term in core_terms for d in term.aggressive_dimensions)
        ) or "(none)"
        forbidden_text = ", ".join(intent.forbidden_topics) or "(none)"
        frozen_text = ", ".join(frozen_entities) or "(none)"
        scene_hint = "QA retrieval" if scene == "qa" else "task retrieval"
        if mode == "aggressive":
            mode_rule = (
                "Aggressive mode is retrieval-enhanced rewriting, not answer-style expansion. "
                "Rewrite the query so RAG can match more relevant evidence: preserve the original question intent, "
                "add standard English aliases, canonical technical terms, and 3-6 high-value retrieval terms from "
                "Allowed dimensions when they directly describe the same core subject. "
                "Do not write vague phrases such as '核心步骤', '参数配置', '完整流程' unless they are paired with "
                "specific retrieval terms. Do not add unrelated simulation types, unrelated APIs/classes/functions, "
                "new numeric constraints, or executable steps."
            )
        else:
            mode_rule = (
                "Conservative mode: only add the standard English full name of the core term in parentheses. "
                "Do not change sentence structure or add any other content."
            )
        return [
            {
                "role": "system",
                "content": (
                    "You are a technical query rewrite expert for TED documentation retrieval.\n"
                    "First identify and preserve the user's exact core intent, core technical subject, and core terminology.\n"
                    "All rewriting must serve better RAG retrieval for that core intent only.\n"
                    "Output exactly one natural Chinese query line, not an explanation or keyword list.\n"
                    "Rules:\n"
                    "1. Keep the user's intent unchanged.\n"
                    "2. Preserve all frozen entities exactly.\n"
                    "3. Prefer retrieval terms that may appear in TED docs: Chinese term, English full name, acronym, module/domain words.\n"
                    "4. Do not add unrelated simulation types, API names, operation methods, parameters, values, or steps.\n"
                    "5. If you use English terminology, embed it in parentheses or a natural phrase; do not append raw keyword soup.\n"
                    "6. Never output keyword soup like 'TED AC 仿真 simulation ac'.\n"
                    "7. Output one single line only, with no explanation or bullets.\n"
                    f"{mode_rule}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Scene: {scene_hint}\n"
                    f"Rewrite mode: {mode}\n"
                    f"Original query: {query}\n"
                    f"Frozen entities: {frozen_text}\n"
                    f"Recognized core intent: {intent.core_intent or '(none)'}\n"
                    f"Recognized core subjects: {', '.join(intent.core_subjects) or '(none)'}\n"
                    f"Identified core subjects:\n{core_text}\n"
                    f"Allowed dimensions: {allowed_text}\n"
                    f"Forbidden topics: {forbidden_text}\n"
                    "Return a retrieval-enhanced natural query. It should be more searchable than the original, "
                    "not a generic question about steps."
                ),
            },
        ]

    @classmethod
    def _sanitize_model_output(cls, raw: str) -> str:
        text = cls._normalize_query(_CODE_FENCE_RE.sub("", raw or ""))
        if not text:
            return ""
        line = text.splitlines()[0].strip()
        if len(line) >= 2 and line[0] == line[-1] and line[0] in {'"', "'"}:
            line = line[1:-1].strip()
        return cls._normalize_query(line)

    @classmethod
    def _looks_like_keyword_soup(cls, text: str) -> bool:
        normalized = cls._normalize_query(text)
        if not normalized:
            return True

        english_tokens = _EN_TOKEN_RE.findall(normalized)
        has_zh = bool(_ZH_CHAR_RE.search(normalized))
        has_parenthetical_alias = "（" in normalized or "(" in normalized

        if not has_zh:
            return len(english_tokens) >= 3

        if not has_parenthetical_alias:
            if _TRAILING_EN_KEYWORDS_RE.search(normalized):
                return True
            if _UNPARENTHESIZED_ALIAS_RE.search(normalized):
                return True

        has_natural_marker = any(marker in normalized for marker in _NATURAL_QUERY_MARKERS)
        if not has_natural_marker and len(english_tokens) >= 3 and len(normalized.split()) >= 3:
            return True
        return False

    def _repair_natural_rewrite(
        self,
        original: str,
        rewritten: str,
        scene: str,
        mode: str,
        frozen_entities: Sequence[str],
        core_terms: Sequence[_CoreTerm],
        intent: _RecognizedIntent,
    ) -> str:
        core_text = "\n".join(
            (
                f"- core={term.canonical}; standard_english={term.english}; "
                f"allowed_dimensions={', '.join(term.aggressive_dimensions)}"
            )
            for term in core_terms
        ) or "- none"
        frozen_text = ", ".join(frozen_entities) or "(none)"
        forbidden_text = ", ".join(intent.forbidden_topics) or "(none)"
        mode_hint = (
            "保守型：只给核心术语补标准英文全称，不改变句式，不额外扩写。"
            if mode == "conservative"
            else "激进型：只围绕已识别核心主体补充允许的专业维度，禁止引入无关主题。"
        )
        try:
            raw = self.chat_client.chat(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You repair a TED query rewrite so it reads like a natural Chinese user query.\n"
                            "Only fix wording. Do not change the original intent.\n"
                            "Preserve all frozen entities exactly.\n"
                            "Preserve the identified core technical subject.\n"
                            "Use English terminology only in parentheses or natural short phrases.\n"
                            "Do not add unrelated simulation types, APIs, parameters, values, or steps.\n"
                            "Never output a raw keyword list.\n"
                            "Output one line only, no explanation."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Scene: {scene}\n"
                            f"Rewrite mode: {mode}\n"
                            f"Mode requirement: {mode_hint}\n"
                            f"Original query: {original}\n"
                            f"Current rewrite: {rewritten}\n"
                            f"Frozen entities: {frozen_text}\n"
                            f"Recognized core intent: {intent.core_intent or '(none)'}\n"
                            f"Forbidden topics: {forbidden_text}\n"
                            f"Identified core subjects:\n{core_text}\n"
                            "Convert the current rewrite into a natural Chinese query."
                        ),
                    },
                ],
                temperature=0.0,
            )
        except Exception:  # noqa: BLE001
            return rewritten
        repaired = self._sanitize_model_output(raw)
        return repaired or rewritten

    @classmethod
    def _enhance_aggressive_retrieval_terms(
        cls,
        original: str,
        rewritten: str,
        core_terms: Sequence[_CoreTerm],
        intent: _RecognizedIntent,
    ) -> str:
        if not rewritten:
            return rewritten
        original_tokens = {token.lower() for token in _EN_TOKEN_RE.findall(original)}
        rewritten_tokens = {token.lower() for token in _EN_TOKEN_RE.findall(rewritten)}
        added_tokens = rewritten_tokens - original_tokens
        if added_tokens:
            return rewritten

        retrieval_terms = cls._retrieval_terms_for_aggressive(core_terms, intent)
        if not retrieval_terms:
            return rewritten
        suffix = "\u68c0\u7d22\u76f8\u5173\u672f\u8bed\uff1a" + "\u3001".join(retrieval_terms)
        end = ""
        body = rewritten
        if body.endswith(("?", "\uff1f")):
            end = body[-1]
            body = body[:-1].rstrip()
        return f"{body}\uff0c{suffix}{end}"

    @classmethod
    def _retrieval_terms_for_aggressive(
        cls,
        core_terms: Sequence[_CoreTerm],
        intent: _RecognizedIntent,
    ) -> List[str]:
        candidates: list[str] = []
        for term in core_terms:
            if term.english:
                candidates.append(term.english)
            for alias in term.aliases:
                if alias and _EN_TOKEN_RE.search(alias):
                    candidates.append(alias)
            candidates.extend(item for item in term.aggressive_dimensions if _EN_TOKEN_RE.search(item))
        candidates.extend(item for item in intent.allowed_dimensions if _EN_TOKEN_RE.search(item))

        results: list[str] = []
        seen: set[str] = set()
        for item in candidates:
            text = cls._normalize_query(item)
            key = text.lower()
            if not text or key in seen:
                continue
            seen.add(key)
            results.append(text)
            if len(results) >= 5:
                break
        return results

    @classmethod
    def _validate_rewrite(
        cls,
        original: str,
        rewritten: str,
        frozen_entities: Sequence[str],
        core_terms: Sequence[_CoreTerm],
        mode: str,
        intent: _RecognizedIntent | None = None,
        forbidden_topics: Sequence[str] = (),
    ) -> tuple[bool, str]:
        if not rewritten:
            return False, "rewrite returned empty content"

        if mode == "aggressive":
            max_len = min(240, max(len(original) + 160, len(original) * 8))
        else:
            max_len = min(160, max(len(original) + 80, len(original) * 5))
        if len(rewritten) > max_len:
            return False, "rewrite exceeded safe length limit"

        if not cls._preserve_frozen_entities(rewritten, frozen_entities):
            return False, "rewrite dropped frozen entities"

        allowed_identifiers = cls._allowed_identifier_terms(core_terms)
        if cls._introduces_disallowed_identifier(original, rewritten, allowed_identifiers, frozen_entities):
            return False, "rewrite introduced unsupported specific identifiers"

        if core_terms and not cls._preserves_core_terms(rewritten, core_terms):
            return False, "rewrite dropped core technical subject"

        if cls._introduces_unmatched_core_term(original, rewritten, core_terms):
            return False, "rewrite introduced unrelated core subject"

        if cls._introduces_cross_topic_terms(original, rewritten, core_terms):
            return False, "rewrite introduced unrelated technical terms"

        if cls._introduces_forbidden_topic(original, rewritten, forbidden_topics):
            return False, "rewrite introduced forbidden topics"

        if not cls._preserves_salient_intent(original, rewritten, core_terms, frozen_entities, intent):
            return False, "rewrite drifted from original intent"

        return True, ""

    @staticmethod
    def _preserve_frozen_entities(rewritten: str, frozen_entities: Sequence[str]) -> bool:
        lowered = rewritten.lower()
        return all(entity.lower() in lowered for entity in frozen_entities)

    @classmethod
    def _introduces_disallowed_identifier(
        cls,
        original: str,
        rewritten: str,
        allowed_terms: Sequence[str],
        frozen_entities: Sequence[str],
    ) -> bool:
        original_allowed = {token.lower() for token in _CODE_ENTITY_RE.findall(original)}
        original_allowed.update(token.lower() for token in _IDENTIFIER_RE.findall(original))
        original_allowed.update(entity.lower() for entity in frozen_entities)
        original_allowed.update(term.lower() for term in allowed_terms if _IDENTIFIER_RE.fullmatch(term))

        introduced = set(_CODE_ENTITY_RE.findall(rewritten))
        introduced.update(token for token in _CAMEL_IDENTIFIER_RE.findall(rewritten))
        for item in introduced:
            lowered = item.lower()
            if lowered not in original_allowed:
                return True
        return False

    @classmethod
    def _allowed_identifier_terms(cls, core_terms: Sequence[_CoreTerm]) -> List[str]:
        terms: list[str] = ["TED"]
        for term in core_terms:
            terms.extend(term.aliases)
            terms.append(term.canonical)
            terms.append(term.english)
            terms.extend(_EN_TOKEN_RE.findall(term.english))
            for dimension in term.aggressive_dimensions:
                terms.extend(_EN_TOKEN_RE.findall(dimension))

        deduped: list[str] = []
        seen: set[str] = set()
        for term in terms:
            key = term.lower()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(term)
        return deduped

    @classmethod
    def _preserves_core_terms(cls, rewritten: str, core_terms: Sequence[_CoreTerm]) -> bool:
        return all(cls._term_in_text(term, rewritten) for term in core_terms)

    @classmethod
    def _introduces_unmatched_core_term(
        cls,
        original: str,
        rewritten: str,
        core_terms: Sequence[_CoreTerm],
    ) -> bool:
        matched_keys = {term.key for term in core_terms}
        for term in _CORE_TERMS:
            if term.key in matched_keys:
                continue
            if cls._term_in_text(term, rewritten) and not cls._term_in_text(term, original):
                return True
        return False

    @classmethod
    def _introduces_cross_topic_terms(
        cls,
        original: str,
        rewritten: str,
        core_terms: Sequence[_CoreTerm],
    ) -> bool:
        for term in _DISALLOWED_CROSS_TOPIC_TERMS:
            if not cls._token_or_phrase_in_text(term, rewritten):
                continue
            if cls._token_or_phrase_in_text(term, original):
                continue
            if cls._term_allowed_by_core(term, core_terms):
                continue
            return True
        return False

    @classmethod
    def _introduces_forbidden_topic(
        cls,
        original: str,
        rewritten: str,
        forbidden_topics: Sequence[str],
    ) -> bool:
        for topic in forbidden_topics:
            if not topic:
                continue
            if cls._token_or_phrase_in_text(topic, original):
                continue
            if cls._token_or_phrase_in_text(topic, rewritten):
                return True
        return False

    @classmethod
    def _term_allowed_by_core(cls, value: str, core_terms: Sequence[_CoreTerm]) -> bool:
        for term in core_terms:
            allowed_text = " ".join(
                (*term.aliases, term.canonical, term.english, *term.aggressive_dimensions)
            )
            if cls._token_or_phrase_in_text(value, allowed_text):
                return True
        return False

    @staticmethod
    def _token_or_phrase_in_text(value: str, text: str) -> bool:
        if _ALNUM_LOWER_RE.search(value):
            pattern = rf"(?<![A-Za-z0-9_]){re.escape(value)}(?![A-Za-z0-9_])"
            return bool(re.search(pattern, text, flags=re.IGNORECASE))
        return value in text

    @classmethod
    def _preserves_salient_intent(
        cls,
        original: str,
        rewritten: str,
        core_terms: Sequence[_CoreTerm],
        frozen_entities: Sequence[str],
        intent: _RecognizedIntent | None = None,
    ) -> bool:
        rewritten_l = rewritten.lower()

        if intent is not None:
            recognized_parts = [
                intent.core_intent,
                *intent.core_subjects,
                *(term.canonical for term in core_terms),
                *(term.english for term in core_terms),
            ]
            recognized_parts = [part for part in recognized_parts if part]
            if recognized_parts and any(cls._fragment_matches(part, rewritten) for part in recognized_parts):
                return True

        meaningful_entities = [entity for entity in frozen_entities if cls._is_meaningful_entity(entity)]
        if meaningful_entities and any(entity.lower() in rewritten_l for entity in meaningful_entities):
            return True

        salient = cls._salient_fragments(original)
        if not salient:
            return True

        if core_terms:
            core_aliases = {
                alias.lower()
                for term in core_terms
                for alias in (*term.aliases, term.canonical, term.english)
            }
            salient = [
                fragment
                for fragment in salient
                if not any(fragment.lower() in alias or alias in fragment.lower() for alias in core_aliases)
            ]
            if not salient:
                return True

        matched_count = sum(1 for fragment in salient if cls._fragment_matches(fragment, rewritten))
        required = 1 if len(salient) <= 3 else 2
        return matched_count >= required

    @classmethod
    def _is_meaningful_entity(cls, entity: str) -> bool:
        lowered = entity.lower()
        if lowered in _GENERIC_ENTITY_STOPWORDS:
            return False
        if _NUMBER_UNIT_RE.fullmatch(entity):
            return True
        if len(entity) <= 2:
            return False
        return True

    @classmethod
    def _fragment_matches(cls, fragment: str, rewritten: str) -> bool:
        rewritten_l = rewritten.lower()
        if _ALNUM_LOWER_RE.search(fragment):
            return fragment.lower() in rewritten_l
        if fragment in rewritten:
            return True
        return any(bigram in rewritten for bigram in cls._zh_bigrams(fragment))

    @classmethod
    def _salient_fragments(cls, text: str) -> List[str]:
        normalized = cls._normalize_query(text)
        for filler in _QUESTION_FILLERS:
            normalized = normalized.replace(filler, "")
        for suffix in _QUESTION_SUFFIXES:
            if normalized.endswith(suffix):
                normalized = normalized[: -len(suffix)]
        normalized = normalized.strip(" ，。,.?？!！:：")

        fragments: list[str] = []
        for token in re.split(r"[\s,，。.:：/()（）\-]+", normalized):
            clean = token.strip()
            if not clean:
                continue
            clean_l = clean.lower()
            if clean in _CORE_INTENT_STOPWORDS or clean_l in _CORE_INTENT_STOPWORDS:
                continue
            if len(clean) >= 2:
                fragments.append(clean)
        return fragments

    @staticmethod
    def _zh_bigrams(text: str) -> List[str]:
        phrases = _ZH_PHRASE_RE.findall(text)
        results: list[str] = []
        for phrase in phrases:
            if len(phrase) <= 2:
                results.append(phrase)
                continue
            results.extend(phrase[idx: idx + 2] for idx in range(len(phrase) - 1))
        return results

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, List

from backend.config import settings
from backend.llm.client import OpenAICompatClient
from backend.rag.retriever import ScoredChunk


@dataclass
class GeneratedScript:
    code: str
    test_type: str
    plan_text: str


@dataclass
class BandwidthTemplateParams:
    input_node: str = "vin"
    output_node: str = "vout"
    start_hz: float = 1.0
    stop_hz: float = 1e9
    points: int = 200
    target_gain_db: float = -3.0


class CodegenError(RuntimeError):
    pass


class ScriptGenerator:
    def __init__(self, llm_client: OpenAICompatClient | None = None) -> None:
        self.llm_client = llm_client

    def generate(
        self,
        query: str,
        circuit_description: str | None,
        evidence: List[ScoredChunk],
        execute: bool = False,
    ) -> GeneratedScript:
        if not evidence:
            raise CodegenError("code generation failed: no retrieval evidence available")
        if not self.llm_client or not settings.openai_api_key:
            raise CodegenError("code generation failed: LLM client is not configured")

        test_type = self._infer_test_type(query)
        plan_text = self._make_plan(test_type)
        circuit = circuit_description or "用户未提供电路细节"

        if execute:
            if test_type != "opamp_bandwidth":
                raise CodegenError(
                    f"code generation failed: real template not integrated for {test_type}; "
                    "only opamp_bandwidth is supported in execute=true"
                )
            params = self._extract_bandwidth_params(query, circuit, evidence)
            code = self._render_bandwidth_template(params, evidence)
            return GeneratedScript(code=code, test_type=test_type, plan_text=plan_text)

        return self._generate_preview_script(query, circuit, evidence, test_type, plan_text)

    def _generate_preview_script(
        self,
        query: str,
        circuit: str,
        evidence: List[ScoredChunk],
        test_type: str,
        plan_text: str,
    ) -> GeneratedScript:
        metric_key = {
            "opamp_bandwidth": "bandwidth_hz",
            "dac_sfdr": "sfdr_db",
            "amp_loop_gain": "loop_gain_db",
        }[test_type]
        method_note = {
            "opamp_bandwidth": "通过AC扫频后寻找增益下降3dB频点",
            "dac_sfdr": "通过FFT频谱识别主信号与最大杂散分量",
            "amp_loop_gain": "通过环路断开或stb方式计算环路增益",
        }[test_type]
        evidence_text = "\n".join([f"- source: {e.chunk.source}\n  snippet: {e.chunk.text[:400]}" for e in evidence[:4]])

        try:
            code = self.llm_client.chat(
                model=settings.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是EDA测试脚本生成器。"
                            "只返回单个Python脚本源码，不要markdown，不要解释。"
                            "脚本必须是demo骨架，不要求真实pyted可执行，但要明显体现检索证据。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"用户需求: {query}\n"
                            f"电路描述: {circuit}\n"
                            f"测试类型: {test_type}\n"
                            f"目标指标字段: {metric_key}\n"
                            f"方法提示: {method_note}\n"
                            f"规划摘要: {plan_text}\n"
                            "检索证据:\n"
                            f"{evidence_text}\n\n"
                            "请生成一个Python脚本，并满足以下要求:\n"
                            "1. 只使用Python标准库。\n"
                            "2. 脚本包含 run_test() -> dict。\n"
                            "3. 在注释和变量命名中体现检索证据中的术语、步骤或API线索。\n"
                            "4. 明确标出 TODO: replace with real pyted API calls。\n"
                            "5. 返回结构必须包含 status、metrics、notes，metrics 中必须包含目标指标字段。\n"
                            "6. 在 __main__ 中把结果写入 result.json 并打印JSON。\n"
                            "7. 不要输出任何解释文字或代码块围栏。"
                        ),
                    },
                ],
                temperature=0.1,
            )
        except Exception as exc:  # noqa: BLE001
            raise CodegenError(f"code generation failed: {exc}") from exc

        code = self._strip_code_fence(code).strip()
        if not code:
            raise CodegenError("code generation failed: model returned empty content")
        return GeneratedScript(code=code, test_type=test_type, plan_text=plan_text)

    def _extract_bandwidth_params(
        self,
        query: str,
        circuit: str,
        evidence: List[ScoredChunk],
    ) -> BandwidthTemplateParams:
        defaults = BandwidthTemplateParams()
        evidence_text = "\n".join([f"- {e.chunk.source}: {e.chunk.text[:180]}" for e in evidence[:4]])
        try:
            raw = self.llm_client.chat(
                model=settings.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是参数提取器。仅返回一个JSON对象，不要额外解释。"
                            "字段必须是: input_node, output_node, start_hz, stop_hz, points, target_gain_db。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"需求: {query}\n"
                            f"电路描述: {circuit}\n"
                            f"证据: {evidence_text}\n"
                            "请提取带宽AC仿真参数。若无法确定，使用保守默认值。"
                        ),
                    },
                ],
                temperature=0.0,
            )
            payload = self._parse_json_object(raw)
            if not payload:
                return defaults
            params = BandwidthTemplateParams(
                input_node=self._node_or_default(payload.get("input_node"), defaults.input_node),
                output_node=self._node_or_default(payload.get("output_node"), defaults.output_node),
                start_hz=self._float_or_default(payload.get("start_hz"), defaults.start_hz),
                stop_hz=self._float_or_default(payload.get("stop_hz"), defaults.stop_hz),
                points=self._int_or_default(payload.get("points"), defaults.points),
                target_gain_db=self._float_or_default(payload.get("target_gain_db"), defaults.target_gain_db),
            )
            if params.start_hz <= 0:
                params.start_hz = defaults.start_hz
            if params.stop_hz <= params.start_hz:
                params.stop_hz = max(params.start_hz * 10.0, defaults.stop_hz)
            params.points = min(max(params.points, 20), 20000)
            return params
        except Exception:
            return defaults

    @staticmethod
    def _render_bandwidth_template(params: BandwidthTemplateParams, evidence: List[ScoredChunk]) -> str:
        evidence_sources = [e.chunk.source for e in evidence[:4]]
        return f"""import json
import os
import shlex
import subprocess

DEFAULT_INPUT_NODE = {params.input_node!r}
DEFAULT_OUTPUT_NODE = {params.output_node!r}
DEFAULT_START_HZ = {params.start_hz}
DEFAULT_STOP_HZ = {params.stop_hz}
DEFAULT_POINTS = {params.points}
DEFAULT_TARGET_GAIN_DB = {params.target_gain_db}
EVIDENCE_SOURCES = {json.dumps(evidence_sources, ensure_ascii=False)}


def run_test():
    result = {{
        "status": "FAIL",
        "metrics": {{"bandwidth_hz": 0.0}},
        "notes": [],
    }}
    try:
        cmd_text = os.getenv("TED_BANDWIDTH_CMD", "").strip()
        if not cmd_text:
            raise RuntimeError("TED_BANDWIDTH_CMD is not configured")

        args = shlex.split(cmd_text) + [
            "--input-node",
            DEFAULT_INPUT_NODE,
            "--output-node",
            DEFAULT_OUTPUT_NODE,
            "--start-hz",
            str(DEFAULT_START_HZ),
            "--stop-hz",
            str(DEFAULT_STOP_HZ),
            "--points",
            str(DEFAULT_POINTS),
            "--target-gain-db",
            str(DEFAULT_TARGET_GAIN_DB),
        ]
        proc = subprocess.run(args, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"TED bandwidth command failed: {{proc.stderr.strip() or proc.stdout.strip()}}")

        payload = json.loads(proc.stdout.strip() or "{{}}")
        bandwidth_hz = float(payload.get("bandwidth_hz"))
        if bandwidth_hz <= 0:
            raise RuntimeError(f"invalid bandwidth_hz from TED command: {{bandwidth_hz}}")

        result["status"] = "PASS"
        result["metrics"]["bandwidth_hz"] = bandwidth_hz
        result["notes"] = [
            "real bandwidth flow executed",
            f"input_node={{DEFAULT_INPUT_NODE}}",
            f"output_node={{DEFAULT_OUTPUT_NODE}}",
            f"freq_range={{DEFAULT_START_HZ}}..{{DEFAULT_STOP_HZ}}",
            f"target_gain_db={{DEFAULT_TARGET_GAIN_DB}}",
            f"evidence={{EVIDENCE_SOURCES}}",
        ]
    except Exception as exc:  # noqa: BLE001
        result["notes"].append(str(exc))
    return result


if __name__ == "__main__":
    out = run_test()
    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False))
"""

    @staticmethod
    def _parse_json_object(content: str) -> dict[str, Any]:
        text = content.strip()
        if not text:
            return {}
        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else {}
        except Exception:
            pass
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return {}
        try:
            data = json.loads(m.group(0))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _node_or_default(value: Any, default: str) -> str:
        if isinstance(value, str):
            node = value.strip()
            if re.match(r"^[A-Za-z_][A-Za-z0-9_.$:-]*$", node):
                return node
        return default

    @staticmethod
    def _float_or_default(value: Any, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return default

    @staticmethod
    def _int_or_default(value: Any, default: int) -> int:
        try:
            return int(value)
        except Exception:
            return default

    @staticmethod
    def _make_plan(test_type: str) -> str:
        return {
            "opamp_bandwidth": "1) 配置AC扫频 2) 提取增益曲线 3) 计算-3dB带宽",
            "dac_sfdr": "1) 配置瞬态与采样 2) 执行FFT 3) 计算主瓣与最大杂散差值",
            "amp_loop_gain": "1) 构建环路增益测试 2) 扫频或stb分析 3) 提取低频环路增益",
        }[test_type]

    @staticmethod
    def _infer_test_type(query: str) -> str:
        q = query.lower()
        if "sfdr" in q:
            return "dac_sfdr"
        if "loop" in q or "环路" in q:
            return "amp_loop_gain"
        return "opamp_bandwidth"

    @staticmethod
    def _strip_code_fence(content: str) -> str:
        fenced = re.match(r"^\s*```(?:python)?\s*(.*?)\s*```\s*$", content, flags=re.DOTALL)
        return fenced.group(1) if fenced else content

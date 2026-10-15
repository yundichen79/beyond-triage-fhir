"""Native Qwen tool calling with bounded episodes and an identical public contract."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import time

from dataset.common import canonical
from pilot.tools import PatientTools, TOOL_DEFINITIONS


MODEL_ID = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
MODEL_CACHE = Path(__file__).resolve().parents[1] / "vendor/models"

# This complete, value-free contract is the same for every question. It is not
# derived from a question's family, oracle, answer, or supporting resource IDs.
SYSTEM_PROMPT = """Answer one patient's clinical chart-review question using the FHIR tools.
Use only records obtained through these tools. Patient scope is automatic. Search
with useful dates, codes, encounter references, statuses and sorting. Follow a
search's next_page cursor when the remaining matches are needed. An incomplete
page or tool error cannot establish that a record does not exist. You may refine
failed searches. Resource references are authoritative for report results and
documented indications. Orders do not establish medication administration or use.
Return UTC timestamps with timezone, coded concepts, stated units, and requested
calculations. Do not invent values, indications, discontinuation, or causal effects.
Finish with the finish tool, alone in its turn, after reading its evidence. A final
JSON object with keys answer and evidence is also accepted. Cite ResourceType/id
references actually returned by tools. Use the following public output contract;
these are schemas, not worked examples or patient-specific hints.

Shared structures:
Concept = {system: string, code: string, name: string}.
ConditionEntry = {condition: Concept, onset: timestamp or null}.
Order = {medication: Concept, ordered_at: timestamp, recorded_status: string}.
Measurement when found = {status: "found", time: timestamp, value: number, unit: string}.
When a matching measurement has no numeric value, use the same fields with status
"missing_value" and value null. When no matching record exists, use only
{status: "no_matching_record"}. Numeric strings are accepted. Use mm[Hg] for blood
pressure, g/dL for hemoglobin, 10*3/uL for platelets, mg/dL for creatinine and % for HbA1c.
CBC = {report_time: timestamp, hemoglobin: Measurement, platelets: Measurement}.
Products = {both: Concept[], only_first: Concept[], only_second: Concept[]}.
For two stays, first means earlier and second later; for discharge follow-up,
first means inpatient and second ambulatory. Product identity is system and code.
Arrays are unordered, but retain separate orders when the question requests orders.
Missing calculated changes are null. Round only calculations that the question
requests to one decimal; retain measurement precision.

Answer objects by requested chart-review task:
- Stay conditions: admitted_at, discharged_at, conditions: ConditionEntry[].
- Stay orders: admitted_at, discharged_at, orders: Order[].
- Preadmission prescribing history: admitted_at, discharged_at,
  conditions_with_prior_orders: [{condition: Concept, onset: timestamp,
  abatement: timestamp or null, orders: [{medication: Concept, ordered_at: timestamp}]}].
  An empty documented history is an empty array.
- Latest CBC: use the CBC structure directly.
- CBC comparison: previous: CBC, latest: CBC, hemoglobin_change_g_dL: number or null,
  platelet_change_10_3_uL: number or null. Changes are latest minus previous.
- Ambulatory blood pressure: visit_started_at: timestamp, systolic: Measurement,
  diastolic: Measurement.
- Discharge follow-up: admitted_at, discharged_at, followup: null or
  {started_at: timestamp, days_after_discharge: number, conditions: ConditionEntry[]}.
- Products across two stays: admitted_at and discharged_at of the later stay,
  previous_admitted_at of the earlier stay, products: Products.
- Products at discharge follow-up: admitted_at, discharged_at, followup: null or
  {started_at: timestamp, days_after_discharge: number}; include products: Products
  only when a follow-up exists.
- Procedure indications: encounter_started_at: timestamp,
  procedures: [{procedure: Concept, performed_at: timestamp, indications: Concept[]}].
- Prescribing indications: encounter_started_at: timestamp,
  orders: [{medication: Concept, indications: Concept[]}].
- Glycemic follow-up: condition: Concept, onset: timestamp, followup: Measurement.
  Only when a follow-up record exists, also include baseline: Measurement and
  change_percentage_points: number or null (follow-up minus baseline).
- Renal laboratory follow-up: baseline: Measurement, followup: Measurement,
  change_mg_dL: number or null (follow-up minus baseline).
- Subsequent admission: admitted_at and discharged_at of the initial stay,
  subsequent_admission: null or {admitted_at: timestamp, days_after_discharge: number,
  conditions: ConditionEntry[], shared_conditions: Concept[]}.
Include exactly the requested answer structure. Do not include a task label.
"""


@dataclass(frozen=True)
class Settings:
    max_steps: int = 24
    max_tool_calls: int = 48
    max_context_tokens: int = 16384
    max_new_tokens: int = 2048
    max_output_tokens: int = 16384
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    seed: int = 4880

    def __post_init__(self):
        for name in ("max_steps", "max_tool_calls", "max_context_tokens", "max_new_tokens", "max_output_tokens", "top_k"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.temperature) or not 0 < self.temperature or not 0 < self.top_p <= 1:
            raise ValueError("Sampling requires a positive temperature and 0 < top_p <= 1")
        if self.max_new_tokens >= self.max_context_tokens:
            raise ValueError("Generation budget must be smaller than the context limit")


class ContextLimit(ValueError):
    pass


@dataclass
class Generation:
    text: str
    input_tokens: int
    output_tokens: int
    truncated: bool = False


class TransformersModel:
    """Load only the prepared checkpoint; no model download during a GPU run."""
    def __init__(self, settings, model_path=None):
        try:
            import torch
            import transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ValueError("Pilot dependencies are missing; run bash scripts/prepare_pilot.sh") from exc
        if not torch.cuda.is_available():
            raise ValueError("The pilot requires a GPU allocation; use scripts/run_pilot.sbatch")
        self.settings, self.torch, self.transformers = settings, torch, transformers
        source = str(model_path) if model_path else MODEL_ID
        kwargs = {"local_files_only": True, "trust_remote_code": False}
        if not model_path:
            kwargs.update(revision=MODEL_REVISION, cache_dir=str(MODEL_CACHE))
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
            self.model = AutoModelForCausalLM.from_pretrained(
                source, torch_dtype=torch.bfloat16, device_map={"": "cuda:0"}, **kwargs).eval()
        except OSError as exc:
            raise ValueError("Prepared model checkpoint is unavailable; run bash scripts/prepare_pilot.sh") from exc
        self.metadata = {"model": source, "revision": getattr(self.model.config, "_commit_hash", None)
                         or (MODEL_REVISION if not model_path else None), "thinking": False,
                         "torch": torch.__version__, "transformers": transformers.__version__,
                         "gpu": torch.cuda.get_device_name(0), "dtype": "bfloat16"}

    def seed(self, value):
        self.transformers.set_seed(value)

    def generate(self, messages, tools, *, max_new_tokens=None):
        budget = min(self.settings.max_new_tokens, max_new_tokens or self.settings.max_new_tokens)
        text = self.tokenizer.apply_chat_template(messages, tools=tools, tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=False)
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        count = inputs["input_ids"].shape[-1]
        limit = min(self.settings.max_context_tokens, int(self.model.config.max_position_embeddings))
        if count + budget > limit:
            raise ContextLimit(f"Context would exceed {limit} tokens; no history was silently removed")
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=budget,
                do_sample=True, temperature=self.settings.temperature, top_p=self.settings.top_p,
                top_k=self.settings.top_k, pad_token_id=self.tokenizer.eos_token_id)
        generated = output[0, count:]
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        truncated = len(generated) >= budget and int(generated[-1]) not in eos
        text = self.tokenizer.decode(generated, skip_special_tokens=False).rstrip()
        for token in (self.tokenizer.eos_token, self.tokenizer.pad_token):
            if token:
                text = text.removesuffix(token).rstrip()
        return Generation(text, count, len(generated), truncated)


def json_object(text):
    def invalid_constant(value):
        raise ValueError(f"Invalid JSON numeric constant: {value}")
    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("JSON number exceeds finite floating-point range")
        return number
    value = json.loads(text, parse_constant=invalid_constant, parse_float=finite_float)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def parse_calls(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
    if blocks:
        calls = [json_object(block) for block in blocks]
    else:
        if text.startswith("```json") and text.endswith("```"):
            text = text[7:-3].strip()
        value = json_object(text)
        if set(value) != {"answer", "evidence"}:
            raise ValueError("Use tool_call blocks or a final JSON object containing answer and evidence")
        calls = [{"name": "finish", "arguments": value}]
    for call in calls:
        if set(call) != {"name", "arguments"} or not isinstance(call["name"], str) or not isinstance(call["arguments"], dict):
            raise ValueError("Each call needs a name and an arguments object")
    if len(calls) > 1 and any(c["name"] == "finish" for c in calls):
        raise ValueError("Submit finish alone after observing tool results")
    return calls


def run_agent(question, patient, model, client, settings, seed):
    """Only question and patient context enter the model; all scoring is external."""
    tools = PatientTools(client, patient)
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Patient reference: {patient}\n\n{question}"}]
    started = time.monotonic()
    totals = {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "protocol_errors": 0, "truncated_turns": 0}
    status, error, final = "step_limit", None, None
    model.seed(seed)
    for _ in range(settings.max_steps):
        remaining = settings.max_output_tokens - totals["output_tokens"]
        if remaining <= 0:
            status = "output_token_limit"
            break
        totals["llm_calls"] += 1
        try:
            generation = model.generate(messages, TOOL_DEFINITIONS,
                                        max_new_tokens=min(settings.max_new_tokens, remaining))
        except ContextLimit as exc:
            status, error = "context_limit", str(exc)
            break
        except InterruptedError:
            raise
        except (OSError, ValueError, RuntimeError) as exc:
            status, error = "model_error", str(exc)
            break
        totals["input_tokens"] += generation.input_tokens
        totals["output_tokens"] += generation.output_tokens
        totals["truncated_turns"] += int(generation.truncated)
        try:
            calls = parse_calls(generation.text)
        except (ValueError, TypeError) as exc:
            totals["protocol_errors"] += 1
            messages.extend([{"role": "assistant", "content": generation.text},
                {"role": "user", "content": "Output format error: " + str(exc)
                 + ". Use the tool signatures and public answer contract; do not guess missing evidence."}])
            if totals["output_tokens"] >= settings.max_output_tokens:
                status = "output_token_limit"
                break
            continue
        messages.append({"role": "assistant", "content": generation.text.split("<tool_call>", 1)[0].strip() if "<tool_call>" in generation.text else "",
                         "tool_calls": [{"type": "function", "function": call} for call in calls]})
        for call in calls:
            name, arguments = call["name"], call["arguments"]
            if name == "finish":
                evidence = arguments.get("evidence")
                if (set(arguments) != {"answer", "evidence"} or not isinstance(arguments["answer"], dict)
                        or not isinstance(evidence, list) or any(not isinstance(ref, str) for ref in evidence)):
                    result = {"error": "finish requires an answer object and an evidence array of references"}
                elif set(evidence) - tools.seen:
                    result = {"error": "Citations must refer to resources actually returned by tools"}
                else:
                    final = {"answer": arguments["answer"], "evidence": sorted(set(evidence))}
                    status = "completed"
                    break
                totals["protocol_errors"] += 1
            elif tools.calls >= settings.max_tool_calls:
                status, error = "tool_call_limit", "The episode exhausted its FHIR tool-call budget"
                break
            else:
                result = tools.execute(name, arguments)
            messages.append({"role": "tool", "name": name, "content": canonical(result)})
        if final is not None or status == "tool_call_limit":
            break
        if totals["output_tokens"] >= settings.max_output_tokens:
            status = "output_token_limit"
            break
    totals.update(tool_calls=tools.calls, http_requests=tools.http_requests, tool_errors=tools.errors,
                  incomplete_searches=sum(not s["complete"] for s in tools.searches),
                  elapsed_seconds=round(time.monotonic() - started, 3))
    # The shared system prompt and tools are stored once in the run header.
    return {"status": status, "error": error, "final": final, "retrieved_resource_ids": sorted(tools.seen),
            "metrics": totals, "messages": messages[1:]}

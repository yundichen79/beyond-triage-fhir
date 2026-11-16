"""Native Qwen tool calling with bounded episodes and an identical public contract."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import time

from dataset.common import canonical
from student.tools import PatientTools, tool_definitions as shared_tool_definitions
from student.contract import answer_error, contract_for_question, json_schema


MODEL_ID = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
MODEL_CACHE = Path(__file__).resolve().parents[1] / "vendor/models"

# Shared instructions plus one value-free schema selected from visible wording.
# Evaluator labels, family metadata, or supporting IDs never select the schema.
SYSTEM_PROMPT = """Answer one patient's clinical chart-review question using the FHIR tools.
Use only records obtained through these tools. Patient scope is automatic. Search
with useful dates, codes, encounter references, statuses and sorting. Follow a
search's next_page cursor when the remaining matches are needed. An incomplete
page or tool error cannot establish that a record does not exist. You may refine
failed searches. Resource references are authoritative for report results and
documented indications. An empty filtered search may mean the filters are wrong;
broaden it within the question's scope rather than repeatedly issuing the same
request. Do not impose an unstated recent-history cutoff on a complete chart.
Use Encounter.class IMP for inpatient and AMB for ambulatory. Search CBC
DiagnosticReports and read their result references. Arrays in search params
are repeated AND constraints; comma-separated strings express alternatives.
Inspect reasonReference on retrieved orders/procedures instead of searching it.
Return valid JSON: close every object and array before ending a tool_call block.
Orders do not establish medication administration or use.
Return UTC timestamps with timezone, coded concepts, stated units, and requested
calculations. Do not invent values, indications, discontinuation, or causal effects.
For a DiagnosticReport, report_time is effectiveDateTime (or effectivePeriod.start).
Copy the recorded timestamp exactly.
For changes requested to one decimal, compute later minus earlier and round the
change to one decimal using decimal half-even rounding. Do not round source measurements.
Finish with the finish tool, alone in its turn, after reading its evidence. A final
JSON object with keys answer and evidence, or a bare JSON tool call with keys
name and arguments, is also accepted. Cite ResourceType/id
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

The user message contains ONE answer JSON Schema for this question; finish's answer
must follow it.
Follow its required fields and nesting exactly. Conditional alternatives describe
records found versus absent, not different tasks. Do not include a task-name wrapper.
This schema specifies the answer format; it supplies no patient values or evidence.

Retrieval procedure:
First identify the requested anchor and time window from the question. Search the
complete history up to the stated cutoff; never supply a lower bound merely to
make the search smaller. For earliest/latest selection, sort ascending/descending
and use small pages. If the latest encounter lacks a required order or measurement,
continue to older encounters until the latest QUALIFYING encounter is found.
After selecting the anchor, derive the requested relative window from its dates.
For a preceding-year question, search BEFORE admission, not during the stay.
For alternative codes, classes, or encounters use a comma-separated string;
an array means all constraints must hold simultaneously. If a search is empty,
check these constraints and remove any assumptions not stated in the question.
For indications, retain only orders/procedures with their OWN explicit
reasonReference links. Do not copy one resource's indications onto other resources.
Avoid unrelated record types and repeatedly retrieving the same pages. Use small
pages and follow next_page only as required to establish the requested selection.
Use calculate for requested numeric differences, with decimal_places=1 when the
question asks for one decimal. Preserve the original measurement values.
Before finish, check the anchor, time window, required links, and rounding. Cite
retrieved anchor encounters/reports AND supporting measurements, orders, and
conditions. A correct value alone does not document its encounter or report context.
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
    max_repeated_failures: int = 4

    def __post_init__(self):
        for name in ("max_steps", "max_tool_calls", "max_context_tokens", "max_new_tokens", "max_output_tokens", "top_k", "max_repeated_failures"):
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
    def __init__(self, settings, model_path=None, adapter=None):
        try:
            import torch
            import transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ValueError("Model dependencies are missing; run bash scripts/prepare_model.sh") from exc
        if not torch.cuda.is_available():
            raise ValueError("Running the student model requires a GPU allocation; use scripts/run_student.sbatch")
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
            raise ValueError("Prepared model checkpoint is unavailable; run bash scripts/prepare_model.sh") from exc
        if adapter:
            # A LoRA adapter trained by sft.train, applied on top of the same pinned base model.
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise ValueError("Loading an adapter needs peft; run bash scripts/prepare_model.sh") from exc
            self.model = PeftModel.from_pretrained(self.model, str(adapter)).eval()
        self.metadata = {"model": source, "revision": getattr(self.model.config, "_commit_hash", None)
                         or (MODEL_REVISION if not model_path else None), "thinking": False,
                         "torch": torch.__version__, "transformers": transformers.__version__,
                         "gpu": torch.cuda.get_device_name(0), "dtype": "bfloat16",
                         "adapter": str(adapter) if adapter else None}

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
        if set(value) == {"name", "arguments"}:
            calls = [value]
        elif set(value) == {"answer", "evidence"}:
            calls = [{"name": "finish", "arguments": value}]
        else:
            raise ValueError("Use tool_call blocks, a JSON call with name and arguments, "
                             "or a final JSON object containing answer and evidence")
    for call in calls:
        if set(call) != {"name", "arguments"} or not isinstance(call["name"], str) or not isinstance(call["arguments"], dict):
            raise ValueError("Each call needs a name and an arguments object")
    if len(calls) > 1 and any(c["name"] == "finish" for c in calls):
        raise ValueError("Submit finish alone after observing tool results")
    return calls


def run_agent(question, patient, model, client, settings, seed):
    """Only question and patient context enter the model; all scoring is external."""
    contract_name, answer_shape = contract_for_question(question)
    answer_schema = json_schema(answer_shape)
    tool_definitions = shared_tool_definitions()
    tools = PatientTools(client, patient)
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Patient reference: {patient}\n\n{question}"
                 + "\n\nAnswer JSON Schema:\n" + canonical(answer_schema)}]
    started = time.monotonic()
    totals = {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "protocol_errors": 0,
              "truncated_turns": 0, "repeat_warnings": 0}
    status, error, final = "step_limit", None, None
    failures = {}

    def repeated_failure(key):
        count = failures.get(key, 0) + 1
        failures[key] = count
        stop = count >= settings.max_repeated_failures
        warning = None
        if count == settings.max_repeated_failures - 1:
            warning = (f"The identical failed action or empty search occurred {count} times "
                       "without retrieving new resources. Correct the format or change the search "
                       "within the question's scope. The next identical failure ends the episode; "
                       "do not invent an answer to avoid the limit.")
            totals["repeat_warnings"] += 1
        return warning, stop

    model.seed(seed)
    for _ in range(settings.max_steps):
        remaining = settings.max_output_tokens - totals["output_tokens"]
        if remaining <= 0:
            status = "output_token_limit"
            break
        totals["llm_calls"] += 1
        try:
            generation = model.generate(messages, tool_definitions,
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
            warning, stop = repeated_failure(("parse", generation.text.strip()))
            messages.extend([{"role": "assistant", "content": generation.text},
                {"role": "user", "content": "Output format error: " + str(exc)
                 + ". Use the tool signatures and public answer contract; do not guess missing evidence."
                 + (" " + warning if warning else "")}])
            if stop:
                status, error = "repeated_failure", "Identical malformed output repeated without new resources"
                break
            if totals["output_tokens"] >= settings.max_output_tokens:
                status = "output_token_limit"
                break
            continue
        messages.append({"role": "assistant", "content": generation.text.split("<tool_call>", 1)[0].strip() if "<tool_call>" in generation.text else "",
                         "tool_calls": [{"type": "function", "function": call} for call in calls]})
        for call in calls:
            name, arguments = call["name"], call["arguments"]
            before_seen = len(tools.seen)
            if name == "finish":
                evidence = arguments.get("evidence")
                if (set(arguments) != {"answer", "evidence"} or not isinstance(arguments["answer"], dict)
                        or not isinstance(evidence, list) or any(not isinstance(ref, str) for ref in evidence)):
                    result = {"error": "finish requires an answer object and an evidence array of references"}
                elif set(evidence) - tools.seen:
                    result = {"error": "Citations must refer to resources actually returned by tools"}
                elif (schema_error := answer_error(arguments["answer"], answer_shape)):
                    result = {"error": schema_error}
                else:
                    final = {"answer": arguments["answer"], "evidence": sorted(set(evidence))}
                    status = "completed"
                    break
                totals["protocol_errors"] += 1
            elif tools.calls >= settings.max_tool_calls:
                status, error = "tool_call_limit", "The episode exhausted its tool-call budget"
                break
            else:
                result = tools.execute(name, arguments)
            if len(tools.seen) > before_seen:
                failures.clear()
            failed = "error" in result
            empty_search = (name == "search_fhir" and result.get("complete") is True
                            and result.get("resources") == [] and result.get("total") == 0)
            stop = False
            if failed or empty_search:
                warning, stop = repeated_failure(("call", canonical(call)))
                if warning:
                    result["retry_warning"] = warning
            messages.append({"role": "tool", "name": name, "content": canonical(result)})
            if stop:
                status, error = "repeated_failure", "Identical failed action or empty search repeated without new resources"
                break
        if final is not None or status in {"tool_call_limit", "repeated_failure"}:
            break
        if totals["output_tokens"] >= settings.max_output_tokens:
            status = "output_token_limit"
            break
    totals.update(tool_calls=tools.calls, http_requests=tools.http_requests, tool_errors=tools.errors,
                  incomplete_searches=sum(not s["complete"] for s in tools.searches),
                  elapsed_seconds=round(time.monotonic() - started, 3))
    # The shared system prompt and tools are stored once in the run header.
    return {"status": status, "error": error, "final": final, "answer_contract": contract_name,
            "retrieved_resource_ids": sorted(tools.seen),
            "metrics": totals, "messages": messages[1:]}

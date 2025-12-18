#!/usr/bin/env python3
# dcr_llm_controller.py — LLM controller that generates and self-validates DCR graphs
#
# Pipeline:
# - Get raw text (law/rules/description) from CLI or interactive input.
# - (Optional) Preprocess into a structured SPEC via LLM.
# - Generate DCR JSON via LLM in a self-repair loop using dcr_validate.py.
# - Wrap into DCR Import format: {"DCRModel":[{...}]}.
# - Evaluate each run with prompt metrics and append to a JSONL log.

import argparse
import json
import os
import time
import uuid
from datetime import datetime
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv
from openai import OpenAI

from dcr_validate import validate_raw_json


# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------

load_dotenv()
client = OpenAI()

# Change this to a model you have access to, e.g. "gpt-4.1-mini" or "gpt-4o-mini"
DEFAULT_MODEL = "gpt-5.1"


SYSTEM_PROMPT_GENERATOR = """
You are an expert in Dynamic Condition Response (DCR) Graphs.

Given a structured specification of a process (plus optionally the original legal text),
you MUST output a single JSON object representing a DCR graph.

The JSON you output must have this structure:

{
  "title": "short name of the process",
  "description": "short description of the process",
  "events": [
    {
      "id": "UNIQUE_MACHINE_FRIENDLY_ID",
      "label": "Human readable label"
    }
  ],
  "rules": [
    {
      "type": "condition | response | milestone | include | exclude",
      "source": "event_id",
      "target": "event_id",
      "description": "Short explanation of the relation",
      "duration": "optional ISO-8601 duration, e.g. PT5S"
    }
  ]
}

Rules and constraints:
- Use ONLY event IDs that appear in the events array.
- Use ONLY allowed rule types: condition, response, milestone, include, exclude.
- Every event MUST have a non-empty "id" and "label".
- Event IDs MUST be unique.
- You may omit "duration" if not needed.
- Do NOT include roles or marking — only title, description, events, rules.
- Output ONLY the JSON object. No explanations, no markdown.
""".strip()


SYSTEM_PROMPT_PREPROCESSOR = """
You are a requirements engineer specialized in Dynamic Condition Response (DCR) Graphs.

You will receive long, messy text: laws, regulations, descriptions, and rules.
Your job is to transform this into a concise, structured specification that is easy
to turn into a DCR graph.

Output a SPEC ONLY, with this structure in plain text:

Title: <short name of process>

Actors:
- <actor 1>
- <actor 2>
...

Main Events (what can happen in the process):
- <event 1>
- <event 2>
...

Constraints and Ordering:
- <constraints about ordering, conditions, responses, milestones>
- <e.g. "Event X requires Event Y before it", "If A happens then B must eventually happen">

Guards / Conditions (optional):
- <any data or state based conditions, if relevant; otherwise say "None">

Ignore legal boilerplate and focus ONLY on the behavioural rules relevant to the process.
Do NOT output JSON. Just the structured text in this format.
""".strip()


# ---------------------------------------------------------------------
# LLM helper functions
# ---------------------------------------------------------------------

def preprocess_spec(raw_text: str, model_name: str, verbose: bool = True) -> str:
    """
    Use the LLM once to turn a long messy text (laws, rules, etc.)
    into a structured specification that is DCR-friendly.
    """
    if verbose:
        print("\n=== Preprocessing specification with LLM ===")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_PREPROCESSOR},
        {
            "role": "user",
            "content": (
                "Here is the raw text describing the process. "
                "Please convert it into the structured SPEC format:\n\n"
                + raw_text
            ),
        },
    ]

    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
    )

    structured_spec = resp.choices[0].message.content.strip()

    if verbose:
        print("\n=== Structured SPEC (LLM output) ===")
        print(structured_spec)
        print("====================================")

    return structured_spec


def call_dcr_generator(messages: List[Dict[str, str]], model_name: str) -> str:
    """
    Call the Chat Completions API with JSON mode enforced.
    Returns the raw JSON string from the model.
    """
    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content


def generate_dcr_graph(
    structured_spec: str,
    original_text: str,
    model_name: str,
    max_retries: int = 5,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Main loop:
    1) Ask model for JSON using the structured spec + original text.
    2) Run symbolic validation.
    3) If errors, re-prompt with error messages and ask for a corrected graph.

    Returns:
      (model_dict, stats_dict) where stats_dict contains:
        - num_attempts
        - num_validation_failures
    """

    combined_description = (
        "Structured SPEC (primary source for the DCR graph):\n"
        f"{structured_spec}\n\n"
        "Original raw text (secondary, for context only):\n"
        f"{original_text}\n"
    )

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT_GENERATOR},
        {"role": "user", "content": combined_description},
    ]

    last_json: str | None = None
    num_attempts = 0
    num_validation_failures = 0

    for attempt in range(1, max_retries + 1):
        num_attempts = attempt

        if verbose:
            print(f"\n=== DCR Generation Attempt {attempt} ===")

        raw_json = call_dcr_generator(messages, model_name=model_name)
        last_json = raw_json

        # Try validation
        try:
            model_obj, errors = validate_raw_json(raw_json)
        except json.JSONDecodeError as e:
            errors = [f"Model output was not valid JSON: {e}"]
            model_obj = None
        except SystemExit as e:
            # load_dcrmodel may call SystemExit on structural problems
            errors = [f"Structural error while loading DCRModel: {e}"]
            model_obj = None

        if errors:
            num_validation_failures += 1

        if not errors and model_obj is not None:
            if verbose:
                print("Validation passed ✅")
            stats = {
                "num_attempts": num_attempts,
                "num_validation_failures": num_validation_failures,
            }
            return model_obj, stats

        # If we reach here, there were errors.
        error_text = "\n".join(f"- {err}" for err in errors)
        if verbose:
            print("Validation errors:")
            print(error_text)

        # Append previous JSON as assistant content, then ask to fix it
        messages.append(
            {
                "role": "assistant",
                "content": raw_json,
            }
        )
        messages.append(
            {
                "role": "user",
                "content": (
                    "Your previous JSON DCR graph failed validation with the "
                    "following problems:\n"
                    f"{error_text}\n\n"
                    "Please output a FULLY CORRECTED JSON object that fixes "
                    "ALL of these issues. Output ONLY the JSON, no comments."
                ),
            }
        )

    # If we still don't have a valid graph after max_retries:
    raise RuntimeError(
        f"Failed to produce a valid DCR graph after {max_retries} attempts.\n"
        f"Last JSON from model was:\n{last_json}"
    )


# ---------------------------------------------------------------------
# Sanitisation for DCR Import format
# ---------------------------------------------------------------------

def _sanitize_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Keep only id + label (and optional description).
    """
    clean_events: List[Dict[str, Any]] = []
    for ev in events:
        if not isinstance(ev, dict):
            continue

        eid = str(ev.get("id", "")).strip()
        label = str(ev.get("label", "")).strip()
        desc = ev.get("description")

        if not eid:
            continue
        if not label:
            label = eid

        ev_clean: Dict[str, Any] = {"id": eid, "label": label}
        if isinstance(desc, str) and desc.strip():
            ev_clean["description"] = desc.strip()

        clean_events.append(ev_clean)

    return clean_events


def _sanitize_rules(rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Keep only type, source, target, description, duration.
    """
    clean_rules: List[Dict[str, Any]] = []
    for r in rules:
        if not isinstance(r, dict):
            continue

        rtype = str(r.get("type", "")).strip()
        src = str(r.get("source", "")).strip()
        tgt = str(r.get("target", "")).strip()
        desc = r.get("description")
        dur = r.get("duration")

        if not rtype or not src or not tgt:
            continue

        r_clean: Dict[str, Any] = {
            "type": rtype,
            "source": src,
            "target": tgt,
        }

        if isinstance(desc, str) and desc.strip():
            r_clean["description"] = desc.strip()
        if isinstance(dur, str) and dur.strip():
            r_clean["duration"] = dur.strip()

        clean_rules.append(r_clean)

    return clean_rules


# ---------------------------------------------------------------------
# Prompt metrics
# ---------------------------------------------------------------------

def estimate_difficulty_score(
    raw_prompt_words: int,
    structured_words: int,
    num_attempts: int,
) -> float:
    """
    Heuristic difficulty score in [1, 10].
    Bigger for long prompts, dense structured specs, and many attempts.

    - base from raw_prompt_words: 1..5 (1 per 100 words, capped at 5)
    - structure_bonus from structured_words: 0..3 (1 per 80 words, capped at 3)
    - repair_penalty: +0.5 per extra attempt beyond first, capped at +2
    """
    base = min(max(raw_prompt_words / 100.0, 1.0), 5.0)
    struct_bonus = min(structured_words / 80.0, 3.0)
    repair_penalty = min(max(num_attempts - 1, 0) * 0.5, 2.0)

    score = base + struct_bonus + repair_penalty
    return float(min(max(score, 1.0), 10.0))


def compute_quality_score(
    one_shot_success: bool,
    num_attempts: int,
    raw_prompt_words: int,
    total_time_seconds: float,
) -> Tuple[float, str]:
    """
    Prompt Quality Score in [0, 100], higher is better.

    Intuition:
      - Start at 100.
      - Penalty for extra attempts (self-repair).
      - Penalty for very long prompts.
      - Penalty for slow runs.

    Returns: (score, qualitative_label)
    """
    score = 100.0

    # Attempts penalty: each extra attempt costs 12 points (strong incentive)
    score -= max(num_attempts - 1, 0) * 12.0

    # Length penalty: 0..20 depending on length (0 if <=200 words, up to 20 if very long)
    if raw_prompt_words > 200:
        length_penalty = min((raw_prompt_words - 200) / 100.0 * 4.0, 20.0)
        score -= length_penalty

    # Time penalty: 0..25 (0 if <= 3s, up to 25 for 30+ seconds)
    if total_time_seconds > 3.0:
        time_penalty = min((total_time_seconds - 3.0) / 3.0 * 5.0, 25.0)
        score -= time_penalty

    # Bonus for true one-shot success
    if one_shot_success:
        score += 5.0

    score = max(min(score, 100.0), 0.0)

    if score >= 85:
        level = "excellent"
    elif score >= 70:
        level = "good"
    elif score >= 50:
        level = "fair"
    else:
        level = "poor"

    return float(score), level


def append_metrics_log(
    path: str,
    record: Dict[str, Any],
) -> None:
    """
    Append a single JSON object as one line to a JSONL file.
    """
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False))
        f.write("\n")


# ---------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------

def load_spec_interactive() -> str:
    """
    Ask the user to paste a long, possibly multi-line spec.
    The user finishes by typing a single line containing 'END'.
    """
    print(
        "No --spec or --spec-file provided.\n"
        "Paste or write your process description / law / rules below.\n"
        "When you're done, type a single line containing 'END' and press Enter.\n"
    )

    lines: List[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break

        if line.strip() == "END":
            break
        lines.append(line)

    text = "\n".join(lines).strip()
    if not text:
        raise SystemExit("ERROR: No description text provided.")

    return text


def load_spec_from_args(args: argparse.Namespace) -> str:
    if args.spec_file:
        with open(args.spec_file, "r", encoding="utf-8") as f:
            return f.read().strip()

    if args.spec:
        return args.spec.strip()

    return load_spec_interactive()


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Generate and self-validate a DCR graph using the ChatGPT API."
    )
    ap.add_argument(
        "--spec",
        help="Short natural-language description of the process. "
             "For long laws/rules, consider omitting this and using interactive input.",
    )
    ap.add_argument(
        "--spec-file",
        help="Path to a text file with the process description. "
             "If provided, overrides --spec.",
    )
    ap.add_argument(
        "--task-label",
        default="",
        help="Optional label for this experiment (e.g. 'law_v1_long', 'casual_description').",
    )
    ap.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum number of LLM repair attempts (default: 5).",
    )
    ap.add_argument(
        "--output",
        default="generated_dcr.json",
        help="Where to save the final DCR JSON (default: generated_dcr.json).",
    )
    ap.add_argument(
        "--metrics-output",
        default="prompt_metrics.jsonl",
        help="JSONL file where prompt metrics will be appended (default: prompt_metrics.jsonl).",
    )
    ap.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI Chat Completions model name (default: {DEFAULT_MODEL}).",
    )
    ap.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip the LLM preprocessing step and feed the raw text directly to the DCR generator.",
    )
    return ap.parse_args()


def main() -> None:
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "ERROR: OPENAI_API_KEY is not set. "
            "Create a .env file with OPENAI_API_KEY=... or set it in your environment."
        )

    args = parse_args()
    model_name = args.model

    run_id = str(uuid.uuid4())
    start_time = time.time()

    # 1) Raw description
    raw_spec = load_spec_from_args(args)

    print("\n=== Raw description (user input) ===")
    print(raw_spec)
    print("====================================")

    raw_chars = len(raw_spec)
    raw_words = len(raw_spec.split())

    # 2) Structured spec
    if args.skip_preprocess:
        structured_spec = raw_spec
    else:
        structured_spec = preprocess_spec(raw_spec, model_name=model_name, verbose=True)

    struct_chars = len(structured_spec)
    struct_words = len(structured_spec.split())

    # 3) Generate DCR inner model + stats
    dcr_model, gen_stats = generate_dcr_graph(
        structured_spec=structured_spec,
        original_text=raw_spec,
        model_name=model_name,
        max_retries=args.max_retries,
        verbose=True,
    )

    num_attempts = gen_stats["num_attempts"]
    num_validation_failures = gen_stats["num_validation_failures"]

    # 4) Sanitize events and rules
    events = dcr_model.get("events", [])
    rules = dcr_model.get("rules", [])

    if not isinstance(events, list):
        raise SystemExit("Generated model 'events' is not a list.")
    if not isinstance(rules, list):
        raise SystemExit("Generated model 'rules' is not a list.")

    clean_events = _sanitize_events(events)
    clean_rules = _sanitize_rules(rules)

    if not clean_events:
        raise SystemExit("After sanitisation, no valid events remain.")
    if not clean_rules:
        raise SystemExit("After sanitisation, no valid rules remain.")

    event_ids = [ev["id"] for ev in clean_events]

    # 5) Wrap in the DCRModel format
    wrapped = {
        "DCRModel": [
            {
                "title": dcr_model.get("title", "Generated DCR Model"),
                "description": dcr_model.get(
                    "description",
                    "Generated via LLM controller"
                ),
                "type": "DCRModel",
                "roles": [],
                "events": clean_events,
                "rules": clean_rules,
                "marking": {
                    "executed": [],
                    "included": event_ids,  # all events initially included
                    "pending": []
                }
            }
        ]
    }

    # 6) Save + print JSON
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(wrapped, f, indent=2, ensure_ascii=False)

    print(f"\nFinal DCR model written to: {args.output}")
    print("\nDCR JSON:")
    print(json.dumps(wrapped, indent=2, ensure_ascii=False))

    # 7) Compute metrics
    total_time = time.time() - start_time
    one_shot = (num_attempts == 1 and num_validation_failures == 0)

    difficulty_score = estimate_difficulty_score(
        raw_prompt_words=raw_words,
        structured_words=struct_words,
        num_attempts=num_attempts,
    )
    quality_score, quality_level = compute_quality_score(
        one_shot_success=one_shot,
        num_attempts=num_attempts,
        raw_prompt_words=raw_words,
        total_time_seconds=total_time,
    )

    metrics_record = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "task_label": args.task_label,
        "model_name": model_name,

        "raw_prompt_length_chars": raw_chars,
        "raw_prompt_length_words": raw_words,
        "structured_spec_length_chars": struct_chars,
        "structured_spec_length_words": struct_words,

        "num_generation_attempts": num_attempts,
        "num_validation_failures": num_validation_failures,
        "one_shot_success": one_shot,
        "total_time_seconds": total_time,

        "difficulty_score": difficulty_score,
        "quality_score": quality_score,
        "quality_level": quality_level,
    }

    # 8) Log metrics
    append_metrics_log(args.metrics_output, metrics_record)

    # 9) Print a small summary for you
    print("\n=== Prompt Evaluation Metrics ===")
    print(f"Run ID:                 {run_id}")
    if args.task_label:
        print(f"Task label:             {args.task_label}")
    print(f"Model:                  {model_name}")
    print(f"Raw prompt:             {raw_words} words ({raw_chars} chars)")
    print(f"Structured SPEC:        {struct_words} words ({struct_chars} chars)")
    print(f"Attempts:               {num_attempts}")
    print(f"Validation failures:    {num_validation_failures}")
    print(f"One-shot success:       {one_shot}")
    print(f"Total time:             {total_time:.2f} s")
    print(f"Difficulty score:       {difficulty_score:.2f} / 10")
    print(f"Quality score:          {quality_score:.1f} / 100  ({quality_level})")
    print(f"Metrics appended to:    {args.metrics_output}")
    print("================================")


if __name__ == "__main__":
    main()

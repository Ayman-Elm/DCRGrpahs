#!/usr/bin/env python3
# dcr_validate.py — local validator for simple DCR JSON
#
# It checks:
# - JSON is valid
# - DCRModel[0].events exists and all events have non-empty "id"
# - DCRModel[0].rules exists and each rule has type/source/target
# - rule source/target must refer to an existing event id
# - event ids are unique

import json
import sys
import argparse
from typing import Any, Dict, List, Tuple


def read_text(path: str) -> str:
    if path in ("-", "/dev/stdin"):
        return sys.stdin.read()
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_dcrmodel(obj: Any) -> Dict[str, Any]:
    """
    Supported shapes:
      { "DCRModel": [ { ... } ] }
      or just { ... } with events/rules
    """
    if isinstance(obj, dict) and "DCRModel" in obj:
        arr = obj["DCRModel"]
        if isinstance(arr, list) and arr:
            return arr[0]
        raise SystemExit("ERROR: 'DCRModel' must be a non-empty array.")
    if isinstance(obj, dict) and "events" in obj and "rules" in obj:
        return obj
    raise SystemExit("ERROR: JSON is not a simple DCRModel (no 'DCRModel' or 'events'/'rules').")


def validate_dcrmodel(m: Dict[str, Any]) -> List[str]:
    errors: List[str] = []

    events = m.get("events")
    if not isinstance(events, list) or not events:
        errors.append("No 'events' array or it is empty.")
        return errors

    rules = m.get("rules")
    if not isinstance(rules, list):
        errors.append("No 'rules' array.")
        return errors

    # Collect event ids
    event_ids: List[str] = []
    for idx, ev in enumerate(events):
        if not isinstance(ev, dict):
            errors.append(f"Event #{idx} is not an object.")
            continue
        eid = ev.get("id")
        if not isinstance(eid, str) or not eid.strip():
            errors.append(f"Event #{idx} missing non-empty 'id'.")
            continue
        event_ids.append(eid)

    # Check for duplicate event IDs
    if len(event_ids) != len(set(event_ids)):
        errors.append("Duplicate event 'id' values are not allowed.")

    event_id_set = set(event_ids)

    # Allowed rule types (extend if needed)
    allowed_types = {"condition", "response", "milestone", "include", "exclude"}

    for idx, r in enumerate(rules):
        if not isinstance(r, dict):
            errors.append(f"Rule #{idx} is not an object.")
            continue

        rtype = r.get("type")
        src = r.get("source")
        tgt = r.get("target")

        if not isinstance(rtype, str) or not rtype.strip():
            errors.append(f"Rule #{idx} missing 'type'.")
        else:
            if rtype.lower() not in allowed_types:
                errors.append(
                    f"Rule #{idx} has unknown type '{rtype}'. "
                    f"Allowed: {', '.join(sorted(allowed_types))}."
                )

        if not isinstance(src, str) or not src.strip():
            errors.append(f"Rule #{idx} missing 'source'.")
        elif src not in event_id_set:
            errors.append(f"Rule #{idx} source '{src}' not found in events.")

        if not isinstance(tgt, str) or not tgt.strip():
            errors.append(f"Rule #{idx} missing 'target'.")
        elif tgt not in event_id_set:
            errors.append(f"Rule #{idx} target '{tgt}' not found in events.")

    return errors


def validate_raw_json(raw: str) -> Tuple[Dict[str, Any], List[str]]:
    """
    Helper for programmatic use:
    - parses raw JSON string
    - extracts the DCRModel
    - returns (model_dict, list_of_errors)
    """
    obj = json.loads(raw)
    model = load_dcrmodel(obj)
    errors = validate_dcrmodel(model)
    return model, errors


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Local validator for simple DCR JSON."
    )
    ap.add_argument(
        "--payload-file",
        default="vending.txt",
        help="File containing DCR JSON (default: vending.txt)",
    )
    args = ap.parse_args()

    raw = read_text(args.payload_file).strip()
    if not raw:
        raise SystemExit(f"ERROR: {args.payload_file} is empty.")

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        raise SystemExit(f"ERROR: Invalid JSON in {args.payload_file}: {e}")

    model = load_dcrmodel(obj)
    errors = validate_dcrmodel(model)

    if errors:
        print(" INVALID DCR JSON")
        for e in errors:
            print(" -", e)
        sys.exit(1)
    else:
        title = model.get("title") or "Untitled"
        print(f" VALID DCR JSON for model: {title}")
        sys.exit(0)


if __name__ == "__main__":
    main()

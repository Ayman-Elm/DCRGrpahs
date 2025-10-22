#!/usr/bin/env python3
# dcr_validate.py — validate JSON against DCR Repository (/api/graphs/validate)

import argparse, json, sys, requests
from requests.auth import HTTPBasicAuth

# --- TEST CREDS (replace for your quick test) ---
DCR_USERNAME = "ayman.elm01@gmail.com"      # <-- put your DCR email here
DCR_PASSWORD = "AymanDCR123"   # <-- put your DCR password here
# -------------------------------------------------

FILEPATH = "test.json"

BASE_URL = "https://repository.dcrgraphs.net"
VALIDATE_PATH = "/api/graphs/validate"  # you just hit this successfully

def auth_probe(base: str, user: str, pwd: str, timeout: int = 30):
    url = base.rstrip("/") + "/api/graphs"
    r = requests.get(url, auth=HTTPBasicAuth(user, pwd), timeout=timeout)
    if r.status_code != 200:
        if r.status_code == 401:
            raise SystemExit("Auth failed (401). Check username/password.")
        if r.status_code == 403:
            raise SystemExit("Forbidden (403). Basic Auth missing/invalid.")
        raise SystemExit(f"Auth probe unexpected {r.status_code}: {r.text[:300]}")

def read_json_bytes(path: str) -> bytes:
    raw = sys.stdin.read() if path in ("-", "/dev/stdin") else open(path, "r", encoding="utf-8").read()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise SystemExit(f"ERROR: Invalid JSON: {e}")
    return json.dumps(parsed, ensure_ascii=False).encode("utf-8")

def post_validate(url: str, payload: bytes, user: str, pwd: str, timeout: int):
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json,text/plain,*/*"
    }
    return requests.post(url, data=payload, headers=headers,
                         auth=HTTPBasicAuth(user, pwd), timeout=timeout)

def main():
    ap = argparse.ArgumentParser(description="Validate JSON payload via /api/graphs/validate.")
    ap.add_argument("--base-url", default=BASE_URL)
    ap.add_argument("--endpoint", default=VALIDATE_PATH)
    ap.add_argument("--payload-file", default= FILEPATH, help="JSON file to validate (or '-' for stdin)")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--user")
    ap.add_argument("--password")
    ap.add_argument("--verbose", action="store_true", help="Print response headers too")
    args = ap.parse_args()

    user = args.user or DCR_USERNAME
    pwd  = args.password or DCR_PASSWORD
    if not user or not pwd:
        raise SystemExit("Fill DCR_USERNAME/DCR_PASSWORD at top or pass --user/--password")

    base = args.base_url.rstrip("/")
    url = base + args.endpoint

    # 1) prove auth works
    print("Checking Basic Auth via /api/graphs …")
    auth_probe(base, user, pwd, args.timeout)
    print("✓ Auth OK")

    # 2) load JSON + call validate
    payload = read_json_bytes(args.payload_file)
    print(f"Posting JSON to: {url}")
    try:
        resp = post_validate(url, payload, user, pwd, args.timeout)
    except requests.RequestException as e:
        raise SystemExit(f"Request failed: {e}")

    # 3) show status (+headers if verbose)
    print(f"HTTP {resp.status_code}")
    if args.verbose:
        for k, v in resp.headers.items():
            print(f"{k}: {v}")

    # 4) interpret common outcomes
    body_text = resp.text or ""
    if resp.status_code in (200, 204):
        if not body_text.strip():
            print("✅ VALID")
        else:
            # Sometimes servers return a minimal message; show it
            print(body_text)
        sys.exit(0)

    # If not 2xx, try to show JSON errors nicely
    try:
        err = resp.json()
        print(json.dumps(err, indent=2, ensure_ascii=False))
    except ValueError:
        print(body_text if body_text else "<no body>")

    # Suggestive hints for typical failures
    if resp.status_code == 400:
        print("❌ Validation failed (400). Check your JSON against the expected schema.")
    elif resp.status_code == 415:
        print("❌ Unsupported media type (415). Ensure Content-Type is application/json.")
    elif resp.status_code in (401, 403):
        print("❌ Auth issue despite probe; re-check credentials/permissions.")

    sys.exit(1)

if __name__ == "__main__":
    main()

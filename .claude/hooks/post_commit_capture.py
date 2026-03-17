"""Post-tool-use hook: detects git commits and logs them for moment annotation."""

import json
import os
import re
import sys
from datetime import datetime, timezone

def main():
    raw = sys.stdin.read()
    if not raw.strip():
        return

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return

    command = data.get("tool_input", {}).get("command", "")
    if "git commit" not in command:
        return

    response = data.get("tool_response", {})
    stdout = response.get("stdout", "")
    exit_code = response.get("exit_code", 1)

    if exit_code != 0 or not stdout:
        return

    # Extract commit hash from output like "[master abc1234] message"
    match = re.search(r"\[[\w/.-]+ ([a-f0-9]+)\]", stdout)
    commit_hash = match.group(1) if match else "unknown"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    project_dir = os.environ.get("CLAUDE_PROJECT_DIR", ".")
    pending_file = os.path.join(project_dir, ".claude", "hooks", "pending_moments.jsonl")

    os.makedirs(os.path.dirname(pending_file), exist_ok=True)
    with open(pending_file, "a", encoding="utf-8") as f:
        json.dump({"commit": commit_hash, "timestamp": timestamp, "command": command}, f)
        f.write("\n")

    print(f"[dream-forge] Commit {commit_hash} logged for moment annotation", file=sys.stderr)

if __name__ == "__main__":
    main()

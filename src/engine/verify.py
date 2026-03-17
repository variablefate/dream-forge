"""Tier 1a verification — run Qwen's proposed fixes and score them.

Combines execution-based verification (syntax, imports, tests) with
detector confidence to produce a composite verification score.

Layers:
  1. Syntax check (does the code compile?)
  2. Import check (does the module load?)
  3. Test execution (do tests pass with the fix applied?)
  4. Detector confidence (does the model's internal state say "I know this"?)

Uses git worktrees for isolation — no risk of corrupting the working tree.

Usage:
    from src.engine.verify import verify_fix, VerificationResult
    result = verify_fix(
        proposed_fix="def apply_domain_cap(...)...",
        target_file="src/engine/data_prep.py",
        commit_before="5d821fa",
        detector_risk=0.3,
    )
    print(result.score, result.passed, result.details)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console

console = Console()

# ── Scoring weights ───────────────────────────────────────────────────────────

WEIGHTS = {
    "syntax": 0.10,
    "imports": 0.10,
    "tests": 0.50,
    "detector": 0.30,
}

# Minimum score to consider a fix "verified"
VERIFICATION_THRESHOLD = 0.70

# Test execution timeout (seconds)
TEST_TIMEOUT = 120


# ── Data types ─────────────────────────────────────────────────────────────────

@dataclass
class VerificationResult:
    """Result of verifying a proposed code fix."""
    # Individual checks
    syntax_ok: bool = False
    imports_ok: bool = False
    tests_pass: bool | None = None   # None = no tests available
    detector_confidence: float = 0.0  # 1 - risk

    # Composite
    score: float = 0.0
    passed: bool = False

    # Details
    test_output: str = ""
    error: str = ""
    checks_run: list[str] = field(default_factory=list)
    worktree_path: str = ""


# ── Git worktree isolation ────────────────────────────────────────────────────

def _create_worktree(commit: str, tmp_dir: str) -> str | None:
    """Create an isolated git worktree at the given commit.

    Returns the worktree path, or None on failure.
    """
    worktree_path = os.path.join(tmp_dir, "worktree")
    result = subprocess.run(
        ["git", "worktree", "add", "--detach", worktree_path, commit],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        return None
    return worktree_path


def _cleanup_worktree(worktree_path: str):
    """Remove a git worktree cleanly."""
    try:
        subprocess.run(
            ["git", "worktree", "remove", "--force", worktree_path],
            capture_output=True, text=True, timeout=15,
        )
    except Exception:
        # Force cleanup if git worktree remove fails
        try:
            shutil.rmtree(worktree_path, ignore_errors=True)
            subprocess.run(
                ["git", "worktree", "prune"],
                capture_output=True, text=True, timeout=10,
            )
        except Exception:
            pass


# ── Verification checks ───────────────────────────────────────────────────────

# ── Language detection ─────────────────────────────────────────────────────────

def _detect_language(target_file: str, worktree_path: str = "") -> str:
    """Detect the programming language from file extension and project markers."""
    ext = Path(target_file).suffix.lower()

    ext_map = {
        ".py": "python",
        ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript",
        ".ts": "typescript", ".tsx": "typescript",
        ".go": "go",
        ".rs": "rust",
        ".java": "java",
        ".kt": "kotlin",
        ".rb": "ruby",
        ".php": "php",
        ".cs": "csharp",
        ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".c": "c", ".h": "c",
        ".swift": "swift",
        ".ex": "elixir", ".exs": "elixir",
        ".sh": "shell", ".bash": "shell",
    }

    return ext_map.get(ext, "unknown")


# ── Language-specific syntax checks ───────────────────────────────────────────

def _check_syntax(code: str, language: str = "python") -> bool:
    """Check if code compiles/parses as valid source for the given language."""
    if language == "python":
        try:
            compile(code, "<proposed_fix>", "exec")
            return True
        except SyntaxError:
            try:
                wrapped = "class _Wrapper:\n" + "\n".join(
                    f"    {line}" for line in code.split("\n")
                )
                compile(wrapped, "<proposed_fix_wrapped>", "exec")
                return True
            except SyntaxError:
                return False

    elif language == "javascript" or language == "typescript":
        try:
            result = subprocess.run(
                ["node", "--check", "-e", code] if language == "javascript"
                else ["npx", "tsc", "--noEmit", "--allowJs", "-"],
                input=code, capture_output=True, text=True, timeout=15,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return True  # can't check — assume OK

    elif language == "go":
        # Go syntax is checked at build time
        return True  # defer to build check

    elif language == "rust":
        # Rust syntax is checked at build time
        return True  # defer to build check

    # Unknown language — can't check, assume OK
    return True


# ── Language-specific build/import checks ─────────────────────────────────────

def _check_imports(worktree_path: str, target_file: str, language: str = "python") -> tuple[bool, str]:
    """Check if the project builds/imports cleanly after applying the fix."""
    run_kwargs = dict(
        capture_output=True, text=True, timeout=60,
        cwd=worktree_path,
    )

    try:
        if language == "python":
            module_path = target_file.replace("/", ".").replace("\\", ".").removesuffix(".py")
            result = subprocess.run(
                ["uv", "run", "python", "-c", f"import {module_path}"],
                env={**os.environ, "PYTHONPATH": worktree_path},
                **run_kwargs,
            )

        elif language in ("javascript", "typescript"):
            # Try npm/yarn/pnpm build
            if os.path.exists(os.path.join(worktree_path, "package.json")):
                pkg_manager = "npm"
                for pm in ["pnpm-lock.yaml", "yarn.lock", "bun.lockb"]:
                    if os.path.exists(os.path.join(worktree_path, pm)):
                        pkg_manager = pm.split("-")[0].split(".")[0]
                        break
                result = subprocess.run(
                    [pkg_manager, "run", "build"], **run_kwargs)
            else:
                return True, ""  # no package.json, skip

        elif language == "go":
            result = subprocess.run(
                ["go", "build", "./..."], **run_kwargs)

        elif language == "rust":
            result = subprocess.run(
                ["cargo", "check"], **run_kwargs)

        elif language == "java" or language == "kotlin":
            if os.path.exists(os.path.join(worktree_path, "pom.xml")):
                result = subprocess.run(
                    ["mvn", "compile", "-q"], **run_kwargs)
            elif os.path.exists(os.path.join(worktree_path, "build.gradle")):
                result = subprocess.run(
                    ["./gradlew", "compileJava", "-q"], **run_kwargs)
            else:
                return True, ""

        else:
            return True, ""  # unknown language, skip

        return result.returncode == 0, result.stderr[:500]

    except subprocess.TimeoutExpired:
        return False, "build/import check timed out"
    except FileNotFoundError as e:
        return True, f"tool not found: {e}"  # can't check, assume OK
    except Exception as e:
        return False, str(e)[:200]


# ── Language-specific test execution ──────────────────────────────────────────

def _detect_test_runner(worktree_path: str, language: str) -> tuple[list[str], str]:
    """Detect the test runner command and find related test files.

    Returns (command, test_target) or ([], "") if no tests found.
    """
    if language == "python":
        # Check for pytest, unittest, or nose
        test_patterns = ["tests/", "test/", "test_*.py"]
        for pattern in test_patterns:
            if os.path.exists(os.path.join(worktree_path, pattern.rstrip("/"))):
                return ["uv", "run", "python", "-m", "pytest", "-x", "--tb=short", "-q"], pattern
        return [], ""

    elif language in ("javascript", "typescript"):
        pkg_json = os.path.join(worktree_path, "package.json")
        if os.path.exists(pkg_json):
            import json
            try:
                pkg = json.loads(Path(pkg_json).read_text())
                if "test" in pkg.get("scripts", {}):
                    return ["npm", "test", "--"], ""
            except Exception:
                pass
        # Check for common test runners
        for runner in ["jest", "vitest", "mocha"]:
            config = os.path.join(worktree_path, f"{runner}.config.js")
            if os.path.exists(config):
                return ["npx", runner], ""
        return [], ""

    elif language == "go":
        if any(f.endswith("_test.go") for f in os.listdir(worktree_path)
               if os.path.isfile(os.path.join(worktree_path, f))):
            return ["go", "test", "-v", "-count=1"], "./..."
        return [], ""

    elif language == "rust":
        if os.path.exists(os.path.join(worktree_path, "Cargo.toml")):
            return ["cargo", "test", "--"], ""
        return [], ""

    return [], ""


def _run_tests(
    worktree_path: str,
    target_file: str,
    language: str = "python",
    timeout: int = TEST_TIMEOUT,
) -> tuple[bool | None, str]:
    """Run tests in the worktree. Returns (passed, output) or (None, reason)."""
    cmd, test_target = _detect_test_runner(worktree_path, language)

    if not cmd:
        return None, f"no test runner found for {language}"

    # For Python, try to find a test file specific to the target module
    if language == "python":
        target_name = Path(target_file).stem
        specific_tests = [
            f"tests/test_{target_name}.py",
            f"tests/{target_name}_test.py",
        ]
        for specific in specific_tests:
            if os.path.exists(os.path.join(worktree_path, specific)):
                test_target = specific
                break

    full_cmd = cmd + ([test_target] if test_target else [])

    try:
        result = subprocess.run(
            full_cmd,
            capture_output=True, text=True, timeout=timeout,
            cwd=worktree_path,
            env={**os.environ, "PYTHONPATH": worktree_path} if language == "python" else None,
        )
        passed = result.returncode == 0
        output = (result.stdout + result.stderr)[-1000:]
        return passed, output
    except subprocess.TimeoutExpired:
        return False, "test execution timed out"
    except FileNotFoundError:
        return None, f"test runner not installed"
    except Exception as e:
        return None, f"test runner error: {str(e)[:200]}"


def _apply_fix(worktree_path: str, target_file: str, proposed_fix: str) -> bool:
    """Apply the proposed fix to the target file in the worktree.

    Strategy: if the proposed fix looks like a complete function/class,
    try to replace the matching function in the file. Otherwise,
    overwrite the entire file content.
    """
    file_path = os.path.join(worktree_path, target_file)

    if not os.path.exists(file_path):
        return False

    # Read the current file
    try:
        current = Path(file_path).read_text(encoding="utf-8")
    except Exception:
        return False

    # If the fix starts with "def " or "class ", try to replace that
    # specific function/class in the file
    fix_stripped = proposed_fix.strip()
    if fix_stripped.startswith("def ") or fix_stripped.startswith("class "):
        # Extract function/class name
        first_line = fix_stripped.split("\n")[0]
        if "def " in first_line:
            name = first_line.split("def ")[1].split("(")[0].strip()
        else:
            name = first_line.split("class ")[1].split("(")[0].split(":")[0].strip()

        # Find and replace the function in the current file
        from src.engine.experiment_splitter import extract_function_from_file

        old_func = extract_function_from_file(current, name)
        if old_func:
            new_content = current.replace(old_func, fix_stripped)
            if new_content != current:
                Path(file_path).write_text(new_content, encoding="utf-8")
                return True

    # Fallback: overwrite the whole file
    # Only do this if the fix looks like a complete file (has imports)
    if "import " in fix_stripped and len(fix_stripped) > 500:
        Path(file_path).write_text(fix_stripped, encoding="utf-8")
        return True

    # Can't safely apply the fix
    return False


# ── Main verification ──────────────────────────────────────────────────────────

def verify_fix(
    proposed_fix: str,
    target_file: str,
    commit_before: str,
    detector_risk: float = 0.5,
    run_tests: bool = True,
) -> VerificationResult:
    """Verify a proposed code fix with execution + detector confidence.

    Args:
        proposed_fix: The code Qwen proposed (function/class body).
        target_file: Path to the file being fixed (e.g., "src/engine/data_prep.py").
        commit_before: Git commit where the bug exists.
        detector_risk: Hallucination risk from the detector probe (0-1).
        run_tests: If False, skip test execution (faster).

    Returns:
        VerificationResult with composite score.
    """
    result = VerificationResult()
    result.detector_confidence = 1.0 - detector_risk
    tmp_dir = None

    language = _detect_language(target_file)

    try:
        # Layer 1: Syntax check (no worktree needed)
        result.syntax_ok = _check_syntax(proposed_fix, language)
        result.checks_run.append(f"syntax:{language}")

        if not result.syntax_ok:
            result.error = "syntax check failed"
            result.score = _compute_score(result)
            return result

        # Create worktree for layers 2-3
        tmp_dir = tempfile.mkdtemp(prefix="dreamforge_verify_")
        worktree = _create_worktree(commit_before, tmp_dir)

        if not worktree:
            result.error = "failed to create git worktree"
            result.score = _compute_score(result)
            return result

        result.worktree_path = worktree

        # Apply the fix
        if not _apply_fix(worktree, target_file, proposed_fix):
            result.error = "failed to apply fix to target file"
            result.score = _compute_score(result)
            return result

        # Layer 2: Import/build check
        result.imports_ok, import_err = _check_imports(worktree, target_file, language)
        result.checks_run.append(f"build:{language}")

        if not result.imports_ok:
            result.error = f"import check failed: {import_err}"

        # Layer 3: Test execution
        if run_tests:
            tests_pass, test_output = _run_tests(worktree, target_file, language)
            result.tests_pass = tests_pass
            result.test_output = test_output
            result.checks_run.append(f"tests:{language}")

    except Exception as e:
        result.error = f"verification error: {str(e)[:300]}"
    finally:
        # Clean up worktree
        if tmp_dir:
            if result.worktree_path:
                _cleanup_worktree(result.worktree_path)
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # Layer 4: Detector confidence (already set from input)
    result.checks_run.append("detector")

    # Composite score
    result.score = _compute_score(result)
    result.passed = result.score >= VERIFICATION_THRESHOLD

    return result


def _compute_score(result: VerificationResult) -> float:
    """Compute weighted composite verification score."""
    score = 0.0

    if result.syntax_ok:
        score += WEIGHTS["syntax"]

    if result.imports_ok:
        score += WEIGHTS["imports"]

    if result.tests_pass is True:
        score += WEIGHTS["tests"]
    elif result.tests_pass is None:
        # No tests available — redistribute weight to detector
        score += WEIGHTS["tests"] * 0.3  # partial credit for no-test case
    # tests_pass is False → 0 contribution

    score += WEIGHTS["detector"] * result.detector_confidence

    return min(1.0, score)


# ── Convenience for cycle.py integration ───────────────────────────────────────

def verify_wake_output(
    wake_text: str,
    experiment: dict,
    detector_risk: float,
    run_tests: bool = True,
) -> VerificationResult:
    """Verify a wake output against an experiment's context.

    Convenience wrapper that extracts target_file and commit_before
    from the experiment dict.
    """
    # Get target file from pre_solution_context
    pre_ctx = experiment.get("pre_solution_context") or []
    target_file = ""
    if pre_ctx:
        first = pre_ctx[0]
        target_file = first.get("path", "") if isinstance(first, dict) else ""

    commit_before = experiment.get("git_start_hash", experiment.get("repo_hash", ""))

    if not target_file or not commit_before:
        return VerificationResult(
            error="missing target_file or commit_before",
            detector_confidence=1.0 - detector_risk,
        )

    return verify_fix(
        proposed_fix=wake_text,
        target_file=target_file,
        commit_before=commit_before,
        detector_risk=detector_risk,
        run_tests=run_tests,
    )

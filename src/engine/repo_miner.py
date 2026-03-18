"""Mine starred GitHub repos for code snippets (OSS-Instruct source).

Fetches the user's starred repos via gh CLI, clones them locally,
extracts meaningful functions/classes using AST parsing, and saves
them as candidates for synthetic instruction generation.

Supports: Python, TypeScript/JavaScript, Kotlin, Swift, Rust, Go, Java, Dart

Usage:
    uv run python -m src.engine.repo_miner --fetch          # clone/update repos
    uv run python -m src.engine.repo_miner --extract        # extract functions
    uv run python -m src.engine.repo_miner --fetch --extract  # both
    uv run python -m src.engine.repo_miner --stats          # show what we have
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

console = Console()

REPOS_DIR = Path("data/oss_repos")
SNIPPETS_DIR = Path("data/oss_snippets")
MIN_FUNCTION_LINES = 5
MAX_FUNCTION_LINES = 150
MAX_FILES_PER_REPO = 100


# ── Data types ─────────────────────────────────────────────────────────────────

@dataclass
class CodeSnippet:
    repo: str
    file_path: str
    language: str
    name: str           # function/class name
    code: str           # the actual code
    line_count: int
    has_docstring: bool
    context: str        # surrounding imports/class, for training context


# ── Fetch repos ────────────────────────────────────────────────────────────────

def fetch_starred_repos() -> list[dict]:
    """Get the user's starred repos from GitHub."""
    result = subprocess.run(
        ["gh", "api", "user/starred", "--paginate",
         "--jq", '.[] | {full_name, clone_url: .clone_url, language, description, stargazers_count, size}'],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        console.print(f"[red]Failed to fetch starred repos: {result.stderr}[/red]")
        return []

    repos = []
    for line in result.stdout.strip().split("\n"):
        if line.strip():
            try:
                repos.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return repos


def clone_or_update_repo(repo: dict, repos_dir: Path) -> Path | None:
    """Shallow clone a repo, or pull if already cloned."""
    name = repo["full_name"].replace("/", "__")
    repo_path = repos_dir / name

    if repo_path.exists():
        # Already cloned — pull latest
        result = subprocess.run(
            ["git", "pull", "--ff-only"],
            capture_output=True, text=True, timeout=60,
            cwd=str(repo_path),
        )
        if result.returncode == 0:
            return repo_path
        # Pull failed (diverged, etc.) — skip
        return repo_path

    # Shallow clone (saves disk space)
    clone_url = repo.get("clone_url", f"https://github.com/{repo['full_name']}.git")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", clone_url, str(repo_path)],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        console.print(f"  [red]Failed to clone {repo['full_name']}[/red]")
        return None
    return repo_path


def fetch_all(repos_dir: Path = REPOS_DIR) -> list[Path]:
    """Fetch/update all starred repos."""
    repos_dir.mkdir(parents=True, exist_ok=True)
    starred = fetch_starred_repos()
    console.print(f"[bold]Fetching {len(starred)} starred repos[/bold]")

    paths = []
    for repo in starred:
        lang = repo.get("language") or "?"
        size_mb = (repo.get("size") or 0) / 1024

        # Skip very large repos (>500MB) and repos with no language
        if size_mb > 500:
            console.print(f"  [yellow]SKIP {repo['full_name']} ({size_mb:.0f}MB — too large)[/yellow]")
            continue

        console.print(f"  {repo['full_name']:<40} {lang:<12} {size_mb:.0f}MB", style="dim")
        path = clone_or_update_repo(repo, repos_dir)
        if path:
            paths.append(path)

    console.print(f"\n  {len(paths)} repos ready in {repos_dir}")
    return paths


# ── Python extraction ──────────────────────────────────────────────────────────

def extract_python_functions(file_path: Path) -> list[CodeSnippet]:
    """Extract functions and classes from a Python file using AST."""
    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []

    snippets = []
    lines = source.split("\n")

    # Get imports for context
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            start = node.lineno - 1
            end = node.end_lineno or node.lineno
            imports.append("\n".join(lines[start:end]))
    import_context = "\n".join(imports[:10])  # max 10 import lines

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno - 1
            end = node.end_lineno or (start + 1)
            code = "\n".join(lines[start:end])
            line_count = end - start

            if line_count < MIN_FUNCTION_LINES or line_count > MAX_FUNCTION_LINES:
                continue

            # Check for docstring
            has_docstring = (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, (ast.Constant, ast.Str))
            )

            snippets.append(CodeSnippet(
                repo="",  # filled in by caller
                file_path=str(file_path),
                language="python",
                name=node.name,
                code=code,
                line_count=line_count,
                has_docstring=has_docstring,
                context=import_context,
            ))

    return snippets


# ── Generic extraction (non-Python) ───────────────────────────────────────────

def extract_generic_functions(file_path: Path, language: str) -> list[CodeSnippet]:
    """Extract functions/methods from non-Python files using regex patterns.

    Less precise than AST but works for any language with function-like syntax.
    """
    import re

    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []

    lines = source.split("\n")
    snippets = []

    # Language-specific function patterns
    patterns = {
        "kotlin": r"^\s*(fun\s+\w+|class\s+\w+|object\s+\w+)",
        "java": r"^\s*(public|private|protected|static|\s)*(void|int|String|boolean|[A-Z]\w*)\s+\w+\s*\(",
        "typescript": r"^\s*(export\s+)?(async\s+)?function\s+\w+|^\s*(export\s+)?(const|let)\s+\w+\s*=\s*(async\s+)?\(",
        "javascript": r"^\s*(export\s+)?(async\s+)?function\s+\w+|^\s*(const|let|var)\s+\w+\s*=\s*(async\s+)?\(",
        "swift": r"^\s*(func\s+\w+|class\s+\w+|struct\s+\w+)",
        "rust": r"^\s*(pub\s+)?(fn\s+\w+|struct\s+\w+|impl\s+\w+)",
        "go": r"^\s*func\s+(\(\w+\s+\*?\w+\)\s+)?\w+",
        "dart": r"^\s*(class\s+\w+|\w+\s+\w+\s*\()",
    }

    pattern = patterns.get(language.lower())
    if not pattern:
        return []

    # Find function starts
    func_starts = []
    for i, line in enumerate(lines):
        if re.match(pattern, line):
            func_starts.append(i)

    # Extract each function (up to next function or max lines)
    for idx, start in enumerate(func_starts):
        end = func_starts[idx + 1] if idx + 1 < len(func_starts) else min(start + MAX_FUNCTION_LINES, len(lines))
        end = min(end, start + MAX_FUNCTION_LINES)

        # Try to find the actual end by brace matching
        if language.lower() in ("kotlin", "java", "typescript", "javascript", "swift", "rust", "go", "dart"):
            brace_count = 0
            found_open = False
            for i in range(start, min(start + MAX_FUNCTION_LINES, len(lines))):
                brace_count += lines[i].count("{") - lines[i].count("}")
                if "{" in lines[i]:
                    found_open = True
                if found_open and brace_count <= 0:
                    end = i + 1
                    break

        code = "\n".join(lines[start:end])
        line_count = end - start

        if line_count < MIN_FUNCTION_LINES:
            continue

        # Extract name from first line
        name_match = re.search(r'\b(fun|func|function|class|struct|impl|object)\s+(\w+)', lines[start])
        name = name_match.group(2) if name_match else f"unnamed_{start}"

        snippets.append(CodeSnippet(
            repo="",
            file_path=str(file_path),
            language=language.lower(),
            name=name,
            code=code,
            line_count=line_count,
            has_docstring="/**" in code[:200] or '"""' in code[:200] or "///" in code[:200],
            context="",
        ))

    return snippets


# ── File discovery ─────────────────────────────────────────────────────────────

LANG_EXTENSIONS = {
    "python": [".py"],
    "kotlin": [".kt", ".kts"],
    "typescript": [".ts", ".tsx"],
    "javascript": [".js", ".jsx", ".mjs"],
    "swift": [".swift"],
    "rust": [".rs"],
    "go": [".go"],
    "java": [".java"],
    "dart": [".dart"],
}

SKIP_DIRS = {
    "node_modules", ".git", "__pycache__", "build", "dist", ".gradle",
    "target", ".idea", ".vscode", "venv", ".venv", "Pods",
}


def find_source_files(repo_path: Path) -> list[tuple[Path, str]]:
    """Find source files in a repo with their language."""
    files = []
    for lang, exts in LANG_EXTENSIONS.items():
        for ext in exts:
            for f in repo_path.rglob(f"*{ext}"):
                if any(skip in f.parts for skip in SKIP_DIRS):
                    continue
                # Skip test files, generated files, configs
                name = f.name.lower()
                if name.startswith("test_") or name.endswith("_test.py"):
                    continue
                if "generated" in name or "migration" in name:
                    continue
                files.append((f, lang))
                if len(files) >= MAX_FILES_PER_REPO:
                    return files
    return files


# ── Main extraction ────────────────────────────────────────────────────────────

def extract_all(repos_dir: Path = REPOS_DIR, output_dir: Path = SNIPPETS_DIR) -> int:
    """Extract code snippets from all cloned repos."""
    output_dir.mkdir(parents=True, exist_ok=True)

    total_snippets = 0
    repo_dirs = [d for d in repos_dir.iterdir() if d.is_dir() and (d / ".git").exists()]

    console.print(f"[bold]Extracting from {len(repo_dirs)} repos[/bold]")

    for repo_dir in sorted(repo_dirs):
        repo_name = repo_dir.name.replace("__", "/")
        files = find_source_files(repo_dir)

        if not files:
            continue

        snippets = []
        for file_path, language in files:
            if language == "python":
                extracted = extract_python_functions(file_path)
            else:
                extracted = extract_generic_functions(file_path, language)

            for s in extracted:
                s.repo = repo_name
                # Make file_path relative to repo
                try:
                    s.file_path = str(file_path.relative_to(repo_dir))
                except ValueError:
                    pass

            snippets.extend(extracted)

        if snippets:
            # Save as JSONL
            output_file = output_dir / f"{repo_dir.name}.jsonl"
            with open(output_file, "w", encoding="utf-8") as f:
                for s in snippets:
                    record = {
                        "repo": s.repo,
                        "file_path": s.file_path,
                        "language": s.language,
                        "name": s.name,
                        "code": s.code,
                        "line_count": s.line_count,
                        "has_docstring": s.has_docstring,
                        "context": s.context,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            total_snippets += len(snippets)
            console.print(
                f"  {repo_name:<40} {len(files):>3} files → {len(snippets):>4} snippets",
                style="dim")

    console.print(f"\n[bold green]Total: {total_snippets} snippets from {len(repo_dirs)} repos[/bold green]")
    return total_snippets


def show_stats(output_dir: Path = SNIPPETS_DIR):
    """Show extraction statistics."""
    if not output_dir.exists():
        console.print("[yellow]No snippets extracted yet. Run --extract first.[/yellow]")
        return

    total = 0
    by_lang = {}
    by_repo = {}

    for f in output_dir.glob("*.jsonl"):
        count = 0
        for line in open(f, encoding="utf-8"):
            record = json.loads(line)
            count += 1
            lang = record["language"]
            by_lang[lang] = by_lang.get(lang, 0) + 1
        repo_name = f.stem.replace("__", "/")
        by_repo[repo_name] = count
        total += count

    console.print(f"[bold]Snippet Statistics[/bold]")
    console.print(f"  Total: {total} snippets")
    console.print(f"\n  By language:")
    for lang, count in sorted(by_lang.items(), key=lambda x: -x[1]):
        console.print(f"    {lang:<15} {count:>5}")
    console.print(f"\n  Top repos:")
    for repo, count in sorted(by_repo.items(), key=lambda x: -x[1])[:15]:
        console.print(f"    {repo:<40} {count:>5}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Mine starred GitHub repos for code snippets")
    parser.add_argument("--fetch", action="store_true",
                        help="Clone/update starred repos")
    parser.add_argument("--extract", action="store_true",
                        help="Extract code snippets from cloned repos")
    parser.add_argument("--stats", action="store_true",
                        help="Show extraction statistics")
    args = parser.parse_args()

    if args.fetch:
        fetch_all()
    if args.extract:
        extract_all()
    if args.stats:
        show_stats()
    if not (args.fetch or args.extract or args.stats):
        parser.print_help()


if __name__ == "__main__":
    main()

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "paper" / "main.tex"

ENV_RE = re.compile(r"\\(begin|end)\{([A-Za-z*]+)\}")
INPUT_RE = re.compile(r"\\input\{([^}]+)\}")
GRAPHICS_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
LABEL_RE = re.compile(r"\\label\{([^}]+)\}")
REF_RE = re.compile(r"\\(?:ref|eqref)\{([^}]+)\}")
CITE_RE = re.compile(r"\\cite\{([^}]+)\}")
BIBKEY_RE = re.compile(r"^@\w+\{([^,]+),", re.MULTILINE)


def strip_comments(text: str) -> str:
    out = []
    for line in text.splitlines():
        cleaned, escaped = [], False
        for ch in line:
            if ch == "\\" and not escaped:
                escaped = True
                cleaned.append(ch)
                continue
            if ch == "%" and not escaped:
                break
            escaped = False
            cleaned.append(ch)
        out.append("".join(cleaned))
    return "\n".join(out)


def expand(path: Path, seen: set[Path]) -> str:
    if path in seen:
        return ""
    seen.add(path)
    text = strip_comments(path.read_text(encoding="utf-8"))
    parts, last = [], 0
    for match in INPUT_RE.finditer(text):
        parts.append(text[last : match.start()])
        target = match.group(1)
        child = ROOT / (target if target.endswith(".tex") else target + ".tex")
        if child.exists():
            parts.append(expand(child, seen))
        last = match.end()
    parts.append(text[last:])
    return "".join(parts)


def main() -> int:
    problems: list[str] = []

    if not MAIN.exists():
        print(f"FAIL: {MAIN} not found")
        return 1

    seen: set[Path] = set()
    flat = expand(MAIN, seen)

    stack, env_errors = [], []
    for kind, name in ENV_RE.findall(flat):
        if kind == "begin":
            stack.append(name)
        else:
            if not stack:
                env_errors.append(f"\\end{{{name}}} with no matching \\begin")
            elif stack[-1] != name:
                env_errors.append(f"\\end{{{name}}} closes \\begin{{{stack[-1]}}}")
            else:
                stack.pop()
    for name in stack:
        env_errors.append(f"unclosed \\begin{{{name}}}")
    problems += env_errors

    for source in sorted(seen):
        body = strip_comments(source.read_text(encoding="utf-8"))
        for target in INPUT_RE.findall(body):
            child = ROOT / (target if target.endswith(".tex") else target + ".tex")
            if not child.exists():
                problems.append(f"missing \\input target: {target} (from {source.name})")

    graphics_dirs = [ROOT / "paper_assets" / "figures"]
    for name in GRAPHICS_RE.findall(flat):
        if not any((d / name).exists() for d in graphics_dirs):
            problems.append(f"missing figure: {name}")

    labels = set(LABEL_RE.findall(flat))
    for ref in sorted(set(REF_RE.findall(flat))):
        if ref not in labels:
            problems.append(f"undefined \\ref/\\eqref target: {ref}")

    bib = ROOT / "paper" / "references.bib"
    bibkeys = set(BIBKEY_RE.findall(bib.read_text(encoding="utf-8"))) if bib.exists() else set()
    cited: set[str] = set()
    for group in CITE_RE.findall(flat):
        cited.update(k.strip() for k in group.split(","))
    for key in sorted(cited):
        if key not in bibkeys:
            problems.append(f"citation with no bib entry: {key}")

    braces = flat.count("{") - flat.count("}")
    if braces != 0:
        problems.append(f"unbalanced braces in expanded document: {braces:+d}")

    if flat.count("$$"):
        problems.append("uses $$ display math; prefer \\[ \\] or equation")

    print(f"files expanded      : {len(seen)}")
    print(f"environments checked: {len(ENV_RE.findall(flat))}")
    print(f"labels / refs       : {len(labels)} / {len(set(REF_RE.findall(flat)))}")
    print(f"citations / bibkeys : {len(cited)} / {len(bibkeys)}")
    print(f"figures referenced  : {len(set(GRAPHICS_RE.findall(flat)))}")
    unused = sorted(bibkeys - cited)
    if unused:
        print(f"uncited bib entries : {', '.join(unused)}")

    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("\nStructure OK: environments balanced, all inputs/figures/refs/citations resolve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

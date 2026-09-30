"""Evaluation — measure how well retrieval and generation actually work.

Two kinds of test cases:
  - Normal: question + expected keywords. Answer must contain them all.
  - Refusal: question the docs can't answer. Answer must refuse.

Scoring is keyword-based (fast, free, deterministic). An LLM-judge mode
comes later — this gives you a baseline you can improve against.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from rich.console import Console

from amd.ask import ask

REFUSAL_PHRASE = "i don't have information"

console = Console()


@dataclass
class CaseResult:
    question: str
    answer: str
    passed: bool
    reason: str
    sources: list[dict]


@dataclass
class EvalSummary:
    total: int
    passed: int
    failed: int
    results: list[CaseResult]

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


def load_cases(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"test file not found: {p}")
    data = json.loads(p.read_text())
    cases = data.get("cases", [])
    if not cases:
        raise ValueError("test file has no 'cases'")
    return cases


def _check(case: dict, answer: str) -> tuple[bool, str]:
    """Return (passed, reason)."""
    low = answer.lower()

    if case.get("should_refuse"):
        if REFUSAL_PHRASE in low:
            return True, "refused as expected"
        return False, "should have refused, but answered"

    # normal case: every expected keyword must appear
    expected = [k.lower() for k in case.get("expected_contains", [])]
    missing = [k for k in expected if k not in low]
    if missing:
        return False, f"missing keywords: {', '.join(missing)}"
    return True, "contains all expected keywords"


def run_eval(
    cases: Iterable[dict],
    db_path: str | Path = "amd.db",
    verbose: bool = True,
) -> EvalSummary:
    cases = list(cases)
    results: list[CaseResult] = []

    for i, case in enumerate(cases, 1):
        q = case["question"]
        if verbose:
            console.print(f"[dim]({i}/{len(cases)})[/dim] {q}")

        try:
            result = ask(q, db_path=db_path)
            passed, reason = _check(case, result.text)
            answer = result.text
            sources = result.sources
        except Exception as e:
            passed, reason = False, f"error: {e}"
            answer, sources = "", []

        results.append(
            CaseResult(
                question=q,
                answer=answer,
                passed=passed,
                reason=reason,
                sources=sources,
            )
        )

        if verbose:
            marker = "[green]✓[/green]" if passed else "[red]✗[/red]"
            console.print(f"  {marker} {reason}")

    passed = sum(1 for r in results if r.passed)
    return EvalSummary(
        total=len(results),
        passed=passed,
        failed=len(results) - passed,
        results=results,
    )


def print_summary(summary: EvalSummary) -> None:
    console.print()
    console.print("[bold]Results[/bold]")
    console.print(f"  total:  {summary.total}")
    console.print(f"  passed: [green]{summary.passed}[/green]")
    console.print(f"  failed: [red]{summary.failed}[/red]")
    console.print(f"  rate:   [bold]{summary.pass_rate:.0%}[/bold]")

    failures = [r for r in summary.results if not r.passed]
    if failures:
        console.print()
        console.print("[bold red]Failures[/bold red]")
        for r in failures:
            console.print(f"  [red]✗[/red] {r.question}")
            console.print(f"    [dim]{r.reason}[/dim]")
            if r.answer:
                snippet = r.answer[:150].replace("\n", " ")
                console.print(f"    [dim]answer: {snippet}…[/dim]")

#!/usr/bin/env python3
"""Quality gate: tests, coverage, benchmarks, ruff, bandit.

Prints a human-readable table and a JSON report. Exits 0 when all gates
pass, 1 otherwise.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(cmd: list[str], cwd: Path | None = None) -> tuple[int, str, str]:
    """Run a command, return (exit_code, stdout, stderr)."""
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _tool_available(name: str) -> bool:
    return shutil.which(name) is not None


# ---------------------------------------------------------------------------
# Gate: tests + coverage
# ---------------------------------------------------------------------------


def gate_tests_and_coverage(workdir: Path) -> dict:
    """Run pytest with coverage into a temp dir, parse the JSON report."""
    cov_json = workdir / "coverage.json"
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/",
        "--cov=src/precision_health_os",
        f"--cov-report=json:{cov_json}",
        "-q",
    ]
    code, out, err = _run(cmd)

    result: dict = {
        "gate": "tests",
        "passed": False,
        "test_count": 0,
        "passed_count": 0,
        "failed_count": 0,
        "coverage_total": 0.0,
        "coverage_per_module": {},
        "raw_exit": code,
    }

    if code != 0:
        result["error"] = (out + err)[-2000:]
        return result

    # Parse coverage JSON
    if cov_json.exists():
        cov_data = json.loads(cov_json.read_text())
        totals = cov_data.get("totals", {})
        result["coverage_total"] = round(totals.get("percent_covered", 0.0), 2)

        per_module: dict[str, float] = {}
        for fname, fdata in cov_data.get("files", {}).items():
            # fname is like src/precision_health_os/module.py
            parts = Path(fname).parts
            if len(parts) >= 3 and parts[-2] == "precision_health_os":
                mod = parts[-1].replace(".py", "")
                pct = fdata.get("summary", {}).get("percent_covered", 0.0)
                per_module[mod] = round(pct, 2)
        result["coverage_per_module"] = per_module

    # Parse test counts from pytest output
    for line in out.splitlines():
        if "passed" in line and "==" in line:
            # e.g. "42 passed in 1.23s"
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "passed" and i > 0:
                    with contextlib.suppress(ValueError):
                        result["passed_count"] = int(parts[i - 1])
            result["test_count"] = result["passed_count"]
            break

    result["failed_count"] = result["test_count"] - result["passed_count"]
    result["passed"] = code == 0 and result["failed_count"] == 0
    return result


# ---------------------------------------------------------------------------
# Gate: benchmarks
# ---------------------------------------------------------------------------


def gate_benchmarks() -> dict:
    """Run the benchmark suite in-process via build_harness()."""
    result: dict = {
        "gate": "benchmarks",
        "passed": False,
        "total": 0,
        "passed_count": 0,
        "failed_count": 0,
        "pass_rate": 0.0,
        "mean_score": 0.0,
        "details": [],
    }
    try:
        from precision_health_os.benchmarks import build_harness

        harness = build_harness()
        summary = harness.summary()
        result["total"] = summary["total"]
        result["passed_count"] = summary["passed"]
        result["failed_count"] = summary["failed"]
        result["pass_rate"] = round(summary["pass_rate"], 4)
        result["mean_score"] = round(summary["mean_score"], 4)
        result["details"] = summary["results"]
        result["passed"] = summary["failed"] == 0
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


# ---------------------------------------------------------------------------
# Gate: ruff
# ---------------------------------------------------------------------------


def gate_ruff(workdir: Path) -> dict:
    """Run ruff check; SKIPPED if not installed."""
    if not _tool_available("ruff"):
        return {"gate": "ruff", "passed": True, "status": "SKIPPED", "detail": "ruff not installed"}

    code, out, err = _run(["ruff", "check", "src/", "tests/"])
    return {
        "gate": "ruff",
        "passed": code == 0,
        "status": "PASS" if code == 0 else "FAIL",
        "detail": (out + err)[-1000:],
    }


# ---------------------------------------------------------------------------
# Gate: bandit
# ---------------------------------------------------------------------------


def gate_bandit(workdir: Path) -> dict:
    """Run bandit; SKIPPED if not installed."""
    if not _tool_available("bandit"):
        return {
            "gate": "bandit",
            "passed": True,
            "status": "SKIPPED",
            "detail": "bandit not installed",
        }

    code, out, err = _run(
        [
            "bandit",
            "-r",
            "src/precision_health_os",
            "-c",
            "pyproject.toml",
            "-f",
            "json",
            "-o",
            str(workdir / "bandit.json"),
        ]
    )
    # bandit exits 0 when no issues, 1 when issues found
    return {
        "gate": "bandit",
        "passed": code == 0,
        "status": "PASS" if code == 0 else "FAIL",
        "detail": (out + err)[-1000:],
    }


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def render_table(report: dict) -> str:
    """Render the quality report as a human-readable table."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("PRECISION HEALTH OS — QUALITY REPORT")
    lines.append("=" * 72)

    # Tests
    t = report["tests"]
    status = "PASS" if t["passed"] else "FAIL"
    lines.append(f"\n[{status}] Tests")
    lines.append(f"  total={t['test_count']} passed={t['passed_count']} failed={t['failed_count']}")
    if t.get("error"):
        lines.append(f"  error: {t['error'][:200]}")

    # Coverage
    lines.append(f"\n  Coverage: {t['coverage_total']:.1f}% total")
    if t["coverage_per_module"]:
        for mod, pct in sorted(t["coverage_per_module"].items()):
            lines.append(f"    {mod:30s} {pct:6.1f}%")

    # Benchmarks
    b = report["benchmarks"]
    status = "PASS" if b["passed"] else "FAIL"
    lines.append(f"\n[{status}] Benchmarks")
    lines.append(
        f"  total={b['total']} passed={b['passed_count']} "
        f"failed={b['failed_count']} pass_rate={b['pass_rate']:.2%} "
        f"mean_score={b['mean_score']:.3f}"
    )
    for d in b.get("details", []):
        mark = "PASS" if d["passed"] else "FAIL"
        lines.append(f"    [{mark}] {d['benchmark_name']}  (score {d['score']:.2f})")
        if d.get("error"):
            lines.append(f"          error: {d['error']}")

    # Ruff
    r = report["ruff"]
    lines.append(f"\n[{r['status']}] Ruff")
    if r.get("detail") and r["status"] != "SKIPPED":
        lines.append(f"  {r['detail'][:200]}")

    # Bandit
    bd = report["bandit"]
    lines.append(f"\n[{bd['status']}] Bandit")
    if bd.get("detail") and bd["status"] != "SKIPPED":
        lines.append(f"  {bd['detail'][:200]}")

    lines.append("\n" + "=" * 72)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """Run all quality gates and report results."""
    with tempfile.TemporaryDirectory(prefix="quality_report_") as tmp:
        workdir = Path(tmp)

        tests = gate_tests_and_coverage(workdir)
        benchmarks = gate_benchmarks()
        ruff = gate_ruff(workdir)
        bandit = gate_bandit(workdir)

    report = {
        "tests": tests,
        "benchmarks": benchmarks,
        "ruff": ruff,
        "bandit": bandit,
    }

    # Print human-readable table
    print(render_table(report))

    # Print JSON
    print("\n--- JSON ---")
    print(json.dumps(report, indent=2, default=str))

    # Determine overall pass/fail
    failing: list[str] = []
    for gate_name, gate_data in report.items():
        if not gate_data.get("passed", False):
            failing.append(gate_name)

    if failing:
        print(f"\nQUALITY: FAIL  (failing gates: {', '.join(failing)})")
        return 1
    else:
        print("\nQUALITY: PASS")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

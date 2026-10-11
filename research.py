"""research.py - The main script.   Guide: GUIDE.md, part 3.

Usage:  python research.py "survey about world model"
Result: reports/<slug>.md   reports/<slug>.sources.json   reports/<slug>.meta.json
"""
import json
import os
import re
import sys
import threading
import time
from collections import Counter
from pathlib import Path

from langchain_core.rate_limiters import InMemoryRateLimiter

from agents import FINALIZER_PATH, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, build_lead_agent
from model import make_model
from sandbox import download, open_sandbox, upload

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"   # provided: uploaded next to your validator
MAX_MINUTES = float(os.getenv("LAB_MAX_MINUTES") or 60)  # wall-clock cap of one run (a dead connection must not hang it)
HEARTBEAT_S = 300


def slugify(topic):
    """Turn a topic into a safe file name: lower case, runs of non-word characters become one "-", max 60 chars,
    never empty (fall back to "topic"). The topic is user input: "../../x" must not escape reports/."""
    slug = re.sub(r"[^a-z0-9]+", "-", (topic or "").lower()).strip("-")
    return slug[:60].strip("-") or "topic"


def build_prompt(topic):
    """The user message sent to the lead agent."""
    return (f"Research topic: {topic}\n\n"
            f"Produce the cited survey report at {REPORT_PATH} (sources in {SOURCES_PATH}), following every step of "
            "your instructions: plan, delegate at least 3 sub-questions to researchers in parallel, check their notes, "
            "merge sources (at least 3 source families), write the body, run the finalizer, run the validator until "
            "it prints OK, and spot-check claims with the citation-checker.")


def summarize(messages, elapsed, model_name):
    """Return {"model", "elapsed_s", "subagent_calls", "tool_calls": {name: count}, "tokens": {"input", "output"}}.
    Lead messages only: subagent tokens are not included, so this undercounts the real cost."""
    calls = Counter()
    tokens = {"input": 0, "output": 0}
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            calls[call["name"]] += 1
        usage = getattr(message, "usage_metadata", None) or {}
        tokens["input"] += usage.get("input_tokens", 0) or 0
        tokens["output"] += usage.get("output_tokens", 0) or 0
    return {"model": model_name, "elapsed_s": round(elapsed, 1), "subagent_calls": calls["task"],
            "tool_calls": dict(calls), "tokens": tokens}


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    """Download the report from the sandbox and write the three files into reports_dir. Return the report path.
    A missing/empty report or a missing/invalid sources.json raises RuntimeError and writes nothing."""
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report, raw_sources = files.get(REPORT_PATH), files.get(SOURCES_PATH)
    if not report or not report.strip():
        raise RuntimeError(f"the agent produced no report ({REPORT_PATH} missing or empty)")
    if not raw_sources:
        raise RuntimeError(f"the agent produced no {SOURCES_PATH}")
    try:
        sources = json.loads(raw_sources)
    except ValueError as exc:
        raise RuntimeError(f"{SOURCES_PATH} is not valid JSON: {exc}") from exc
    if not isinstance(sources, list) or not sources:
        raise RuntimeError(f"{SOURCES_PATH} must be a non-empty JSON list")

    families = sorted({str(s.get("source")) for s in sources if isinstance(s, dict) and s.get("source")})
    meta = {"topic": topic, **summarize(messages, elapsed, model_name), "n_sources": len(sources),
            "source_families": families}
    reports_dir.mkdir(parents=True, exist_ok=True)
    slug = slugify(topic)
    # sources.json is written byte for byte as downloaded: the submitted files must equal the sandbox output
    (reports_dir / f"{slug}.sources.json").write_bytes(raw_sources)
    (reports_dir / f"{slug}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = reports_dir / f"{slug}.md"
    report_path.write_bytes(report)
    return report_path


def _model_name(model):
    return getattr(model, "model_name", None) or getattr(model, "model", None) or type(model).__name__


def limit_rate(model):
    """Optional LAB_RPM (model calls per minute): one limiter on the shared model object paces the lead AND every
    subagent, which all reuse it. Needed on free tiers with a few requests per minute."""
    rpm = float(os.getenv("LAB_RPM") or 0)
    if rpm > 0:
        model.rate_limiter = InMemoryRateLimiter(requests_per_second=rpm / 60, check_every_n_seconds=0.5,
                                                 max_bucket_size=1)
    return model


def run_with_deadline(fn, seconds):
    """Run fn() in a worker thread and return its result; raise TimeoutError when it is still running after `seconds`.
    The Daytona client reads without a timeout, so a dropped connection would otherwise block the run forever."""
    outcome = {}

    def target():
        try:
            outcome["result"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised in the caller's thread
            outcome["error"] = exc

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    deadline = time.monotonic() + seconds
    while worker.is_alive() and time.monotonic() < deadline:
        worker.join(min(HEARTBEAT_S, max(deadline - time.monotonic(), 0)))
        if worker.is_alive():
            print(f"[research] still running... {(seconds - (deadline - time.monotonic())) / 60:.0f} min",
                  file=sys.stderr, flush=True)
    if worker.is_alive():
        raise TimeoutError(f"the agent did not finish within {seconds / 60:.0f} minutes")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["result"]


def main(topic):
    """Return the process exit code (0 ok, 1 failed run, 2 no topic)."""
    topic = (topic or "").strip()
    if not topic:
        print('usage: python research.py "<topic>"', file=sys.stderr)
        return 2
    model = limit_rate(make_model())
    start = time.monotonic()
    with open_sandbox() as backend:  # the sandbox is always cleaned up, even on errors
        backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
        upload(backend, {VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(), FINALIZER_PATH: FINALIZER_SOURCE.read_bytes()})
        agent = build_lead_agent(backend, model)
        try:
            result = run_with_deadline(
                lambda: agent.invoke({"messages": [{"role": "user", "content": build_prompt(topic)}]},
                                     config={"recursion_limit": 1000}),
                MAX_MINUTES * 60)
            report_path = save_outputs(backend, topic, result["messages"], time.monotonic() - start,
                                       _model_name(model))
        except (RuntimeError, TimeoutError) as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1
    print(f"report saved to {report_path}")
    return 0


if __name__ == "__main__":
    exit_code = main(" ".join(sys.argv[1:]))
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)  # do not wait for agent threads left hanging after a timeout (the sandbox is already removed)

"""Test suite for the multi-agent review pipeline. No external services needed —
model calls are stubbed at the orchestrator boundary.

Run:  python tests/run_tests.py
Exits non-zero on any failure.
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app.pipeline as pipeline
from app.pipeline import (
    anchor_changes,
    build_stats,
    compute_scores,
    dedupe_findings,
    drop_overlaps,
    run_pipeline,
    split_spans,
)

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = ""):
    if cond:
        print(f"PASS {name}")
    else:
        FAILURES.append(name)
        print(f"FAIL {name} — {detail}")


# ---------------- unit: chunking ----------------

def test_split_spans():
    text = "para one. " * 100 + "\n\n" + "para two! " * 800 + "\n\n" + "tail."
    spans = split_spans(text)
    check("split covers whole text", spans[0][0] == 0 and spans[-1][1] == len(text))
    check("split has no gaps", all(e1 == s2 for (_, e1), (s2, _) in zip(spans, spans[1:])))
    check("split reconstructs text", "".join(text[s:e] for s, e in spans) == text)
    check("split short text is single span", split_spans("hello") == [(0, 5)])
    check("split empty-ish", split_spans("x") == [(0, 1)])
    # pathological: no spaces at all
    blob = "a" * (pipeline.CHUNK_TARGET * 2 + 10)
    bspans = split_spans(blob)
    check("split handles spaceless blob", "".join(blob[s:e] for s, e in bspans) == blob)


# ---------------- unit: anchoring ----------------

def test_anchoring():
    chunk = "He dont know. She dont care. He dont know."
    changes = [
        {"original": "dont", "corrected": "doesn't", "category": "grammar", "reason": "", "severity": "minor"}
        for _ in range(3)
    ] + [
        {"original": "absent", "corrected": "x", "category": "weird", "reason": "", "severity": "huge"},
        {"original": "same", "corrected": "same", "category": "grammar", "reason": "", "severity": "minor"},
    ]
    out = anchor_changes(chunk, changes, offset=10)
    check("anchor drops no-op change", len(out) == 4)
    check("anchor finds successive occurrences", [c["start"] for c in out[:3]] == [13, 28, 42])
    check("anchor unfound gets null", out[3]["start"] is None)
    check("anchor normalises category", out[3]["category"] == "clarity")
    check("anchor normalises severity", out[3]["severity"] == "minor")

    overlapping = [
        {"original": "ab", "corrected": "x", "category": "grammar", "reason": "", "severity": "minor", "start": 0, "end": 5},
        {"original": "b", "corrected": "y", "category": "grammar", "reason": "", "severity": "minor", "start": 3, "end": 5},
        {"original": "c", "corrected": "z", "category": "grammar", "reason": "", "severity": "minor", "start": 7, "end": 8},
    ]
    kept = drop_overlaps(overlapping)
    check("overlap keeps non-overlapping", [c["start"] for c in kept if c["start"] is not None] == [0, 7])
    check("overlap demotes overlapped to unanchored", sum(1 for c in kept if c["start"] is None) == 1)


# ---------------- unit: dedupe + scores ----------------

def test_dedupe():
    def f(t, a):
        return {"title": t, "agent": a, "severity": "minor", "verified": True}
    out = dedupe_findings([f("Role mismatch", "terminology"), f("role  MISMATCH!", "procedure"), f("Other", "iso")])
    check("dedupe collapses same title across agents", len(out) == 2)
    check("dedupe keeps first occurrence", out[0]["agent"] == "terminology")


def test_scores():
    clean = compute_scores([], [], 500)
    check("clean doc scores 100 everywhere", all(v == 100 for v in clean.values()), str(clean))

    changes = [
        {"category": "grammar", "severity": "major", "verified": True},
        {"category": "spelling", "severity": "minor", "verified": True},
        {"category": "grammar", "severity": "minor", "verified": False},  # rejected: no penalty
    ]
    findings = [
        {"agent": "procedure", "severity": "major", "verified": True},
        {"agent": "logic", "severity": "major", "verified": False},  # rejected: no penalty
    ]
    s = compute_scores(changes, findings, 500)
    check("grammar penalised", s["grammar"] < 100)
    check("procedure penalised", s["procedure"] < 100)
    check("rejected finding not penalised", s["logic"] == 100)
    check("scores bounded", all(0 <= v <= 100 for v in s.values()))
    check("overall present", 0 <= s["overall"] <= 100)

    # a disaster document must floor at 0, not go negative
    many = [{"category": "grammar", "severity": "major", "verified": True}] * 200
    s2 = compute_scores(many, [], 100)
    check("score floors at zero", s2["grammar"] == 0)


def test_stats():
    st = build_stats("One two three. Four five!", [], 1)
    check("stats words", st["words"] == 5)
    check("stats sentences", st["sentences"] == 2)


# ---------------- integration: full pipeline with stubbed agents ----------------

async def fake_chunk(client, text, attempts=3):
    changes = []
    if "teh " in text:
        changes.append({"original": "teh ", "corrected": "the ", "category": "spelling",
                        "reason": "Misspelling.", "severity": "minor"})
    if "dont" in text:
        changes.append({"original": "dont", "corrected": "don't", "category": "punctuation",
                        "reason": "Missing apostrophe.", "severity": "minor"})
    return {"corrected_text": text.replace("teh ", "the "), "changes": changes, "summary": "ok"}


async def fake_doc_agent(client, agent, text):
    if agent["key"] == "logic":
        return {"findings": [{"title": "Missing No branch", "detail": "Approved? has no No path.",
                              "location": "Procedure", "severity": "major"}], "verdict": "One gap."}
    if agent["key"] == "terminology":
        return {"findings": [{"title": "Missing No branch", "detail": "dup title, must dedupe",
                              "severity": "minor"},
                             {"title": "Dept variants", "detail": "Two names for one dept.",
                              "severity": "minor"}], "verdict": "Mostly fine."}
    if agent["key"] == "iso":
        return {"findings": [], "present_sections": ["Purpose"], "missing_sections": ["Records"],
                "is_sop": True, "verdict": "Partial."}
    return {"findings": [], "verdict": "Clean."}


async def fake_verifier(client, text, items):
    # reject the first change and the "Dept variants" finding
    out = {}
    for it in items:
        keep = True
        if it["id"] == "c0":
            keep = False
        if "Dept variants" in it["text"]:
            keep = False
        out[it["id"]] = {"keep": keep, "confidence": 0.9 if keep else 0.2,
                         "note": "" if keep else "not convincing"}
    return out


async def fake_summary(client, stats, scores, top):
    return {"summary": "Decent document.", "risk_level": "medium",
            "top_issues": ["Missing No branch"], "readability": 80}


def test_pipeline_integration():
    pipeline.proofread_chunk = fake_chunk
    pipeline.run_doc_agent = fake_doc_agent
    pipeline.run_verifier = fake_verifier
    pipeline.run_summary = fake_summary

    text = "teh start. " + "filler words here. " * 400 + "\n\nAnd he dont stop."
    progress_events = []
    agent_events = []

    async def progress(done, total):
        progress_events.append((done, total))

    async def agent_status(key, label, state):
        agent_events.append((key, state))

    result = asyncio.run(run_pipeline(text, progress, agent_status))
    st = result["stats"]
    report = st["report"]
    n_chunks = st["chunks"]
    expected_total = n_chunks + len(pipeline.DOC_AGENTS) + 2

    check("progress starts at 0/total", progress_events[0] == (0, expected_total))
    check("progress ends at total", progress_events[-1][0] == expected_total)
    check("all doc agents reported done",
          all(any(k == a["key"] and s == "done" for k, s in agent_events) for a in pipeline.DOC_AGENTS),
          str(agent_events))
    check("verifier + summary statuses present",
          any(k == "verifier" and s == "done" for k, s in agent_events)
          and any(k == "summary" and s == "done" for k, s in agent_events))

    check("changes anchored globally",
          all(text[c["start"]:c["end"]] == c["original"] for c in result["changes"] if c["start"] is not None))
    rejected = [c for c in result["changes"] if c.get("verified") is False]
    check("verifier rejection applied to change", len(rejected) == 1 and rejected[0]["confidence"] == 0.2)

    check("findings deduped across agents",
          sum(1 for f in report["findings"] if f["title"] == "Missing No branch") == 1)
    dept = next(f for f in report["findings"] if f["title"] == "Dept variants")
    check("verifier rejection applied to finding", dept["verified"] is False)
    check("iso sections surfaced", report["iso"]["missing_sections"] == ["Records"])
    check("scores in report", 0 <= report["scores"]["overall"] <= 100)
    check("exec summary in report", report["summary"]["risk_level"] == "medium")
    check("agent statuses in report", len(report["agents"]) == len(pipeline.DOC_AGENTS))
    check("summary sentence mentions score", "Overall score" in result["summary"], result["summary"])
    check("corrected text merged", "the start" in result["corrected_text"])


def test_pipeline_clean_doc():
    pipeline.proofread_chunk = fake_chunk

    async def no_findings(client, agent, text):
        r = {"findings": [], "verdict": "Clean."}
        if agent["key"] == "iso":
            r.update({"present_sections": [], "missing_sections": [], "is_sop": False})
        return r

    async def keep_all(client, text, items):
        return {it["id"]: {"keep": True, "confidence": 1.0, "note": ""} for it in items}

    pipeline.run_doc_agent = no_findings
    pipeline.run_verifier = keep_all
    pipeline.run_summary = fake_summary
    result = asyncio.run(run_pipeline("A perfectly clean sentence."))
    check("clean doc: no changes", result["changes"] == [])
    check("clean doc: all scores 100",
          all(v == 100 for v in result["stats"]["report"]["scores"].values()))
    check("clean doc: unicode-safe on emoji doc",
          asyncio.run(run_pipeline("Café résumé 💰 naïve."))["stats"]["words"] >= 3)


if __name__ == "__main__":
    test_split_spans()
    test_anchoring()
    test_dedupe()
    test_scores()
    test_stats()
    test_pipeline_integration()
    test_pipeline_clean_doc()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        sys.exit(1)
    print("ALL TESTS PASSED")

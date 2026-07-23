# Proofreader Agent — Document Quality Assurance

Proofreader Agent is a format-aware document quality platform for business
documents, controlled procedures, policies, technical material, and ISO/QMS
documentation. Version 3 prioritizes evidence, low false-positive rates,
deterministic scoring, and graceful failure over the number of suggestions.

## Review architecture

```text
TXT / Markdown / DOCX / HTML / PDF
                 |
       extraction + format metadata
                 |
      boundary-aware correction chunks
                 |
   16 specialist reviewers in parallel
     |                         |
 deterministic rules       focused LLM review
 terminology              roles and sections
 headings                 procedure consistency
 numbering                workflows and flowcharts
 tables                   ISO/QMS applicability
 cross-references
     |                         |
     +------ evidence normalization ------+
                        |
          protected-content suppression
                        |
        batched final verifier (all items)
                        |
     confidence thresholds + overlap resolution
                        |
      verified-only output and deterministic scores
                        |
         SSE UI, exports, bounded history
```

The 17 review roles are:

1. Grammar
2. Spelling
3. Punctuation
4. Readability
5. Style
6. Terminology consistency
7. Role consistency
8. Section consistency
9. Procedure consistency
10. Cross-references
11. Heading structure
12. Numbering
13. Tables
14. Workflow
15. Flowchart
16. ISO/QMS
17. Final verification

Simple structural checks use deterministic code. Nuanced semantic checks use
focused model calls only when the document contains relevant signals. This
reduces cost, latency, and speculative findings.

## Verification contract

Every document-level candidate carries:

- finding
- evidence
- rule
- confidence
- severity
- reason
- suggested fix
- supporting context

The final verifier independently validates both the evidence and the rule.
Grammar, spelling, punctuation, structure, procedure, logic, and ISO findings
need confidence of at least 0.75. Style, readability, terminology, and
consistency findings need at least 0.85. Items below threshold, omitted verifier
items, and unanchored corrections are not returned as findings or corrections.

The correction engine suppresses edits inside:

- fenced and inline code
- URLs and Markdown link targets
- email addresses
- file paths
- table, figure, form, and record identifiers
- likely company, product, role, and defined multi-word terminology

Terminology variants are advisory findings; they are never automatically
replaced.

## Format awareness

| Format | Extraction and structural behavior |
|---|---|
| TXT | Plain-text and process heuristics |
| Markdown | Markdown heading levels, code spans, links, and table structure |
| DOCX | Paragraph/table order plus native Word heading-level metadata |
| HTML | Visible blocks, headings, list items, and rows; scripts/styles removed |
| PDF | Text by page with a 200-page safety limit |

Markdown heading rules are never applied to DOCX. DOCX hierarchy comes from
Word paragraph styles such as `Heading 1` and `Heading 2`.

Uploads are limited to 2 MB. DOCX files also have entry-count and expanded-size
limits to reject malformed or compressed-bomb archives.

## Long documents and stability

- Maximum input: 100,000 characters.
- Correction chunks target 6,000 characters and include read-only boundary
  context.
- Specialist agents receive the complete document, not a truncated prefix.
- Every verifier item is processed in bounded batches.
- External calls use transient-status retries, exponential backoff with jitter,
  `Retry-After`, per-call timeouts, and a whole-pipeline timeout.
- Agent and chunk failures are isolated and reported without discarding other
  successful work.
- SSE uses progress events, per-reviewer states, heartbeats, disconnect
  cancellation, and proxy buffering controls.
- API corrected text is reconstructed only from exact, verifier-approved
  offsets. Model-generated full text is never trusted for merging.

## Deterministic scoring

Quality areas are grammar, spelling, punctuation, readability, style,
consistency, structure, procedure, logic, and ISO/QMS. Scores start at 100 and
use fixed severity penalties, correction-density scaling, and fixed weights.
Only verified items affect scores. Overall risk is derived from the
deterministic overall score; the summary model cannot override it.

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Version, model, storage mode, and limits |
| `POST` | `/proofread` | Complete JSON review |
| `POST` | `/proofread/stream` | SSE progress and final result |
| `POST` | `/extract` | Extract TXT, Markdown, DOCX, HTML, or PDF |
| `POST` | `/export/docx` | Generate corrected DOCX |
| `GET` | `/jobs` | Recent memory/Supabase history |
| `GET` | `/jobs/{id}` | Retrieve one result |

Proofread request:

```json
{
  "text": "# Document",
  "document_type": "auto",
  "metadata": {}
}
```

`document_type` may be `auto`, `txt`, `markdown`, `docx`, `html`, or `pdf`.
The browser supplies extraction metadata automatically for uploaded files.

## Exports and review UI

The browser supports individual and bulk accept/reject decisions, correction
category filters, finding severity filters, finding/history search, markup and
clean views, and these exports:

- corrected TXT
- corrected DOCX
- Markdown review report
- CSV findings/corrections with spreadsheet-formula hardening
- full JSON
- decision audit JSON
- print/PDF

## Local development

Python 3.12.6 is the deployment version.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
export CEREBRAS_API_KEY=...
uvicorn app.main:app --reload
```

Quality gate:

```bash
ruff check .
ruff format --check .
mypy app
pytest --cov=app
python -m pip check
```

Coverage is enforced at 95%. The suite includes unit, integration, API, export,
frontend syntax/workflow, malformed DOCX, HTML/PDF, Unicode, retry, timeout,
concurrency, SSE, memory-history, maximum-size, protected-content, SOP, and
structural-review regression tests.

## Benchmarks

`benchmarks/corpus.json` is a human-labelled corpus covering business
copy-editing, technical Markdown, professional SOP review, terminology
consistency, and flowchart logic.

```bash
python benchmarks/evaluate.py https://proofreader-agent.onrender.com \
  --output benchmarks/latest.json
```

The preserved v2.0.0 live baseline (`benchmarks/baseline-v2.json`) measured:

- precision: 17.65%
- recall: 75.00%
- false-positive rate: 82.35%
- mean processing time: 115.581 seconds
- protected-content changes: 0

These figures use a deliberately strict, small gold corpus and are intended for
regression comparison, not general claims about all documents.

## Render deployment

`render.yaml` installs production dependencies, starts Uvicorn, configures
`/health`, and auto-deploys `main`. Required environment:

- `CEREBRAS_API_KEY`
- optional `CEREBRAS_MODEL`
- optional Supabase variables

## Known limitations

- Image-only/scanned PDFs need OCR before upload.
- Drawn flowchart images cannot be validated unless their nodes and edges are
  present in extracted text.
- DOCX export produces clean text rather than round-tripping the source file's
  full styles, tracked changes, headers, or embedded objects.
- Human approval remains mandatory for controlled-document release.
- Authentication and tenant isolation are intentionally outside the current
  evaluation scope.

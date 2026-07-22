# Proofreader Agent — End to End

An end-to-end proofreading agent: paste text or drop a file, and it reads the whole
document, marks every issue inline (grammar, spelling, punctuation, clarity,
consistency), lets you accept or reject each correction, then export the result.

**Stack:** FastAPI backend · Cerebras API for inference (fast Llama models, free
tier) · Supabase Postgres for job history (optional — falls back to in-memory) ·
deployable free on Render.

## What "end to end" means here

1. **Ingest** — paste text, or upload / drag-and-drop `.txt`, `.md`, `.docx` (extracted server-side).
2. **Chunk** — long documents (up to 100,000 chars) are split on paragraph/sentence boundaries into ~6k-char sections.
3. **Analyse** — sections are proofread concurrently on Cerebras; progress streams to the browser live (SSE).
   In parallel, a **whole-document structural review** checks terminology/role consistency, heading
   conventions, procedural logic (e.g. decision points missing a No branch), and cross-section
   consistency (prose steps vs tables/flowcharts) — reported as advisory findings, not auto-edits.
4. **Anchor** — every change is located at an exact character offset in the original, so the UI renders true inline markup.
5. **Review** — accept/reject each correction (inline or from the sidebar), filter by category, see stats and a category breakdown.
6. **Export** — copy the corrected text or download it as `.txt`; only *accepted* changes are applied.
7. **Persist** — every job is saved (Supabase if configured, in-memory otherwise) and reloadable from the History drawer.

If a section fails after retries, it degrades gracefully (returned unchanged with a
note) rather than failing the whole document.

## 1. Cerebras API key (required)

1. cloud.cerebras.ai → sign up → API Keys → create key.
2. Free tier gives generous daily token limits, no card needed.
3. Copy key → `CEREBRAS_API_KEY`.
4. Default model: `gpt-oss-120b`. Change via `CEREBRAS_MODEL` env var if needed
   (check cloud.cerebras.ai/models for the current list — availability changes).

## 2. Supabase (optional)

Without Supabase the app still works end to end — history is kept in memory
(lost on restart). For persistent history:

1. supabase.com → New project (free tier).
2. SQL Editor → paste `schema.sql` → Run. (Safe to re-run — it also migrates
   older tables to add the `stats` column.)
3. Project Settings → API → copy:
   - `Project URL` → `SUPABASE_URL`
   - `service_role` key (NOT anon) → `SUPABASE_SERVICE_KEY`

## 3. Local run

```bash
cd proofreader-agent
cp .env.example .env   # fill in real values (Supabase lines can stay as-is)
pip install -r requirements.txt
export $(cat .env | xargs)   # or use python-dotenv / direnv
uvicorn app.main:app --reload
```

Open http://localhost:8000 — paste text (or click "Try a sample"), then Proofread.

Or via curl:
```bash
curl -X POST http://localhost:8000/proofread \
  -H "Content-Type: application/json" \
  -d '{"text": "He dont know nothing about it."}'
```

## 4. Push to GitHub

```bash
git init && git add . && git commit -m "proofreader agent"
git remote add origin https://github.com/<you>/proofreader-agent.git
git push -u origin main
```

`.env` is in `.gitignore` — never commit real keys.

## 5. Deploy on Render (free)

1. render.com → New → Web Service → connect the GitHub repo.
2. Render detects `render.yaml` automatically (Blueprint). If not, set manually:
   - Runtime: Python 3
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - Plan: Free
3. Environment → add secrets: `CEREBRAS_API_KEY` (required), plus
   `CEREBRAS_MODEL`, `SUPABASE_URL`, `SUPABASE_SERVICE_KEY` (optional).
4. Deploy. Render gives you a live URL like `https://proofreader-agent.onrender.com`.

Note: Render free web services spin down after 15 min idle — first request after
idle takes ~30–50 s to wake up. Fine for a demo/internal tool.

## API

| Method | Path | Body | Returns |
|---|---|---|---|
| POST | `/proofread` | `{"text": "..."}` | `{id, corrected_text, changes[], summary, stats}` |
| POST | `/proofread/stream` | `{"text": "..."}` | SSE: `progress` events, then `result` |
| POST | `/extract` | multipart file (`.txt`/`.md`/`.docx`, max 2 MB) | `{filename, text, chars}` |
| GET | `/jobs/{id}` | — | stored job |
| GET | `/jobs?limit=20` | — | recent jobs |
| GET | `/health` | — | status, model, storage mode |

Each entry in `changes[]`:
```json
{
  "original": "their",
  "corrected": "they're",
  "category": "grammar",
  "reason": "Wrong homophone.",
  "severity": "minor",
  "start": 0,
  "end": 5
}
```
`start`/`end` are character offsets into the original text (`null` if the snippet
could not be located — such changes are listed but not auto-applied).

`stats.review` holds the document-level structural review:
```json
{
  "findings": [
    {"title": "...", "detail": "...", "category": "terminology|structure|logic|consistency", "severity": "minor|major"}
  ],
  "verdict": "one-sentence structural assessment",
  "truncated": false
}
```
The correction pass is also tuned against false positives: it will not rename
roles/titles/defined terms, will not impose optional style (e.g. the Oxford
comma or colons on headings) unless the document is internally inconsistent,
and prefers fewer high-confidence corrections over marginal ones.

## What this does NOT do

- No human sign-off gate beyond the in-app accept/reject review — wire your own
  approval step before anything goes to print (AI flags, human signs off).
- No auth on the endpoints. Add an API key check or Supabase auth before exposing
  this publicly beyond a demo.
- 100,000-char limit per request (~20k words). Longer manuscripts: split client-side.

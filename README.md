# Document Q&A with Citations (RAG over PDFs)

A Q&A system that answers questions using only the content of your PDFs,
always shows exactly which file/page/chunk it got the answer from, and
says clearly when the documents don't cover the question.

Built for the Potens.ai take-home (Q1).

## Tech stack

|     Piece     |                     Choice                    |
|---------------|-----------------------------------------------|
| LLM (answers) | Gemini `gemini-2.5-flash`                     |
| Embeddings    | Gemini `gemini-embedding-001` (768 dims)      |
| Vector store  | ChromaDB (local, no server needed)            |
| UI            | Streamlit                                     |
| API           | FastAPI (`/ask`, `/contradict`)               |

## Project structure

```
app/
  ingest.py   → reads PDFs in docs/, chunks them, embeds them, stores in Chroma
  rag.py      → the core logic: retrieve chunks, ask the LLM, detect contradictions
  api.py      → FastAPI wrapper exposing /ask and /contradict
  ui.py       → Streamlit app (the easiest way to try this)
  utils.py    → PDF reading, chunking, language detection
docs/         → put your PDFs here
chroma_db/    → the vector database (built automatically, gitignored)
.env          → your GOOGLE_API_KEY (gitignored)
```

## How to run it

```bash
# 1. set up environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. add your Gemini API key (free at https://aistudio.google.com/apikey)
cp .env.example .env
# open .env and paste your key in

# 3. get some PDFs into docs/ (or generate sample ones)
python scripts/make_sample_docs.py

# 4. build the index — do this once, and again whenever docs/ changes
cd app
python ingest.py

# 5. launch the UI
streamlit run ui.py
```

Prefer the API instead of the UI:

```bash
uvicorn api:app --reload --port 8000
```

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "How many days notice do I need for vacation?"}'

curl -X POST http://localhost:8000/contradict \
  -H "Content-Type: application/json" \
  -d '{"doc_a": "refund_policy_v1.pdf", "doc_b": "refund_policy_v2.pdf", "topic": "refund window"}'
```

## How chunking works

Each PDF page is broken into small overlapping pieces of text ("chunks") before being stored. Here's the process, step by step:

1. **Read page by page.** We never merge all pages into one big blob —
   each chunk stays tied to the exact page it came from, so we can always cite "page 3" and mean it.

2. **Cut on sentence boundaries.** We never chop a chunk mid-sentence.
   A sentence like *"Refunds are given within 30 days, provided the item is unused"* always stays whole — otherwise we could lose the condition attached to a fact.

3. **Target size: ~900 characters per chunk** (roughly 150–200 words).
   Small enough that a chunk is about *one* idea, big enough that it still makes sense on its own.

4. **Overlap consecutive chunks by 2 sentences.** The last two
   sentences of one chunk are repeated at the start of the next. This protects against an important fact landing right on the boundary between two chunks and getting "cut in half" in both.

5. **Every chunk gets an ID** like `refund_policy_v1.pdf::p1::c0`,
   meaning: *file → page → chunk number*. This ID doubles as the citation shown to the user — no separate lookup needed.


## What the output looks like

**`/ask` returns:**
```json
{
  "answer": "Employees must submit vacation requests at least two weeks in advance [employee_handbook.pdf::p1::c0].",
  "citations": [
    {
      "source_file": "employee_handbook.pdf",
      "page_number": 1,
      "chunk_id": "employee_handbook.pdf::p1::c0",
      "snippet": "Employees must submit vacation requests at least two weeks in advance through the HR portal..."
    }
  ],
  "language": "en",
  "grounded": true
}
```

- `answer` — the model's response, with citation tags like `[file::pN::cM]` inline.
- `citations` — the exact chunks used, so you can verify the answer yourself.
- `language` — detected automatically; the answer comes back in the same language as the question.
- `grounded` — `true` if the answer actually cited a chunk, `false` if the model had to say "not covered."

**If the documents don't cover the question:**
```json
{
  "answer": "The provided documents do not cover this question (no relevant content found).",
  "citations": [],
  "language": "en",
  "grounded": false
}
```

**`/contradict` returns:**
```json
{
  "contradiction": "YES",
  "reasoning": "Document A allows refunds within 30 days [refund_policy_v1.pdf::p1::c0], while Document B restricts this to 14 days [refund_policy_v2.pdf::p1::c0].",
  "evidence_a": [{"chunk_id": "refund_policy_v1.pdf::p1::c0", "page_number": 1}],
  "evidence_b": [{"chunk_id": "refund_policy_v2.pdf::p1::c0", "page_number": 1}]
}
```

`contradiction` is always one of `YES`, `NO`, or `INSUFFICIENT EVIDENCE`
— never a guess dressed up as certainty.


## Design decisions

- No hallucination, two layers deep. The prompt tells the model to only use the given context and to say plainly when it can't answer. On top of that, if retrieval finds literally nothing relevant, the system returns "not covered" *without even calling the LLM.

- One vendor (Gemini) for everything — embeddings and generation — to keep the stack simple for a one-day build.

- Multilingual by prompting, not by a separate translation step. The query's language is detected, and the same Gemini call that generates the answer is instructed to reply in that language. This keeps the pipeline to one LLM call instead of a translate → search → translate round trip.

## What's broken / unfinished

- Chroma always returns its top-k nearest chunks, even if none of them are actually relevant — there's no distance cutoff yet. The prompt is doing most of the anti-hallucination work rather than a hard threshold.

- No conversation memory — every question is independent, no follow-ups.

- No auth or rate limiting on the API (fine for local use, not for anything public).

- Contradiction detection only looks at the top few chunks matching a topic — it can miss a conflict the topic search doesn't surface.

- PDF reading is text-only — tables and scanned/image PDFs aren't handled specially.
- Only tested against the 5 sample PDFs so far.

## What I'd build next

- A similarity threshold so retrieval can honestly return "nothing relevant" instead of always returning top-k.

- Re-ranking retrieved chunks with a cross-encoder for better precision.

- Streaming answers in the UI instead of a full-response spinner.

- A small saved test set (question → expected answer/source) to catch regressions when the chunking or prompt changes.


## AI USE LOG

- Claude (Anthropic): ~[4] — architecture, all app code, SDK migration debugging.

- ChatGPT: ~[40] — problem understanding, project working flow, project structure
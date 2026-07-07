import os
import re
import time
from typing import List, Dict, Optional

import chromadb
from google import genai
from google.genai import types
from dotenv import load_dotenv

from utils import detect_language

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY not set. Add it to your .env file. "
        "Get a free key at https://aistudio.google.com/apikey"
    )
_client_ai = genai.Client(api_key=API_KEY)

EMBED_MODEL = "gemini-embedding-001"
GEN_MODEL = "gemini-2.5-flash"

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
COLLECTION_NAME = "documents"

_client = chromadb.PersistentClient(path=CHROMA_DIR)


def get_collection():
    return _client.get_or_create_collection(name=COLLECTION_NAME)


def embed_text(text: str, task_type: str) -> List[float]:
    """task_type is 'RETRIEVAL_DOCUMENT' at ingest time or
    'RETRIEVAL_QUERY' at query time - Gemini's embedding model uses this
    to optimize the vector for each role, which measurably improves
    retrieval quality vs. embedding both the same way.

    output_dimensionality is pinned explicitly (gemini-embedding-001
    defaults to 3072-dim, which is overkill and slower to store/search
    for a small doc set) - must match the value used in ingest.py or
    Chroma will reject/mis-compare vectors of different lengths."""
    result = _client_ai.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=768),
    )
    return result.embeddings[0].values


def _call_gemini(prompt: str, retries: int = 3) -> str:
    last_err = None
    for attempt in range(retries):
        try:
            response = _client_ai.models.generate_content(model=GEN_MODEL, contents=prompt)
            return response.text.strip()
        except Exception as e:  # rate limits / transient errors on free tier
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Gemini generation failed after {retries} attempts: {last_err}")


def list_sources() -> List[str]:
    """Distinct source filenames currently in the collection - used to
    populate the document pickers in the UI and validate /contradict input."""
    col = get_collection()
    data = col.get(include=["metadatas"])
    return sorted({m["source_file"] for m in data["metadatas"]})


def retrieve(query: str, k: int = 8, source_filter: Optional[str] = None) -> List[Dict]:
    col = get_collection()
    query_embedding = embed_text(query, task_type="RETRIEVAL_QUERY")

    where = {"source_file": source_filter} if source_filter else None
    results = col.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where=where,
    )

    hits = []
    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]
    for i in range(len(ids)):
        hits.append(
            {
                "chunk_id": ids[i],
                "text": docs[i],
                "source_file": metas[i]["source_file"],
                "page_number": metas[i]["page_number"],
                "distance": dists[i],
            }
        )
    return hits


def _format_context(hits: List[Dict]) -> str:
    blocks = []
    for h in hits:
        blocks.append(
            f"[{h['chunk_id']}] (source: {h['source_file']}, page: {h['page_number']})\n{h['text']}"
        )
    return "\n\n---\n\n".join(blocks)


ANSWER_PROMPT_TEMPLATE = """You are a careful document-QA assistant. Answer the QUESTION using ONLY the
CONTEXT below. The context is made of numbered chunks, each tagged with its
source file and page number.

Rules (follow all of them):
1. Only use facts that are explicitly present in the CONTEXT. Do not use
   outside knowledge, even if you are confident about it.
2. If the CONTEXT does not contain enough information to answer the
   question, you MUST say so explicitly and clearly - do not guess, and
   do not partially answer with unsupported filler. Say something like:
   "The provided documents do not cover this."
3. For every factual claim you make, cite the exact chunk id(s) it came
   from in square brackets, copied exactly as shown in the CONTEXT, e.g.
   [handbook.pdf::p3::c1]. Only cite chunks you actually used - do not
   cite a chunk just because it was provided if you didn't use it.
4. Answer in the same language the QUESTION was written in. Judge the
   QUESTION's language yourself directly - do not rely on any language
   hint elsewhere. Translate context content as needed, but keep the
   bracketed chunk-id citations in their original form (do not translate
   the citation tags themselves).
5. Be concise. Do not repeat the context verbatim; synthesize it.

Respond in EXACTLY this structure, with nothing before or after it:

LANGUAGE: <ISO 639-1 two-letter code for the language the QUESTION was written in>
ANSWER: <your answer, written entirely in that language, with bracketed citations>

CONTEXT:
{context}

QUESTION:
{question}
"""


def _parse_structured_answer(raw: str) -> Dict[str, str]:
    """Pulls LANGUAGE and ANSWER out of the model's structured response.
    Falls back gracefully if the model doesn't follow the format exactly
    (e.g. adds stray text), so a formatting slip never turns into a
    crash for the user."""
    lang_match = re.search(r"LANGUAGE:\s*([a-zA-Z-]+)", raw)
    answer_match = re.search(r"ANSWER:\s*(.*)", raw, re.DOTALL)
    language = lang_match.group(1).strip().lower() if lang_match else "en"
    answer = answer_match.group(1).strip() if answer_match else raw.strip()
    return {"language": language, "answer": answer}


def _extract_cited_chunk_ids(answer_text: str) -> set:
    """Finds every chunk id the model actually put in brackets in its
    answer (comma-separated ids within one bracket are supported), so we
    can show only the citations that were really used - not every chunk
    that happened to be retrieved."""
    cited = set()
    for bracket in re.findall(r"\[([^\]]+)\]", answer_text):
        for part in bracket.split(","):
            part = part.strip()
            if re.match(r"^[^:\s]+::p\d+::c\d+$", part):
                cited.add(part)
    return cited


def answer_query(query: str, k: int = 8, source_filter: Optional[str] = None) -> Dict:
    hits = retrieve(query, k=k, source_filter=source_filter)

    if not hits:
        return {
            "answer": "The provided documents do not cover this question (no relevant content found).",
            "citations": [],
            "language": detect_language(query),
            "grounded": False,
        }

    context = _format_context(hits)
    prompt = ANSWER_PROMPT_TEMPLATE.format(context=context, question=query)
    raw = _call_gemini(prompt)
    parsed = _parse_structured_answer(raw)

    cited_ids = _extract_cited_chunk_ids(parsed["answer"])
    # Anti-hallucination signal: did the model actually cite a chunk id
    # that we really retrieved, or did it fall back to "not covered"?
    grounded = len(cited_ids) > 0

    # Only surface chunks the model actually cited - showing every
    # retrieved-but-unused chunk as a "citation" would be misleading,
    # since retrieval always returns its top-k regardless of relevance.
    citations = [
        {
            "source_file": h["source_file"],
            "page_number": h["page_number"],
            "chunk_id": h["chunk_id"],
            "snippet": h["text"][:280] + ("..." if len(h["text"]) > 280 else ""),
        }
        for h in hits
        if h["chunk_id"] in cited_ids
    ]

    return {
        "answer": parsed["answer"],
        "citations": citations,
        "language": parsed["language"],
        "grounded": grounded,
    }


CONTRADICTION_PROMPT_TEMPLATE = """You are comparing two documents to see whether they make CONFLICTING claims
on a given topic. Below are excerpts from each document.

DOCUMENT A ({doc_a}):
{context_a}

DOCUMENT B ({doc_b}):
{context_b}

TOPIC TO COMPARE: {topic}

Instructions:
- Only judge based on the excerpts given. If the excerpts do not contain
  enough overlapping content to judge a conflict either way, say so
  explicitly instead of guessing.
- Respond in this exact format:

CONTRADICTION: <YES / NO / INSUFFICIENT EVIDENCE>
REASONING: <2-4 sentences explaining your judgment, citing chunk ids from
both documents in brackets, e.g. [a.pdf::p1::c0] vs [b.pdf::p2::c1]>
"""


def check_contradiction(doc_a: str, doc_b: str, topic: Optional[str] = None) -> Dict:
    """Retrieve the most relevant chunks from each document on `topic`
    (or a generic overlap probe if no topic given) and ask the LLM to
    judge whether they conflict."""
    probe = topic or "the main claims, rules, or figures in this document"

    hits_a = retrieve(probe, k=4, source_filter=doc_a)
    hits_b = retrieve(probe, k=4, source_filter=doc_b)

    if not hits_a or not hits_b:
        missing = doc_a if not hits_a else doc_b
        return {
            "contradiction": "INSUFFICIENT EVIDENCE",
            "reasoning": f"No indexed content found for '{missing}'. Check the document id.",
            "evidence_a": [],
            "evidence_b": [],
        }

    prompt = CONTRADICTION_PROMPT_TEMPLATE.format(
        doc_a=doc_a,
        context_a=_format_context(hits_a),
        doc_b=doc_b,
        context_b=_format_context(hits_b),
        topic=probe,
    )
    raw = _call_gemini(prompt)

    verdict = "INSUFFICIENT EVIDENCE"
    reasoning = raw
    for line in raw.splitlines():
        if line.upper().startswith("CONTRADICTION:"):
            verdict = line.split(":", 1)[1].strip()
        if line.upper().startswith("REASONING:"):
            reasoning = line.split(":", 1)[1].strip()

    return {
        "contradiction": verdict,
        "reasoning": reasoning,
        "evidence_a": [{"chunk_id": h["chunk_id"], "page_number": h["page_number"]} for h in hits_a],
        "evidence_b": [{"chunk_id": h["chunk_id"], "page_number": h["page_number"]} for h in hits_b],
    }
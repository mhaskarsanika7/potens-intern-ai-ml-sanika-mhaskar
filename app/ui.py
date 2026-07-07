import streamlit as st
import rag

st.set_page_config(page_title="Document Q&A RAG", layout="wide")
st.title("Document Q&A with Citations")

try:
    all_sources = rag.list_sources()
except Exception as e:
    st.error(f"Could not reach the vector store. Did you run `python ingest.py` first? ({e})")
    st.stop()

if not all_sources:
    st.warning("No documents ingested yet. Run `python ingest.py` from the app/ folder first.")
    st.stop()

st.caption(f"Indexed documents: {', '.join(all_sources)}")

tab_ask, tab_contradict = st.tabs(["Ask a question", "Compare two documents"])

with tab_ask:
    st.write("Ask in any language - the answer comes back in the same language.")
    query = st.text_input("Your question", placeholder="e.g. What is the cancellation policy?")
    scope = st.selectbox("Limit search to one document (optional)", ["All documents"] + all_sources)
    top_k = st.slider("Number of chunks to retrieve", min_value=3, max_value=15, value=8)

    if st.button("Ask", type="primary") and query.strip():
        with st.spinner("Retrieving and generating..."):
            source_filter = None if scope == "All documents" else scope
            result = rag.answer_query(query, k=top_k, source_filter=source_filter)

        st.subheader("Answer")
        if not result["grounded"]:
            st.info("The system did not find grounded support for a full answer - see below.")
        st.write(result["answer"])
        st.caption(f"Detected query language: `{result['language']}`")

        if result["citations"]:
            st.subheader("Citations")
            for c in result["citations"]:
                with st.expander(f"{c['source_file']} — page {c['page_number']} — {c['chunk_id']}"):
                    st.write(c["snippet"])
        else:
            st.caption("No chunks were retrieved for this query.")

with tab_contradict:
    st.write("Check whether two documents make conflicting claims on a topic.")
    col1, col2 = st.columns(2)
    with col1:
        doc_a = st.selectbox("Document A", all_sources, key="doc_a")
    with col2:
        remaining = [s for s in all_sources if s != doc_a] or all_sources
        doc_b = st.selectbox("Document B", remaining, key="doc_b")
    topic = st.text_input("Topic to compare (optional - leave blank for a general scan)")

    if st.button("Check for contradiction", type="primary"):
        if doc_a == doc_b:
            st.warning("Pick two different documents.")
        else:
            with st.spinner("Comparing documents..."):
                result = rag.check_contradiction(doc_a, doc_b, topic=topic or None)

            verdict = result["contradiction"]
            color = {"YES": "red", "NO": "green"}.get(verdict.upper(), "orange")
            st.markdown(f"### Verdict: :{color}[{verdict}]")
            st.write(result["reasoning"])

            colA, colB = st.columns(2)
            with colA:
                st.caption(f"Evidence from {doc_a}")
                for e in result["evidence_a"]:
                    st.code(e["chunk_id"])
            with colB:
                st.caption(f"Evidence from {doc_b}")
                for e in result["evidence_b"]:
                    st.code(e["chunk_id"])

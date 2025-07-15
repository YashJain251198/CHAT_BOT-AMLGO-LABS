import streamlit as st
from retriever import retrieve
from generator import generate_answer

st.set_page_config(page_title="RAG Chatbot - Amlgo Labs")
st.title("RAG Chatbot - Amlgo Labs")

if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Ask your question below:")

if query:
    with st.spinner("Retrieving answer..."):
        docs = retrieve(query)
        context = "\n\n".join(docs)
        answer = generate_answer(context, query)
        st.session_state.history.append((query, answer, docs))

for q, a, refs in reversed(st.session_state.history):
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Bot:** {a}")
    with st.expander("Sources"):
        for ref in refs:
            st.write(ref)

if st.sidebar.button("🔁 Reset"):
    st.session_state.history.clear()
    st.experimental_rerun()

st.sidebar.markdown("**Model:** Mistral-7B-Instruct")
st.sidebar.markdown(f"**Chunks in DB:** ~{len(refs) if query else 0}")

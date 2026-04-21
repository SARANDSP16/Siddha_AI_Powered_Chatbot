from pathlib import Path

import streamlit as st

from rag import DEFAULT_MODEL, RagError, get_groq_client, load_tree, vectorless_rag


TREE_PATHS = [
    Path("data/siddha_tree.json"),
    Path("siddha_tree.json"),
]


st.set_page_config(
    page_title="Siddha Vaithiyam Assistant",
    page_icon="🌿",
    layout="wide",
)


@st.cache_data
def cached_tree():
    for tree_path in TREE_PATHS:
        if tree_path.exists():
            return load_tree(tree_path)
    raise FileNotFoundError(
        "Tree file not found. Expected data/siddha_tree.json or siddha_tree.json."
    )


@st.cache_resource
def cached_client(api_key: str | None):
    return get_groq_client(api_key)


def get_secret_key() -> str | None:
    try:
        return st.secrets.get("GRO_API_KEY") or st.secrets.get("GROQ_API_KEY")
    except Exception:
        return None


st.title("Siddha Vaithiyam Assistant")
st.caption("Vectorless RAG over a structured Siddha Q&A knowledge tree.")

try:
    tree = cached_tree()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

with st.sidebar:
    st.header("Knowledge Base")
    st.write(f"Lessons: **{len(tree['lessons'])}**")
    st.write(f"Q&A records: **{tree['total_items']}**")

    model = st.selectbox(
        "Groq model",
        options=[
            DEFAULT_MODEL,
            "llama-3.1-8b-instant",
        ],
        index=0,
    )

    st.divider()
    st.subheader("Lessons")
    for lesson in tree["lessons"]:
        st.write(
            f"{lesson['lesson_id']}. {lesson['lesson_name']} "
            f"({lesson['items_count']})"
        )

st.info(
    "Ask a health question in Tamil. The app first selects the best lesson, "
    "then retrieves matching Q&A nodes, then writes an answer using only those records."
)

default_question = "முகப்பரு குணமாக என்ன மருந்து?"
query = st.text_area(
    "Question",
    value=default_question,
    height=90,
    placeholder="உங்கள் கேள்வியை இங்கே எழுதுங்கள்...",
)

run = st.button("Ask", type="primary", use_container_width=True)

if run:
    if not query.strip():
        st.warning("Please enter a question.")
        st.stop()

    api_key = get_secret_key()

    try:
        client = cached_client(api_key)
    except RagError as exc:
        st.error(str(exc))
        st.write("Local setup: add `GRO_API_KEY=...` to `.env`.")
        st.write("Streamlit Cloud: add `GRO_API_KEY` in app secrets.")
        st.stop()

    with st.spinner("Searching the Siddha tree with AI..."):
        try:
            result = vectorless_rag(
                query=query.strip(),
                tree=tree,
                client=client,
                model=model,
            )
        except Exception as exc:
            st.error("The RAG pipeline failed.")
            st.exception(exc)
            st.stop()

    st.subheader("Answer")
    st.write(result["answer"])

    with st.expander("Retrieval Trace", expanded=True):
        st.write("Lesson reasoning:", result["lesson_reasoning"])
        st.write("Selected lesson IDs:", result["lesson_ids"])
        st.write("Item reasoning:", result["item_reasoning"])
        st.write("Selected node IDs:", result["node_ids"])

    with st.expander("Source Records"):
        if not result["nodes"]:
            st.write("No source records selected.")
        for node in result["nodes"]:
            st.markdown(f"### {node['node_id']} · {node['title']}")
            st.write(f"Lesson: {node['lesson_name']}")
            st.write(node["text"])

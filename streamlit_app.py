
import requests
import streamlit as st

st.set_page_config(
    page_title="My Web + Local RAG Agent",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 My AI Research Assistant")
st.caption("Local documents + web search + Llama 3.2")

# Set these in Streamlit Cloud > App settings > Secrets.
BACKEND_URL = st.secrets.get("BACKEND_URL", "").rstrip("/")
API_TOKEN = st.secrets.get("RAG_API_TOKEN", "")

if not BACKEND_URL or not API_TOKEN:
    st.error(
        "Backend configuration is missing. Set BACKEND_URL "
        "and RAG_API_TOKEN in the Streamlit app secrets."
    )
    st.stop()

HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}

with st.sidebar:
    st.header("Documents")
    st.write("Upload files to index them on your PC.")

    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX or TXT",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    if st.button("Upload and index", disabled=not uploaded_files):
        progress = st.progress(0)

        for index, uploaded_file in enumerate(uploaded_files):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/upload",
                    headers=HEADERS,
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type or
                            "application/octet-stream",
                        )
                    },
                    timeout=180,
                )
                response.raise_for_status()
                data = response.json()

                st.success(
                    f"{data['filename']}: "
                    f"{data['chunks_indexed']} chunks indexed"
                )

            except requests.RequestException as exc:
                st.error(
                    f"Upload failed for {uploaded_file.name}: {exc}"
                )

            progress.progress((index + 1) / len(uploaded_files))

    st.divider()
    st.info(
        "Every question searches your local knowledge base "
        "and the web. Your PC must be online."
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    label = (
                        f"[{source['id']}] "
                        f"{source['title']} "
                        f"({source['type']})"
                    )

                    if source["type"] == "web":
                        st.markdown(
                            f"- {label}: {source['source']}"
                        )
                    else:
                        st.markdown(
                            f"- {label}: `{source['source']}`"
                        )

if question := st.chat_input("Ask a question..."):
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching your documents and the web..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/chat",
                    headers=HEADERS,
                    json={"question": question},
                    timeout=240,
                )
                response.raise_for_status()
                data = response.json()

                answer = data["answer"]
                sources = data.get("sources", [])

                st.markdown(answer)

                counts = data.get("retrieval", {})
                st.caption(
                    f"Local results: {counts.get('local_results', 0)} · "
                    f"Web results: {counts.get('web_results', 0)}"
                )

                if sources:
                    with st.expander("Sources"):
                        for source in sources:
                            label = (
                                f"[{source['id']}] "
                                f"{source['title']} "
                                f"({source['type']})"
                            )
                            if source["type"] == "web":
                                st.markdown(
                                    f"- {label}: {source['source']}"
                                )
                            else:
                                st.markdown(
                                    f"- {label}: `{source['source']}`"
                                )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except requests.RequestException as exc:
                st.error(
                    "Could not reach your local RAG backend. "
                    "Check your PC, tunnel, URL and access token. "
                    f"Details: {exc}"
                )

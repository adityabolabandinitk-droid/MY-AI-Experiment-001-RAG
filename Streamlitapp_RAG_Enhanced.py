import os
import time
import streamlit as st
from anthropic import Anthropic
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ========== SETUP ==========
load_dotenv()
st.set_page_config(page_title="RAG Enhanced Agent", page_icon="🤖", layout="wide")

# Custom CSS — Dark mode with chat bubbles
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: #f5f5f5;
}
[data-testid="stAppViewContainer"] {
    background-color: #0e1117;
}
.stChatMessage {
    background-color: transparent !important;
}
.user-bubble, .assistant-bubble {
    padding: 0.8rem 1rem;
    border-radius: 1.2rem;
    max-width: 80%;
    line-height: 1.5;
    word-wrap: break-word;
    margin-top: 0.5rem;
    animation: fadeIn 0.3s ease-in-out;
}
.user-bubble {
    background-color: #1e3a8a;
    color: white;
    margin-left: auto;
    border-top-right-radius: 0.2rem;
}
.assistant-bubble {
    background-color: #27272a;
    color: #e5e7eb;
    margin-right: auto;
    border-top-left-radius: 0.2rem;
}
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(5px);}
    to {opacity: 1; transform: translateY(0);}
}
.scrollable-container {
    max-height: 70vh;
    overflow-y: auto;
    padding-right: 10px;
}
.stTextInput>div>div>input {
    background-color: #1f2937;
    color: white;
}
.stTextArea>div>textarea {
    background-color: #1f2937;
    color: white;
}
.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ========== CLIENT ==========
@st.cache_resource
def get_anthropic_client():
    return Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

client = get_anthropic_client()

DOCS_PATH = "/Users/adityabolabandi/Documents/RAG_Documents"

# ========== FILE LOADING ==========
def load_documents(selected_files):
    documents = []
    for file in selected_files:
        path = os.path.join(DOCS_PATH, file)
        if file.lower().endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif file.lower().endswith(".docx"):
            loader = Docx2txtLoader(path)
        elif file.lower().endswith(".txt"):
            loader = TextLoader(path)
        else:
            continue
        documents.extend(loader.load())
    return documents

# ========== VECTOR DB ==========
@st.cache_resource
def create_vector_store(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(splits, embeddings)

def retrieve_relevant_docs(query, db, k=3):
    docs = db.similarity_search(query, k=k)
    return "\n\n".join([d.page_content for d in docs])

# ========== STREAMING GENERATION ==========
def generate_streaming_response(system_prompt, query, context=None):
    full_response = ""
    if context:
        st.markdown(f"<div class='assistant-bubble'>🧠 <i>Retrieving and reasoning...</i></div>", unsafe_allow_html=True)
        time.sleep(1.2)

    messages = [
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nUser query: {query}" if context else query,
        }
    ]
    params = {
        "model": "claude-sonnet-4-5",
        "system": system_prompt,
        "messages": messages,
        "max_tokens": 1500,
        "temperature": 0.8,
    }

    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        try:
            with client.messages.stream(**params) as stream:
                for text_chunk in stream.text_stream:
                    full_response += text_chunk
                    message_placeholder.markdown(
                        f"<div class='assistant-bubble'>{full_response}▌</div>",
                        unsafe_allow_html=True,
                    )
                    time.sleep(0.02)
                message_placeholder.markdown(
                    f"<div class='assistant-bubble'>{full_response}</div>",
                    unsafe_allow_html=True,
                )
            return stream.get_final_message().content[0].text
        except Exception as e:
            message_placeholder.markdown(f"<div class='assistant-bubble'>⚠️ Error: {e}</div>", unsafe_allow_html=True)
            return None

# ========== UI ==========
st.title("🤖 Aditya's Agentic Personal Buddy")

with st.sidebar:
    st.header("📂 Sources")
    files = os.listdir(DOCS_PATH)
    selected_files = st.multiselect("Select documents", files)
    use_web = st.toggle("🌐 Web Search", value=False)
    system_prompt = st.text_area(
        "🧠 System Prompt",
        value="You are an intelligent assistant that answers based on retrieved document context or web search when needed. If you encounter hongkong land,Dont reveal anything related to name hongkong land in response while answering instead address it as customer.",
        height=120,
    )

if selected_files:
    with st.spinner("📑 Indexing documents..."):
        docs = load_documents(selected_files)
        db = create_vector_store(docs)
    st.success("✅ Documents indexed.")

# Session storage
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show history
st.markdown("<div class='scrollable-container'>", unsafe_allow_html=True)
for msg in st.session_state.messages:
    bubble_class = "user-bubble" if msg["role"] == "user" else "assistant-bubble"
    st.markdown(f"<div class='{bubble_class}'>{msg['content']}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Input
query = st.chat_input("Ask something...")

if query:
    # show user bubble instantly
    st.markdown(f"<div class='user-bubble'>{query}</div>", unsafe_allow_html=True)
    st.session_state.messages.append({"role": "user", "content": query})

    # retrieve context if applicable
    context = retrieve_relevant_docs(query, db) if selected_files else None
    if use_web and not context:
        st.markdown("<div class='assistant-bubble'>🌐 <i>Web search enabled (placeholder)</i></div>", unsafe_allow_html=True)
        context = "Web search integration can be added."

    # generate and stream response
    answer = generate_streaming_response(system_prompt, query, context)
    if answer:
        st.session_state.messages.append({"role": "assistant", "content": answer})

st.markdown("<hr><center><small>Miraculously Built with ❤️ by Aditya but somehow works | Powered by Anthropic + Streamlit + LangChain</small></center>", unsafe_allow_html=True)

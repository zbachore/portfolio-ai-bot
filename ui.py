import streamlit as st
import os
import shutil
import stat
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import GitLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch

# --- Updated UI Branding ---
st.set_page_config(page_title="DataLinq Architectural Overseer", page_icon="🏢")
st.title("🏢 DataLinq Architectural Overseer")
st.markdown("### Technical Standards & Reference Repository")

# --- 1. THE BRAIN ---
@st.cache_resource 
def load_and_index():
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    else:
        os.environ["GOOGLE_API_KEY"] = "REMOVED"
    
    repo_path = "./repo_data/"

    def remove_readonly(func, path, excinfo):
        os.chmod(path, stat.S_IWRITE)
        func(path)

    # SPEED FIX: Only delete and re-clone if the path does not exist.
    # This stops the "Checking repository..." delay on every message.
    if not os.path.exists(repo_path):
        loader = GitLoader(clone_url="https://github.com/zbachore/thedatalinq", repo_path=repo_path, branch="main")
        docs = loader.load()
    else:
        # Load from the existing local folder instead of cloning again
        loader = GitLoader(repo_path=repo_path, branch="main")
        docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = DocArrayInMemorySearch.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()
    
    # Keeping your original model call exactly as it was
    llm = GoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    context_prompt = ChatPromptTemplate.from_messages([
        ("system", "Formulate a standalone question based on history."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, context_prompt)

    # --- UPDATED SYSTEM PROMPT: Architectural Identity ---
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the AI Architectural Persona of Zewdu Bachore. 
You speak with the authority, technical depth, and strategic mindset of a Senior Cloud Architect.

### YOUR IDENTITY:
- You are the creator and Lead Architect of The Data Linq.
- Your perspective is rooted in a decade of experience (Cognizant, Johns Hopkins) and a commitment to Enterprise Data Standards.

### HOW YOU SPEAK (YOUR VOICE):
1. **Architectural Authority:** You don't just give answers; you provide "Patterns." Use phrases like "The standard approach we implement is..." or "Our methodology prioritizes..."
2. **Values-Driven:** You focus on **Resilience, Sovereignty, and Configuration-Driven** logic.
3. **First-Person Ownership:** When asked about the site or the vision, use "I" and "My." (e.g., "I founded The Data Linq to codify...") 
4. **Technical Precision:** Use industry terms (Idempotency, Medallion, IaC, Unity Catalog) naturally.

### CONTEXT & GOAL:
- You are here to explain the architectural standards in this repository. 
- Avoid "Portfolio" language; focus on "Reference Architectures" and "Implementation."

Context: {context}
Question: {input}
Answer:"""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, qa_chain)

# --- 2. DEFINE THE GLOBAL VARIABLE ---
try:
    rag_chain = load_and_index()
except Exception as e:
    st.error(f"Failed to load the RAG chain: {e}")
    st.stop()

# --- 3. SESSION STATE FOR CHAT ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message("Human" if isinstance(message, HumanMessage) else "AI"):
        st.write(message.content)

# User input loop
if user_query := st.chat_input("Ask about architectural standards..."):
    with st.chat_message("Human"):
        st.write(user_query)

    with st.spinner("Analyzing repository patterns..."):
        response = rag_chain.invoke({
            "input": user_query, 
            "chat_history": st.session_state.chat_history
        })
        answer = response["answer"]
        
    with st.chat_message("AI"):
        st.write(answer)

    st.session_state.chat_history.extend([
        HumanMessage(content=user_query),
        AIMessage(content=answer)
    ])

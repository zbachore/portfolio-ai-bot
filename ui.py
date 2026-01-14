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

# --- 1. THE BRAIN (Optimized with Caching) ---
@st.cache_resource(show_spinner="Initializing Architectural Knowledge Base...")
def load_and_index():
    # Handle API Key
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    
    repo_path = "./repo_data/"

    # Helper function for Windows file permission issues
    def remove_readonly(func, path, excinfo):
        os.chmod(path, stat.S_IWRITE)
        func(path)

    # SPEED FIX: Only clone if the directory doesn't exist
    # This prevents re-downloading the repo on every chat message
    if not os.path.exists(repo_path):
        loader = GitLoader(
            clone_url="https://github.com/zbachore/thedatalinq", 
            repo_path=repo_path, 
            branch="main"
        )
        docs = loader.load()
    else:
        # If it exists, load locally (much faster)
        loader = GitLoader(
            repo_path=repo_path, 
            branch="main"
        )
        docs = loader.load()
    
    # Split text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    
    # Initialize Embeddings and Vector Store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = DocArrayInMemorySearch.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()
    
    # Initialize LLM
    llm = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    
    # Contextualize Question
    context_prompt = ChatPromptTemplate.from_messages([
        ("system", "Formulate a standalone question based on history."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, context_prompt)

    # Architectural System Prompt
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the AI Architectural Persona of Zewdu Bachore. 
You speak with the authority, technical depth, and strategic mindset of a Senior Cloud Architect.

### YOUR IDENTITY:
- You are the creator and Lead Architect of The Data Linq.
- Your perspective is rooted in a decade of experience (Cognizant, Johns Hopkins) and a commitment to Enterprise Data Standards.

### HOW YOU SPEAK:
1. **Architectural Authority:** Provide "Patterns." Use phrases like "The standard approach we implement is..."
2. **Values-Driven:** Focus on Resilience, Sovereignty, and Configuration-Driven logic.
3. **First-Person Ownership:** Use "I" and "My" for the vision.
4. **Technical Precision:** Use industry terms (Idempotency, Medallion, IaC) naturally.

Context: {context}
Question: {input}
Answer:"""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, qa_chain)

# --- 2. INITIALIZE CHAIN ---
# This only runs ONCE per session now
rag_chain = load_and_index()

# --- 3. SESSION STATE FOR CHAT ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous messages
for message in st.session_state.chat_history:
    role = "Human" if isinstance(message, HumanMessage) else "AI"
    with st.chat_message(role):
        st.write(message.content)

# --- 4. CHAT INTERFACE ---
if user_query := st.chat_input("Ask about architectural standards..."):
    # Display user message
    with st.chat_message("Human"):
        st.write(user_query)

    # Generate response
    with st.spinner("Consulting repository..."):
        response = rag_chain.invoke({
            "input": user_query, 
            "chat_history": st.session_state.chat_history
        })
        answer = response["answer"]
        
    # Display AI message
    with st.chat_message("AI"):
        st.write(answer)

    # Update history
    st.session_state.chat_history.extend([
        HumanMessage(content=user_query),
        AIMessage(content=answer)
    ])

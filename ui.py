import streamlit as st
import os
import shutil
import stat
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import GitLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch

# Page Configuration
st.set_page_config(page_title="DataLinq Architectural Overseer", page_icon="🏢")
st.title("🏢 DataLinq Architectural Overseer")
st.markdown("### Technical Standards & Reference Repository")

# --- 1. THE BRAIN (RAG Logic) ---
@st.cache_resource 
def load_and_index():
    # API Key Configuration
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    else:
        st.error("Missing GOOGLE_API_KEY in Streamlit Secrets.")
        st.stop()

    repo_path = "./repo_data/"

    # Handle Permission issues for clearing existing repo data
    def remove_readonly(func, path, excinfo):
        os.chmod(path, stat.S_IWRITE)
        func(path)

    if os.path.exists(repo_path):
        shutil.rmtree(repo_path, onerror=remove_readonly)

    # Load documentation from GitHub
    try:
        loader = GitLoader(
            clone_url="https://github.com/zbachore/thedatalinq", 
            repo_path=repo_path, 
            branch="main"
        )
        docs = loader.load()
    except Exception as e:
        st.error(f"Failed to clone repository: {e}")
        st.stop()
    
    # Text Processing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    
    # Embeddings and Vector Store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = DocArrayInMemorySearch.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()
    
    # Setup Chat LLM (Corrected model and class)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    
    # History-aware rephrasing prompt
    context_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, context_prompt)

    # Answer generation system prompt
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the DataLinq Architectural Overseer, an authoritative technical documentation specialist.
        
Your purpose is to explain the **architectural standards and reference patterns** hosted in The Data Linq repository. 

### CORE IDENTITY & CONTEXT:
1. This site is an **Independent Repository for Enterprise Data Standards**, not a personal portfolio or project.
2. The repository provides **Reference Architectures** for Cloud Foundations, Lakehouse Engineering, and Governance.
3. Your mission is to help architects and engineers implement **configuration-driven frameworks** and **resilient cloud patterns**.

### RESPONSE GUIDELINES:
- Use professional, objective, and authoritative language.
- Refer to the content as "The Standard" or "The Repository Patterns."
- Never use the word "Portfolio."
- Do not mention personal names unless specifically asked for the Lead Contributor.
- **NEVER print citations, file names, or 'Sources' at the end of your response.** - Provide clean, direct Markdown-formatted architectural advice.

Context: {context}
Question: {input}
Answer:"""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, qa_chain)

# --- 2. INITIALIZE CHAIN ---
try:
    rag_chain = load_and_index()
except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.stop()

# --- 3. SESSION STATE FOR CHAT ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Render chat history
for message in st.session_state.chat_history:
    role = "Human" if isinstance(message, HumanMessage) else "AI"
    with st.chat_message(role):
        st.write(message.content)

# User Interaction
if user_query := st.chat_input("Ask about architectural patterns..."):
    with st.chat_message("Human"):
        st.write(user_query)

    with st.spinner("Analyzing Repository Patterns..."):
        try:
            response = rag_chain.invoke({
                "input": user_query, 
                "chat_history": st.session_state.chat_history
            })
            answer = response["answer"]
            
            with st.chat_message("AI"):
                st.write(answer)

            # Update Session History
            st.session_state.chat_history.extend([
                HumanMessage(content=user_query),
                AIMessage(content=answer)
            ])
        except Exception as e:
            st.error(f"Error generating response: {e}")

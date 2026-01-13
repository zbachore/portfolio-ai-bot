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

st.set_page_config(page_title="Data Linq AI Assistant", page_icon="🤖")
st.title("🤖 Data Linq Portfolio Assistant")

# --- 1. THE BRAIN (Now wrapped to ensure rag_chain is defined) ---
@st.cache_resource # This keeps the bot from re-downloading your GitHub every time you click
def load_and_index():
    # This tells the bot to look in the Streamlit Cloud "Safe" for the key
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    else:
        os.environ["GOOGLE_API_KEY"] = "REMOVED"
    repo_path = "./repo_data/"

    # Handle Windows Permission issues
    def remove_readonly(func, path, excinfo):
        os.chmod(path, stat.S_IWRITE)
        func(path)

    if os.path.exists(repo_path):
        shutil.rmtree(repo_path, onerror=remove_readonly)

    # Load from GitHub
    loader = GitLoader(clone_url="https://github.com/zbachore/thedatalinq", repo_path=repo_path, branch="main")
    docs = loader.load()
    
    # Split and Embed
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = DocArrayInMemorySearch.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()
    
    # Setup LLM & Chain
    llm = GoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    # History-aware rephrasing
    context_prompt = ChatPromptTemplate.from_messages([
        ("system", "Formulate a standalone question based on history."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, context_prompt)

# Answer generation
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the Digital AI Twin of Zewdu Bachore, Sr. Architect and the creator of The Data Linq.
You are a balanced Technical Project Assistant for The Data Linq. 

Your goal is to explain the repository's code, architecture, and data engineering patterns.

### GUIDELINES:
- While "Zero-Touch" is a core philosophy of this project, do not force it into every answer.
- Only mention Zero-Touch if the user specifically asks about it or if it is directly relevant to a technical question about automation.
- Focus equally on Spark logic, Azure infrastructure, Terraform, and general Data Engineering best practices.
- Keep your tone professional, helpful, and concise. 

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

# Display Chat History
for message in st.session_state.chat_history:
    with st.chat_message("Human" if isinstance(message, HumanMessage) else "AI"):
        st.write(message.content)

# User input loop
if user_query := st.chat_input("Ask me about the portfolio..."):
    with st.chat_message("Human"):
        st.write(user_query)

    with st.spinner("Searching repository..."):
        # The chain returns a dictionary containing 'answer' and 'context'
        response = rag_chain.invoke({
            "input": user_query, 
            "chat_history": st.session_state.chat_history
        })
        answer = response["answer"]
        
        # --- NEW: Extract unique source filenames ---
        sources = []
        if "context" in response:
            for doc in response["context"]:
                # 'source' is a metadata field automatically added by GitLoader
                source_name = doc.metadata.get("source", "Unknown Source")
                # Clean the path to show only the filename
                clean_name = os.path.basename(source_name)
                if clean_name not in sources:
                    sources.append(clean_name)

    with st.chat_message("AI"):
        st.write(answer)
        
        # Display the sources as small "chips" or labels
        if sources:
            st.markdown("---")
            st.caption("🔍 **Sources analyzed for this answer:**")
            cols = st.columns(len(sources) if len(sources) < 5 else 5)
            for i, source in enumerate(sources):
                cols[i % 5].status(f"`{source}`", state="complete")

    # Update Session History (Save ONLY the answer text for the next turn)
    st.session_state.chat_history.extend([
        HumanMessage(content=user_query),
        AIMessage(content=answer)

    ])






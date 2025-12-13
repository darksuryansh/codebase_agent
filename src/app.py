"""
Streamlit UI: Interactive chat interface with source code viewer.
Left panel: Chat | Right panel: Source code display
"""

import streamlit as st
from pygments import highlight
from pygments.lexers import get_lexer_by_name, TextLexer
from pygments.formatters import HtmlFormatter
import os

from agent import CodebaseAgent
from vector_store import VectorStore


# Page config
st.set_page_config(
    page_title="Codebase Onboarding Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better code display
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .code-container {
        background-color: #1e1e1e;
        border-radius: 8px;
        padding: 20px;
        margin: 10px 0;
        overflow-x: auto;
    }
    .file-badge {
        background-color: #667eea;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        display: inline-block;
        margin-right: 8px;
    }
    .stTextArea textarea {
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_agent(index_name: str = None):
    """Load agent with specific index (cached to avoid reloading)."""
    return CodebaseAgent(index_name=index_name)


def get_available_indexes():
    """Get list of available vector store indexes."""
    import os
    vector_db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'vector_db')
    if not os.path.exists(vector_db_path):
        return []
    
    # List all directories in vector_db folder
    indexes = [d for d in os.listdir(vector_db_path) 
               if os.path.isdir(os.path.join(vector_db_path, d)) and 
               os.path.exists(os.path.join(vector_db_path, d, 'index.faiss'))]
    return indexes


def syntax_highlight(code: str, language: str) -> str:
    """Apply syntax highlighting to code."""
    try:
        lexer = get_lexer_by_name(language, stripall=True)
    except:
        lexer = TextLexer()
    
    formatter = HtmlFormatter(style='monokai', noclasses=True)
    highlighted = highlight(code, lexer, formatter)
    return highlighted


def display_source(source: dict, index: int):
    """Display a source code snippet with syntax highlighting."""
    file_path = source['file_path']
    language = source['language']
    content = source['content']
    
    st.markdown(f"### 📄 Source {index}: `{file_path}`")
    st.markdown(f"<span class='file-badge'>{language}</span>", unsafe_allow_html=True)
    
    # Syntax highlighted code
    highlighted_code = syntax_highlight(content, language)
    st.markdown(f'<div class="code-container">{highlighted_code}</div>', unsafe_allow_html=True)
    
    # Copy button
    st.code(content, language=language, line_numbers=False)


def ingest_repository(repo_url: str):
    """Ingest a repository from URL."""
    import subprocess
    import sys
    
    try:
        # Run the ingestor script
        result = subprocess.run(
            [sys.executable, "src/ingestor.py", "--repo_path", repo_url],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        if result.returncode == 0:
            return True, "✅ Repository ingested successfully!"
        else:
            return False, f"❌ Error: {result.stderr}"
    except subprocess.TimeoutExpired:
        return False, "❌ Ingestion timed out (>10 minutes)"
    except Exception as e:
        return False, f"❌ Error: {str(e)}"


def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 Codebase Onboarding Agent</h1>', unsafe_allow_html=True)
    st.markdown("*Ask questions about your codebase and get instant answers with source code references*")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 System Info")
        
        # Add new repository section
        st.subheader("➕ Add Repository")
        with st.form("ingest_form"):
            repo_url = st.text_input(
                "Git Repository URL",
                placeholder="https://github.com/username/repo.git",
                help="Enter a GitHub/GitLab repository URL to ingest"
            )
            submit_button = st.form_submit_button("🚀 Ingest Repository")
            
            if submit_button and repo_url:
                with st.spinner("🔄 Cloning and processing repository... This may take a few minutes."):
                    success, message = ingest_repository(repo_url)
                    if success:
                        st.success(message)
                        st.info("🔄 Refreshing page to load new repository...")
                        st.rerun()
                    else:
                        st.error(message)
        
        st.divider()
        
        # Get available indexes
        available_indexes = get_available_indexes()
        
        if not available_indexes:
            st.warning("⚠️ No repositories ingested yet. Add one above!")
            return
        
        # Repo selector
        st.subheader("🔍 Select Repository")
        selected_index = st.selectbox(
            "Choose a repository:",
            available_indexes,
            help="Each ingested repository has its own isolated index"
        )
        
        # Load agent with selected index
        try:
            agent = load_agent(index_name=selected_index)
            stats = agent.vector_store.get_collection_stats()
            
            st.divider()
            st.metric("Total Code Chunks", stats.get('total_documents', 0))
            st.metric("Active Repository", selected_index)
            
            if stats.get('total_documents', 0) == 0:
                st.warning("⚠️ This index appears empty.")
            else:
                st.success("✅ Repository loaded")
        
        except Exception as e:
            st.error(f"❌ Error loading agent: {e}")
            return
        
        st.divider()
        
        st.header("💡 Example Questions")
        example_questions = [
            "Where is the main entry point?",
            "Show me authentication logic",
            "How is the database connected?",
            "Where are API routes defined?",
            "Explain the configuration setup",
        ]
        
        for question in example_questions:
            if st.button(question, key=f"ex_{question}"):
                st.session_state.example_question = question
    
    # Main content area - Two columns
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.header("💬 Chat")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Check for example question
        if "example_question" in st.session_state:
            user_question = st.session_state.example_question
            del st.session_state.example_question
        else:
            user_question = st.chat_input("Ask about the codebase...")
        
        # Process user input
        if user_question:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_question})
            
            with st.chat_message("user"):
                st.markdown(user_question)
            
            # Get answer from agent
            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching codebase..."):
                    result = agent.query(user_question)
                    answer = result['answer']
                    sources = result['sources']
                
                st.markdown(answer)
                
                # Store answer and sources
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })
                st.session_state.latest_sources = sources
            
            # Force rerun to display sources in right column
            st.rerun()
    
    with col2:
        st.header("📂 Source Code")
        
        # Display sources if available
        if "latest_sources" in st.session_state and st.session_state.latest_sources:
            sources = st.session_state.latest_sources
            
            st.info(f"📎 Found {len(sources)} relevant code snippets")
            
            # Tabs for each source
            if len(sources) > 0:
                tabs = st.tabs([f"File {i+1}" for i in range(len(sources))])
                
                for i, (tab, source) in enumerate(zip(tabs, sources)):
                    with tab:
                        display_source(source, i+1)
        else:
            st.info("👈 Ask a question to see relevant source code here")
            st.markdown("""
            **How it works:**
            1. Type your question in the chat
            2. The AI searches the codebase semantically
            3. Source code snippets appear here
            4. Get instant answers with file paths
            """)
    
    # Footer
    st.divider()
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            if "latest_sources" in st.session_state:
                del st.session_state.latest_sources
            st.rerun()
    
    with col_b:
        st.markdown("**Powered by:** Gemini Pro + LangGraph")
    
    # with col_c:
    #     st.markdown("**Embeddings:** Jina Code v2")


if __name__ == "__main__":
    main()

"""
LangGraph Agent: The brain of the codebase onboarding system.
Implements a state graph with Router → Retriever → Grader → Generator flow.
"""

import os
from typing import TypedDict, List, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

from vector_store import VectorStore
from utils import load_config, generate_repo_tree

# Load environment variables
load_dotenv()


class AgentState(TypedDict):
    """State that gets passed between nodes in the graph."""
    question: str
    retrieved_docs: List[Document]
    generation: str
    retry_count: int
    search_query: str
    relevance_grade: str


class CodebaseAgent:
    #LangGraph-based agent for codebase Q&A.
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.vector_store = VectorStore(config_path)
        self.max_retries = self.config.get('max_retries', 3)
        
        # Initialize Gemini LLM
        llm_config = self.config.get('llm', {})
        self.llm = ChatGoogleGenerativeAI(
            model=llm_config.get('model', 'gemini-2.5-flash'),
            temperature=llm_config.get('temperature', 0.1),
            google_api_key=os.getenv('GOOGLE_API_KEY'),
        )
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("router", self.router_node)
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("grade", self.grade_node)
        workflow.add_node("rewrite_query", self.rewrite_query_node)
        workflow.add_node("generate", self.generate_node)
        
        # Define edges
        workflow.set_entry_point("router")
        
        workflow.add_edge("router", "retrieve")
        workflow.add_edge("retrieve", "grade")
        
        # Conditional edge: if grade is bad and retries left, rewrite query
        workflow.add_conditional_edges(
            "grade",
            self.decide_after_grading,
            {
                "generate": "generate",
                "rewrite": "rewrite_query",
                "give_up": "generate",
            }
        )
        
        workflow.add_edge("rewrite_query", "retrieve")
        workflow.add_edge("generate", END)
        
        return workflow.compile()
    
    def router_node(self, state: AgentState) -> AgentState:
        """
        Router: Analyzes the user question and creates initial search query.
        """
        question = state['question']
        
        # Use LLM to understand the query and create search terms
        prompt = ChatPromptTemplate.from_template(
            """You are a code navigation assistant. 
            
User Question: {question}

Your task: Create a concise search query (2-5 keywords) to find relevant code.
Focus on: function names, class names, technical terms, or concepts.

Examples:
- "Where is user login?" → "user authentication login"
- "How does payment work?" → "payment processing stripe"
- "Show me database models" → "database models ORM"

Search Query:"""
        )
        
        chain = prompt | self.llm
        result = chain.invoke({"question": question})
        search_query = result.content.strip()
        
        state['search_query'] = search_query
        state['retry_count'] = 0
        
        print(f"🧭 Router: Search query = '{search_query}'")
        return state
    
    def retrieve_node(self, state: AgentState) -> AgentState:
        """
        Retriever: Performs semantic search in the vector store.
        """
        search_query = state.get('search_query', state['question'])
        
        # Retrieve documents
        docs = self.vector_store.similarity_search(
            query=search_query,
            k=self.config.get('top_k_results', 5)
        )
        
        state['retrieved_docs'] = docs
        
        print(f"📚 Retriever: Found {len(docs)} documents")
        if docs:
            print(f"   Top file: {docs[0].metadata.get('file_path', 'N/A')}")
        
        return state
    
    def grade_node(self, state: AgentState) -> AgentState:
        """
        Grader: LLM judges if retrieved docs are relevant to the question.
        """
        question = state['question']
        docs = state['retrieved_docs']
        
        if not docs:
            state['relevance_grade'] = 'bad'
            print("❌ Grader: No documents retrieved")
            return state
        
        # Prepare document summaries for grading
        doc_summaries = []
        for i, doc in enumerate(docs[:3], 1):  # Grade top 3 only
            summary = f"Doc {i} ({doc.metadata.get('file_path', 'unknown')}):\n{doc.page_content[:200]}"
            doc_summaries.append(summary)
        
        docs_text = "\n\n".join(doc_summaries)
        
        # Use LLM to grade relevance
        prompt = ChatPromptTemplate.from_template(
            """You are a grader evaluating code search results.

User Question: {question}

Retrieved Code:
{docs}

Are these code snippets relevant to answering the question?
- If YES (they contain relevant functions, classes, or logic): respond "good"
- If NO (completely unrelated): respond "bad"

Answer with just one word: good or bad"""
        )
        
        chain = prompt | self.llm
        result = chain.invoke({"question": question, "docs": docs_text})
        grade = result.content.strip().lower()
        
        state['relevance_grade'] = grade
        
        print(f"⚖️  Grader: {grade.upper()}")
        return state
    
    def rewrite_query_node(self, state: AgentState) -> AgentState:
        """
        Query Rewriter: Creates a better search query if previous results were bad.
        """
        question = state['question']
        old_query = state['search_query']
        
        prompt = ChatPromptTemplate.from_template(
            """You are a search query optimizer for code search.

Original Question: {question}
Previous Search Query: {old_query}

The previous query didn't find relevant code. Create a DIFFERENT search query.
Try:
- Different keywords (synonyms, related terms)
- More specific technical terms
- Broader concepts

New Search Query:"""
        )
        
        chain = prompt | self.llm
        result = chain.invoke({"question": question, "old_query": old_query})
        new_query = result.content.strip()
        
        state['search_query'] = new_query
        state['retry_count'] += 1
        
        print(f"🔄 Rewriting query (attempt {state['retry_count']}): '{new_query}'")
        return state
    
    def generate_node(self, state: AgentState) -> AgentState:
        """
        Generator: Creates the final answer with code snippets and file paths.
        """
        question = state['question']
        docs = state['retrieved_docs']
        
        if not docs:
            state['generation'] = "❌ Sorry, I couldn't find relevant code in the codebase. Try rephrasing your question or check if the code has been ingested."
            return state
        
        # Prepare context from retrieved docs
        context_parts = []
        for i, doc in enumerate(docs, 1):
            file_path = doc.metadata.get('file_path', 'unknown')
            language = doc.metadata.get('language', 'text')
            content = doc.page_content
            
            context_parts.append(
                f"**File {i}: `{file_path}`** (Language: {language})\n```{language}\n{content}\n```"
            )
        
        context = "\n\n".join(context_parts)
        
        # Generate answer
        prompt = ChatPromptTemplate.from_template(
            """You are an expert code analyst helping developers understand a codebase.

User Question: {question}

Relevant Code:
{context}

Your task:
1. Answer the question clearly and concisely
2. Reference specific file paths using backticks: `path/to/file.py`
3. Highlight key functions or classes
4. If the code shows implementation details, explain HOW it works
5. If multiple files are involved, explain the flow

Format your answer with:
- Brief summary (1-2 sentences)
- File locations
- Key code explanation
- Any important notes

Answer:"""
        )
        
        chain = prompt | self.llm
        result = chain.invoke({"question": question, "context": context})
        
        state['generation'] = result.content
        
        print("✅ Generated answer")
        return state
    
    def decide_after_grading(self, state: AgentState) -> str:
        """
        Decision function: What to do after grading?
        """
        grade = state['relevance_grade']
        retry_count = state['retry_count']
        
        if grade == 'good':
            return "generate"
        elif retry_count < self.max_retries:
            return "rewrite"
        else:
            print(f"⚠️  Max retries ({self.max_retries}) reached. Generating with current docs.")
            return "give_up"
    
    def query(self, question: str) -> dict:
        """
        Main entry point: Ask a question and get an answer.
        
        Returns:
            dict with 'answer' and 'sources'
        """
        print(f"\n❓ Question: {question}")
        print("=" * 60)
        
        # Run the graph
        initial_state = {
            'question': question,
            'retrieved_docs': [],
            'generation': '',
            'retry_count': 0,
            'search_query': '',
            'relevance_grade': '',
        }
        
        final_state = self.graph.invoke(initial_state)
        
        # Extract sources
        sources = []
        for doc in final_state['retrieved_docs']:
            sources.append({
                'file_path': doc.metadata.get('file_path', 'unknown'),
                'language': doc.metadata.get('language', 'text'),
                'content': doc.page_content,
            })
        
        return {
            'answer': final_state['generation'],
            'sources': sources,
        }


def test_agent():
    """Test the agent with sample queries."""
    print("🧪 Testing Codebase Agent")
    print("=" * 60)
    
    agent = CodebaseAgent()
    
    # Check if vector store has data
    stats = agent.vector_store.get_collection_stats()
    if stats.get('total_documents', 0) == 0:
        print("\n⚠️  No documents in vector store. Run ingestor.py first!")
        return
    
    print(f"📊 Vector Store: {stats.get('total_documents', 0)} documents\n")
    
    # Test queries
    test_questions = [
        "Where is the main entry point of the application?",
        "Show me the configuration files",
        "How is the database connection set up?",
    ]
    
    for question in test_questions:
        result = agent.query(question)
        
        print(f"\n💬 Answer:")
        print(result['answer'])
        print(f"\n📎 Sources: {len(result['sources'])} files")
        
        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        test_agent()
    else:
        print("Usage:")
        print("  python agent.py --test    # Test the agent")

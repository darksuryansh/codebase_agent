"""
Vector Store: FAISS setup with code-optimized embeddings.
Uses HuggingFace jina-embeddings-v2-base-code for semantic code search.
"""

import os
import pickle
from typing import List, Dict, Optional
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from utils import load_config


class VectorStore:
    """Manages FAISS vector store for code embeddings."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        
        # FAISS settings
        vector_config = self.config.get('vector_store', {})
        self.persist_directory = vector_config.get('persist_directory', './data/vector_db')
        self.index_name = vector_config.get('index_name', 'codebase_index')
        
        # Ensure directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize embeddings (HuggingFace code-specific model)
        self.embeddings = self._init_embeddings()
        
        # Initialize or load FAISS
        self.vectorstore = self._init_vectorstore()
    
    def _init_embeddings(self):
        """Initialize code-optimized embeddings."""
        embeddings_config = self.config.get('embeddings', {})
        model_name = embeddings_config.get('model', 'jinaai/jina-embeddings-v2-base-code')
        device = embeddings_config.get('device', 'cpu')
        
        print(f"📦 Loading embeddings model: {model_name}")
        
        # Using HuggingFace embeddings with code-specific model
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        return embeddings
    
    def _init_vectorstore(self):
        """Initialize or load FAISS vector store."""
        index_path = os.path.join(self.persist_directory, self.index_name)
        
        # Try to load existing index
        if os.path.exists(f"{index_path}.faiss"):
            print(f"📂 Loading existing FAISS index from {index_path}")
            try:
                vectorstore = FAISS.load_local(
                    index_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                return vectorstore
            except Exception as e:
                print(f"⚠️  Could not load existing index: {e}")
                print("Creating new index...")
        
        # Return None - will be created when documents are added
        return None
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store."""
        if not documents:
            print("⚠️  No documents to add")
            return
        
        print(f"💾 Adding {len(documents)} documents to FAISS...")
        
        # Create or update vectorstore
        if self.vectorstore is None:
            # Create new FAISS index
            self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        else:
            # Add to existing index
            self.vectorstore.add_documents(documents)
        
        # Save the index
        index_path = os.path.join(self.persist_directory, self.index_name)
        self.vectorstore.save_local(index_path)
        
        print("✅ Documents added and index saved successfully")
    
    def similarity_search(
        self,
        query: str,
        k: int = None,
        filter: Optional[Dict] = None
    ) -> List[Document]:
        """
        Perform similarity search.
        
        Args:
            query: Search query
            k: Number of results to return
            filter: Metadata filter (e.g., {"language": "python"})
        """
        if k is None:
            k = self.config.get('top_k_results', 5)
        
        results = self.vectorstore.similarity_search(
            query,
            k=k,
            filter=filter
        )
        
        return results
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = None,
        filter: Optional[Dict] = None
    ) -> List[tuple]:
        """
        Perform similarity search with relevance scores.
        
        Returns:
            List of (Document, score) tuples
        """
        if k is None:
            k = self.config.get('top_k_results', 5)
        
        results = self.vectorstore.similarity_search_with_score(
            query,
            k=k,
            filter=filter
        )
        
        return results
    
    def delete_collection(self) -> None:
        """Delete the entire index (use with caution!)."""
        print(f"⚠️  Deleting FAISS index")
        index_path = os.path.join(self.persist_directory, self.index_name)
        
        # Delete index files
        for ext in ['.faiss', '.pkl']:
            file_path = f"{index_path}{ext}"
            if os.path.exists(file_path):
                os.remove(file_path)
        
        self.vectorstore = None
        print("✅ Index deleted")
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the index."""
        try:
            if self.vectorstore is None:
                return {
                    'index_name': self.index_name,
                    'total_documents': 0,
                    'persist_directory': self.persist_directory,
                }
            
            # FAISS doesn't have a direct count, but we can estimate
            return {
                'index_name': self.index_name,
                'total_documents': self.vectorstore.index.ntotal if hasattr(self.vectorstore, 'index') else 0,
                'persist_directory': self.persist_directory,
            }
        except Exception as e:
            return {'error': str(e)}


def test_vector_store():
    """Test the vector store with sample queries."""
    print("🧪 Testing Vector Store")
    print("=" * 60)
    
    vs = VectorStore()
    
    # Get stats
    stats = vs.get_collection_stats()
    print(f"\n📊 Collection Stats:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    if stats.get('total_documents', 0) == 0:
        print("\n⚠️  No documents in collection. Run ingestor.py first!")
        return
    
    # Test queries
    test_queries = [
        "user authentication login",
        "database connection setup",
        "API endpoint routes",
    ]
    
    print(f"\n🔍 Testing Similarity Search:")
    for query in test_queries:
        print(f"\n   Query: '{query}'")
        results = vs.similarity_search(query, k=3)
        
        for i, doc in enumerate(results, 1):
            print(f"   Result {i}:")
            print(f"      File: {doc.metadata.get('file_path', 'N/A')}")
            print(f"      Language: {doc.metadata.get('language', 'N/A')}")
            print(f"      Preview: {doc.page_content[:100]}...")
    
    print("\n" + "=" * 60)
    print("✅ Vector Store Test Complete")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        test_vector_store()
    else:
        print("Usage:")
        print("  python vector_store.py --test    # Test the vector store")

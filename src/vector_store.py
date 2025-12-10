"""
Vector Store: ChromaDB setup with code-optimized embeddings.
Uses HuggingFace jina-embeddings-v2-base-code for semantic code search.
"""

import os
from typing import List, Dict, Optional
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb
from chromadb.config import Settings

from utils import load_config


class VectorStore:
    """Manages ChromaDB vector store for code embeddings."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        
        # ChromaDB settings
        chroma_config = self.config.get('chroma', {})
        self.persist_directory = chroma_config.get('persist_directory', './data/chroma_db')
        self.collection_name = chroma_config.get('collection_name', 'codebase_chunks')
        
        # Ensure directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize embeddings (HuggingFace code-specific model)
        self.embeddings = self._init_embeddings()
        
        # Initialize ChromaDB
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
        """Initialize ChromaDB vector store."""
        # Create ChromaDB client with persistence
        client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize LangChain Chroma wrapper
        vectorstore = Chroma(
            client=client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
        )
        
        return vectorstore
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store."""
        if not documents:
            print("⚠️  No documents to add")
            return
        
        print(f"💾 Adding {len(documents)} documents to ChromaDB...")
        
        # Add in batches to avoid memory issues
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            self.vectorstore.add_documents(batch)
            
            if (i + batch_size) % 500 == 0:
                print(f"   Processed {min(i + batch_size, len(documents))}/{len(documents)} documents")
        
        print("✅ Documents added successfully")
    
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
        """Delete the entire collection (use with caution!)."""
        print(f"⚠️  Deleting collection: {self.collection_name}")
        self.vectorstore.delete_collection()
        print("✅ Collection deleted")
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection."""
        try:
            collection = self.vectorstore._collection
            count = collection.count()
            
            return {
                'collection_name': self.collection_name,
                'total_documents': count,
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

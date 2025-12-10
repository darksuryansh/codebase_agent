"""
Utility script to manage vector store indexes.
Allows listing, viewing stats, and deleting indexes.
"""

import os
import shutil
import argparse
from pathlib import Path
from vector_store import VectorStore


def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


def list_indexes():
    """List all available vector store indexes."""
    vector_db_path = get_project_root() / 'data' / 'vector_db'
    
    if not vector_db_path.exists():
        print("❌ No vector database directory found!")
        return []
    
    indexes = [d.name for d in vector_db_path.iterdir() 
               if d.is_dir() and (d / 'index.faiss').exists()]
    
    if not indexes:
        print("📭 No indexes found. Ingest a repository first!")
        return []
    
    print(f"\n📚 Found {len(indexes)} index(es):\n")
    print("-" * 60)
    
    for idx_name in indexes:
        try:
            vs = VectorStore(index_name=idx_name)
            stats = vs.get_collection_stats()
            num_docs = stats.get('total_documents', 0)
            
            # Get index size
            index_dir = vector_db_path / idx_name
            size_mb = sum(f.stat().st_size for f in index_dir.rglob('*') if f.is_file()) / (1024 * 1024)
            
            print(f"📦 {idx_name}")
            print(f"   └─ Documents: {num_docs}")
            print(f"   └─ Size: {size_mb:.2f} MB")
            print()
        except Exception as e:
            print(f"⚠️  {idx_name}: Error loading ({e})")
            print()
    
    print("-" * 60)
    return indexes


def delete_index(index_name: str):
    """Delete a vector store index."""
    vector_db_path = get_project_root() / 'data' / 'vector_db' / index_name
    
    if not vector_db_path.exists():
        print(f"❌ Index '{index_name}' not found!")
        return
    
    confirm = input(f"⚠️  Are you sure you want to delete '{index_name}'? (yes/no): ")
    if confirm.lower() != 'yes':
        print("❌ Deletion cancelled.")
        return
    
    try:
        shutil.rmtree(vector_db_path)
        print(f"✅ Index '{index_name}' deleted successfully!")
    except Exception as e:
        print(f"❌ Error deleting index: {e}")


def main():
    parser = argparse.ArgumentParser(description="Manage vector store indexes")
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help="List all available indexes"
    )
    parser.add_argument(
        '--delete', '-d',
        type=str,
        metavar='INDEX_NAME',
        help="Delete a specific index"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_indexes()
    elif args.delete:
        delete_index(args.delete)
    else:
        # Default: list indexes
        list_indexes()
        print("\nℹ️  Usage:")
        print("  python src/manage_indexes.py --list")
        print("  python src/manage_indexes.py --delete <index_name>")


if __name__ == "__main__":
    main()

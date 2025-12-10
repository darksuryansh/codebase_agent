"""
Code Ingestor: Loads codebase and performs syntax-aware splitting.
This module respects code structure (functions, classes) when splitting.
"""

import os
import argparse
import shutil
import tempfile
from pathlib import Path
from typing import List
from tqdm import tqdm
from git import Repo
from urllib.parse import urlparse

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader

from utils import (
    load_config,
    should_ignore_file,
    get_language_from_extension,
    extract_file_metadata,
)
from vector_store import VectorStore


class CodebaseIngestor:
    """Handles loading and splitting code files from a repository."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.chunk_size = self.config.get('chunk_size', 1500)
        self.chunk_overlap = self.config.get('chunk_overlap', 200)
        self.ignore_patterns = self.config.get('ignore_patterns', [])
        self.allowed_extensions = self.config.get('allowed_extensions', [])
        self.temp_clone_dir = None
    
    def is_git_url(self, path: str) -> bool:
        """Check if the path is a Git repository URL."""
        parsed = urlparse(path)
        return parsed.scheme in ('http', 'https', 'git') or path.endswith('.git')
    
    def get_repo_name_from_url(self, repo_url: str) -> str:
        """Extract repository name from Git URL."""
        parsed = urlparse(repo_url)
        path = parsed.path.rstrip('/')
        # Extract repo name (last part of path, remove .git)
        repo_name = path.split('/')[-1].replace('.git', '')
        return repo_name
    
    def clone_repository(self, repo_url: str) -> Path:
        """Clone a Git repository to a temporary directory."""
        print(f"📥 Cloning repository from: {repo_url}")
        
        # Create temp directory
        self.temp_clone_dir = tempfile.mkdtemp(prefix='codebase_agent_')
        temp_path = Path(self.temp_clone_dir)
        
        try:
            # Clone the repository
            Repo.clone_from(repo_url, temp_path, depth=1)  # Shallow clone for speed
            print(f"✅ Repository cloned to: {temp_path}")
            return temp_path
        except Exception as e:
            print(f"❌ Error cloning repository: {e}")
            if self.temp_clone_dir and os.path.exists(self.temp_clone_dir):
                shutil.rmtree(self.temp_clone_dir)
            raise
    
    def cleanup_temp_clone(self):
        """Clean up temporary cloned repository."""
        if self.temp_clone_dir and os.path.exists(self.temp_clone_dir):
            print(f"🧹 Cleaning up temporary clone...")
            try:
                # On Windows, need to handle read-only files
                def handle_remove_readonly(func, path, exc):
                    """Error handler for Windows readonly file removal."""
                    import stat
                    if not os.access(path, os.W_OK):
                        os.chmod(path, stat.S_IWUSR)
                        func(path)
                    else:
                        raise
                
                shutil.rmtree(self.temp_clone_dir, onerror=handle_remove_readonly)
                print("✅ Cleanup complete")
            except Exception as e:
                print(f"⚠️  Could not fully clean up temp directory: {e}")
                print(f"   You may manually delete: {self.temp_clone_dir}")
            finally:
                self.temp_clone_dir = None
        
    def get_splitter_for_language(self, language: str) -> RecursiveCharacterTextSplitter:
        #Get a syntax-aware text splitter for the given language.
        from langchain_text_splitters import Language
        
        # Map our language names to LangChain's enum
        lang_map = {
            'python': Language.PYTHON,
            'javascript': Language.JS,
            'typescript': Language.TS,
            'java': Language.JAVA,
            'cpp': Language.CPP,
            'go': Language.GO,
            'rust': Language.RUST,
            'ruby': Language.RUBY,
            'php': Language.PHP,
            'swift': Language.SWIFT,
            'kotlin': Language.KOTLIN,
            'scala': Language.SCALA,
            'markdown': Language.MARKDOWN,
        }
        
        chunk_size = self.config.get('language_chunk_sizes', {}).get(
            language, self.chunk_size
        )
        
        if language in lang_map:
            return RecursiveCharacterTextSplitter.from_language(
                language=lang_map[language],
                chunk_size=chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        else:
            # Fallback to generic splitter
            return RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
    
    def scan_repository(self, repo_path: str) -> List[Path]:
        """Scan repository and return list of code files to process."""
        repo_path = Path(repo_path)
        code_files = []
        
        print(f"📂 Scanning repository: {repo_path}")
        
        for file_path in repo_path.rglob('*'):
            if not file_path.is_file():
                continue
            
            # Check if file should be ignored
            if should_ignore_file(file_path, self.ignore_patterns):
                continue
            
            # Check if file extension is allowed
            if self.allowed_extensions and file_path.suffix not in self.allowed_extensions:
                continue
            
            code_files.append(file_path)
        
        print(f"✅ Found {len(code_files)} code files")
        return code_files
    
    def load_and_split_file(self, file_path: Path, repo_root: str) -> List[Document]:
        """Load a single file and split it syntax-aware."""
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            if not content.strip():
                return []
            
            # Determine language
            language = get_language_from_extension(str(file_path))
            
            # Get appropriate splitter
            splitter = self.get_splitter_for_language(language)
            
            # Split the content
            texts = splitter.split_text(content)
            
            # Create documents with metadata
            documents = []
            relative_path = str(file_path.relative_to(repo_root))
            
            for i, text in enumerate(texts):
                metadata = {
                    'file_path': relative_path,
                    'language': language,
                    'file_name': file_path.name,
                    'chunk_index': i,
                    'total_chunks': len(texts),
                }
                
                # Add function/class names if Python
                if language == 'python':
                    import re
                    functions = re.findall(r'^\s*def\s+(\w+)', text, re.MULTILINE)
                    classes = re.findall(r'^\s*class\s+(\w+)', text, re.MULTILINE)
                    if functions:
                        metadata['functions'] = ','.join(functions[:5])
                    if classes:
                        metadata['classes'] = ','.join(classes[:5])
                
                doc = Document(page_content=text, metadata=metadata)
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            print(f"⚠️  Error processing {file_path}: {e}")
            return []
    
    def ingest_repository(self, repo_path: str) -> List[Document]:
        """Ingest entire repository and return all document chunks."""
        repo_path = Path(repo_path).resolve()
        code_files = self.scan_repository(repo_path)
        
        all_documents = []
        
        print(f"\n📝 Processing files...")
        for file_path in tqdm(code_files):
            documents = self.load_and_split_file(file_path, str(repo_path))
            all_documents.extend(documents)
        
        print(f"\n✅ Created {len(all_documents)} document chunks")
        return all_documents


def main():
    parser = argparse.ArgumentParser(
        description="Ingest a codebase into the vector store from a local path or Git URL"
    )
    parser.add_argument(
        '--repo_path',
        type=str,
        required=True,
        help='Path to the repository OR Git URL (e.g., https://github.com/user/repo.git)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to config file'
    )
    
    args = parser.parse_args()
    
    print("🚀 Starting Codebase Ingestion")
    print("=" * 60)
    
    # Initialize ingestor
    ingestor = CodebaseIngestor(config_path=args.config)
    
    try:
        # Check if it's a Git URL or local path
        if ingestor.is_git_url(args.repo_path):
            repo_path = ingestor.clone_repository(args.repo_path)
            repo_name = ingestor.get_repo_name_from_url(args.repo_path)
        else:
            repo_path = args.repo_path
            if not os.path.exists(repo_path):
                print(f"❌ Local path not found: {repo_path}")
                return
            repo_name = Path(repo_path).name
        
        # Ingest repository
        documents = ingestor.ingest_repository(str(repo_path))
        
        if not documents:
            print("❌ No documents to ingest!")
            return
        
        # Initialize vector store with repo-specific index
        print(f"\n💾 Storing in Vector Database (Index: {repo_name})...")
        vector_store = VectorStore(config_path=args.config, index_name=repo_name)
        vector_store.add_documents(documents)
        
        print("\n" + "=" * 60)
        print("✅ Ingestion Complete!")
        print(f"📊 Total chunks stored: {len(documents)}")
        print(f"📁 Repository: {repo_name}")
        print(f"💾 Vector DB Index: {repo_name}")
        
        # Print sample
        print("\n📝 Sample chunk:")
        print("-" * 60)
        sample = documents[0]
        print(f"File: {sample.metadata['file_path']}")
        print(f"Language: {sample.metadata['language']}")
        print(f"Content preview:\n{sample.page_content[:300]}...")
    
    finally:
        # Cleanup temporary clone if it exists
        ingestor.cleanup_temp_clone()


if __name__ == "__main__":
    main()

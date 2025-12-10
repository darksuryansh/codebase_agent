"""
Utility functions for the Codebase Onboarding Agent.
Includes: file tree generator, metadata extraction, and helpers.
"""

import os
from pathlib import Path
from typing import List, Dict
import yaml


def load_config(config_path: str = "config/config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def should_ignore_file(file_path: Path, ignore_patterns: List[str]) -> bool:
    """Check if a file should be ignored based on patterns."""
    file_str = str(file_path)
    
    for pattern in ignore_patterns:
        pattern = pattern.replace('**', '*')
        if pattern.endswith('/**'):
            # Directory pattern
            dir_pattern = pattern[:-3]
            if dir_pattern in file_str:
                return True
        elif pattern.startswith('*.'):
            # Extension pattern
            if file_str.endswith(pattern[1:]):
                return True
        elif pattern in file_str:
            return True
    
    return False


def get_language_from_extension(file_path: str) -> str:
    """Determine programming language from file extension."""
    ext_to_lang = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.jsx': 'javascript',
        '.tsx': 'typescript',
        '.java': 'java',
        '.cpp': 'cpp',
        '.c': 'c',
        '.h': 'cpp',
        '.hpp': 'cpp',
        '.cs': 'csharp',
        '.go': 'go',
        '.rs': 'rust',
        '.rb': 'ruby',
        '.php': 'php',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.scala': 'scala',
        '.md': 'markdown',
    }
    
    ext = Path(file_path).suffix.lower()
    return ext_to_lang.get(ext, 'text')


def generate_repo_tree(repo_path: str, max_depth: int = 3, ignore_patterns: List[str] = None) -> str:
    """
    Generate a text-based tree structure of the repository.
    This gives the AI a "map" of the codebase.
    """
    if ignore_patterns is None:
        config = load_config()
        ignore_patterns = config.get('ignore_patterns', [])
    
    tree_lines = [f"Repository Structure: {Path(repo_path).name}/"]
    
    def add_directory(path: Path, prefix: str = "", depth: int = 0):
        if depth > max_depth:
            return
        
        try:
            items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
        except PermissionError:
            return
        
        for i, item in enumerate(items):
            if should_ignore_file(item, ignore_patterns):
                continue
            
            is_last = i == len(items) - 1
            current_prefix = "└── " if is_last else "├── "
            next_prefix = "    " if is_last else "│   "
            
            tree_lines.append(f"{prefix}{current_prefix}{item.name}{'/' if item.is_dir() else ''}")
            
            if item.is_dir():
                add_directory(item, prefix + next_prefix, depth + 1)
    
    add_directory(Path(repo_path))
    return "\n".join(tree_lines)


def extract_file_metadata(file_path: str, content: str) -> Dict:
    """Extract metadata from a code file."""
    language = get_language_from_extension(file_path)
    
    metadata = {
        'file_path': file_path,
        'language': language,
        'file_name': Path(file_path).name,
        'file_size': len(content),
        'extension': Path(file_path).suffix,
    }
    
    # Extract simple patterns (function/class names) for Python
    if language == 'python':
        import re
        functions = re.findall(r'^\s*def\s+(\w+)', content, re.MULTILINE)
        classes = re.findall(r'^\s*class\s+(\w+)', content, re.MULTILINE)
        metadata['functions'] = functions[:10]  # Limit to first 10
        metadata['classes'] = classes[:10]
    
    return metadata


def format_code_with_line_numbers(code: str, start_line: int = 1) -> str:
    """Add line numbers to code for better readability."""
    lines = code.split('\n')
    numbered_lines = [f"{start_line + i:4d} | {line}" for i, line in enumerate(lines)]
    return '\n'.join(numbered_lines)


def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate text with ellipsis if too long."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Utility functions for codebase agent")
    parser.add_argument('--generate-tree', type=str, help='Generate tree for repo path')
    parser.add_argument('--max-depth', type=int, default=3, help='Max depth for tree')
    
    args = parser.parse_args()
    
    if args.generate_tree:
        tree = generate_repo_tree(args.generate_tree, max_depth=args.max_depth)
        print(tree)
        
        # Save to file
        output_path = "data/repo_tree.txt"
        os.makedirs("data", exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(tree)
        print(f"\n✅ Tree saved to {output_path}")

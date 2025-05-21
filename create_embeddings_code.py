from transformers import AutoTokenizer, AutoModel
import os
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np
from pathlib import Path
import re
import openai
from groq import Groq
# from openai import RateLimitError, APIError, Timeout
from tqdm import tqdm
import time
import tiktoken
import ast
import hashlib
import json
import concurrent.futures

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" 
INDEX_DIR = "/code analyzer/vector_db"
SUMMARY_CACHE = "summary_cache.json"
UPLOADED_SUMMARY_CACHE = "summary_cache.json"
GROQ_MODEL = "llama-3.3-70b-versatile"
MAX_RECENT_MESSAGES = 2
IGNORED_DIRS = {'repos' , 'node_modules', 'venv','myenv', 'env', 'dist', 'build', '.git', '__pycache__' , '.github' , 'lib', 'bin', 'include', 'share', 'tests', 'test' , '.idea' , '.vscode' , '.pytest_cache' , '.mypy_cache' , '.coverage' , '.tox' , '.eggs' , '.hypothesis' , '.pytest' }
IGNORED_FILES = {'.gitignore', 'package-lock.json'}
TARGET_EXTENSION = '.py'
MAX_CHUNK_TOKENS = 6000
MAX_FILE_TOKENS = 15000


embedder = SentenceTransformer(MODEL_NAME)
api_keys = ['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',
            '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' , 
            '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', #mu
            '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', #godz.07
            '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', #shubhankar
           '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' ,
            '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',
            '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',
            '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',
            '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',#arnav
            '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'  #shreyans
           ]
summary_cache = {}

ENCODING = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    cnt = len(ENCODING.encode(text))
    return cnt

def is_ignored(path):
    return any(ignored in path for ignored in IGNORED_DIRS)

def extract_code_chunks_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        source = f.read()
    
    chunks = []
    lines_used = set()
    lines = source.splitlines()

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return chunks

    import_lines = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            start = node.lineno - 1
            end = getattr(node, "end_lineno", node.lineno)
            for i in range(start, end):
                lines_used.add(i)
                import_lines.append(lines[i])

    if import_lines:
        chunks.append({
            "type": "import",
            "name": "imports",
            "source": "\n".join(import_lines),
            "file": file_path
        })

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            try:
                start_line = node.lineno - 1
                end_line = node.end_lineno
                code_lines = lines[start_line:end_line]
                for i in range(start_line, end_line):
                    lines_used.add(i)
                chunks.append({
                    "type": type(node).__name__,
                    "name": node.name,
                    "source": "\n".join(code_lines),
                    "file": file_path
                })
            except AttributeError:
                continue  

    remaining_lines = [lines[i] for i in range(len(lines)) if i not in lines_used and lines[i].strip()]
    if remaining_lines:
        chunks.append({
            "type": "other",
            "name": "top_level_code",
            "source": "\n".join(remaining_lines),
            "file": file_path
        })

    return chunks

def extract_chunks(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        source = f.read()

    if len(source) > 100000:
        return None

    chunks = []

    import_pattern = r'^(import\s.+?;[ \t]*$|import\s.+?from\s+[\'\"].+?[\'\"];?[ \t]*$)'
    imports = re.findall(import_pattern, source, re.MULTILINE)
    import_code = '\n'.join(imports).strip()

    used_lines = set()
    source_lines = source.splitlines()

    for i, line in enumerate(source_lines):
        for imp in imports:
            if line.strip() == imp.strip():
                used_lines.add(i)

    if import_code:
        chunks.append({
            "type": "frontend",
            "name": "imports",
            "source": import_code,
            "file": file_path
        })

    source_wo_imports = '\n'.join(
        line for i, line in enumerate(source_lines) if i not in used_lines
    )

    pattern = re.compile(
        r'^(?P<full>'
        r'(?P<export>export\s+)?(?P<async>async\s+)?function\s+(?P<funcname>\w+)'       
        r'|export\s+default\s+function\s+(?P<defaultname>\w+)?'                         
        r'|class\s+(?P<classname>\w+)'                                                 
        r'|(?P<var_decl>(const|let|var)\s+(?P<varname>\w+)\s*=\s*(async\s+)?(\([^)]*\)|\w+)\s*=>)' 
        r'|(?P<anonfunc>(const|let|var)\s+(?P<anonname>\w+)\s*=\s*function)'          
        r')',
        re.MULTILINE
    )

    matches = list(pattern.finditer(source_wo_imports))
    positions = [match.start() for match in matches]
    positions.append(len(source_wo_imports))

    used_ranges = []

    for i, match in enumerate(matches):
        start = positions[i]
        end = positions[i + 1]
        chunk_code = source_wo_imports[start:end].strip()
        if not chunk_code:
            continue

        name = (
            match.group("funcname") or
            match.group("defaultname") or
            match.group("classname") or
            match.group("varname") or
            match.group("anonname") or
            "anonymous"
        )

        used_ranges.append((start, end))
        chunks.append({
            "type": "frontend",
            "name": name,
            "source": chunk_code,
            "file": file_path
        })

    covered = [False] * len(source_wo_imports)
    for start, end in used_ranges:
        for i in range(start, end):
            if i < len(covered):
                covered[i] = True

    leftover = ''.join(
        ch if not covered[i] else ''
        for i, ch in enumerate(source_wo_imports)
    ).strip()

    if leftover:
        chunks.append({
            "type": "frontend",
            "name": "unmatched",
            "source": leftover,
            "file": file_path
        })

    return chunks



def extract_chunks_md(file_path) :
    with open(file_path, 'r', encoding='utf-8') as f:
        source = f.read()
    if len(source) > 10000 :
        return None
    chunks = []
    chunk_size = 1000
    i = 0
    while i < len(source):
        chunks.append({
            "type" : 'readme',
            'name' : 'documentation' ,
            'source' :  source[i:i + chunk_size],
            'file' : file_path
            })
        i += chunk_size
    return chunks

def get_semantic_chunks_from_repo(repo_path):
    all_chunks = []
    for root, _, files in os.walk(repo_path):
        
        if is_ignored(root):
            continue
        for file in files:
            try :
                file_path_temp = os.path.join(root, file)
                if file in IGNORED_FILES or 'bootstrap' in file or 'fonts' in file_path_temp:
                    continue
                elif file.endswith('.js') or file.endswith('.tsx') or file.endswith('.jsx'):
                    file_path = os.path.join(root, file)
                    chunks = extract_chunks(file_path)
                    if chunks:
                        all_chunks.extend(chunks)
                    continue
                elif file.endswith('.md') or file.endswith('.txt') :
                    file_path = os.path.join(root, file)
                    chunks = extract_chunks_md(file_path)
                    if chunks:
                        all_chunks.extend(chunks)
                    continue
                elif not file.endswith('.py'):
                    continue
                    
                file_path = os.path.join(root, file)
                chunks = extract_code_chunks_from_file(file_path)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
    return all_chunks


def build_vector_db(code_blocks):
    codes = []
    paths = []
    name = []
    summaries = []
    # print(name)
    
    for block in code_blocks :
        codes.append(block['source'])
        paths.append(block['file'])
        name.append(block['name'])
        summaries.append('')
    
    vectors = embedder.encode(codes , show_progress_bar=True)
    
    index = faiss.IndexFlatL2(len(vectors[0]))
    index.add(vectors)

    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, f"{INDEX_DIR}/code_index.faiss")

    with open(f"{INDEX_DIR}/metadata.json", "w") as f:
        json.dump(list(zip(summaries, paths, codes , name)), f)

def run_pipeline():
    repo_dir = input("Enter the path to the repository: ")
    code_blocks = get_semantic_chunks_from_repo(repo_dir)
    print(len(code_blocks))
    build_vector_db(code_blocks)
    
if os.path.exists(UPLOADED_SUMMARY_CACHE):
    with open(UPLOADED_SUMMARY_CACHE, "r") as f:
        summary_cache = json.load(f)
        
count = 0
api_key_index = 7
run_pipeline()
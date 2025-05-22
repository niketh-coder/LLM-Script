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
# from google import genai
from build_call_graph import build_call_graph

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" 
INDEX_DIR = "/code analyzer/vector_db"
SUMMARY_CACHE = "updated_summary_cache.json"
UPLOADED_SUMMARY_CACHE = "summary_cache.json"
# LLM_MODEL = "gpt-4-turbo"
LLM_MODEL = "llama-3.3-70b-versatile"
MAX_RECENT_MESSAGES = 2
IGNORED_DIRS = {'repos' , 'node_modules', 'venv','myenv', 'env', 'dist', 'build', '.git', '__pycache__' , '.github' , 'lib', 'bin', 'include', 'share', 'tests', 'test' , '.idea' , '.vscode' , '.pytest_cache' , '.mypy_cache' , '.coverage' , '.tox' , '.eggs' , '.hypothesis' , '.pytest' }
IGNORED_FILES = {'.gitignore', 'package-lock.json'}
TARGET_EXTENSION = '.py'
CALL_GRAPH_FILE = "call_graph.json"
FILE_SUMMARY_PATH = "file_summary.json"
MAX_CHUNK_TOKENS = 100000
MAX_FILE_TOKENS = 500000


embedder = SentenceTransformer(MODEL_NAME)
api_keys = ['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', #mu
            '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', #godz.07
            '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', #shubhankar
           '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' ,
            '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',
            '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',
            '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',
            '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',#arnav
            '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']
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

    if len(source) > 1000000:
        return None

    chunks = []

    import_pattern = r'^\s*import\s.+?;?\s*$'
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
        r'(<[^>{}()\n]*)?'                                                                
        r'\s*\([^)]*\)\s*(:\s*[\w\[\]<>|]+)?'                                              
        r'|export\s+default\s+function\s+(?P<defaultname>\w+)?'                            
        r'|class\s+(?P<classname>\w+)'                                                     
        r'(\s+extends\s+\w+)?(\s+implements\s+[\w<>,\s]+)?'                               
        r'|(?P<var_decl>(const|let|var)\s+(?P<varname>\w+)\s*'                             
        r'(:\s*[^=]+)?\s*=\s*(async\s+)?(\([^)]*\)|\w+)\s*=>)'                             
        r'|(?P<anonfunc>(const|let|var)\s+(?P<anonname>\w+)\s*'                           
        r'(:\s*[^=]+)?\s*=\s*function)'                                                    
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
    if len(source) > 100000 :
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

        
def chunk_code(code, encoding, max_tokens=MAX_CHUNK_TOKENS):
    lines = code.split('\n')
    chunks, current_chunk = [], ""
    for line in lines:
        test_chunk = current_chunk + line + '\n'
        if len(encoding.encode(test_chunk)) > max_tokens:
            chunks.append(current_chunk)
            current_chunk = line + '\n'
        else:
            current_chunk = test_chunk
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def summarize_chunk(prompt, file):
    global api_key_index
    for attempt in range(3):  
        try:
            client = genai.Client(api_key= api_keys[api_key_index])
            
            print(f"Summarizing {file} using key index {api_key_index}")
            response = client.models.generate_content(
                model=LLM_MODEL, contents=prompt
            )
            
            return response.text
            # print(f"Summarising {file} using api key{api_keys[api_key_index]}")
            # groq_client = Groq(api_key= api_keys[api_key_index])
            # response = groq_client.chat.completions.create(
            #     model= LLM_MODEL,
            #     messages=[{"role": "user", "content": prompt}],
            # )
            # return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"Error in summarzing : {e}")
            if len(api_keys) > 1:
                api_key_index = (api_key_index + 1) % len(api_keys)
                print(f"Switching to API key index {api_key_index}")
                time.sleep(5)
            else:
                wait_time = 10 * (attempt + 1)
                print(f"Rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)

    print(f"[FAILED] {file}")
    return "Failed to summarize chunk."

def summarize_code(block, encoding):
    global count
    code = block["source"]
    code_hash = hashlib.md5(code.encode()).hexdigest()
    if code_hash in summary_cache:
        updated_summary_cache[code_hash] = summary_cache[code_hash]
        return summary_cache[code_hash], block["source"], block["file"], block['name']

    tokens = encoding.encode(code)
    if len(tokens) > MAX_FILE_TOKENS:
        return None 
    count += len(tokens)

    if len(tokens) > MAX_CHUNK_TOKENS:
        chunks = chunk_code(code, encoding)
        summaries = []
        for i, chunk in enumerate(tqdm(chunks, desc=f"Summarizing chunks in {block['file']}")):
            prompt = (
                f"Summarize this part of the code (chunk {i+1}/{len(chunks)}) for semantic search. "
                f"Mention component name or function, API endpoints if any, and what the code does. "
                f"File: {block['file']}:\n\n{chunk}"
            )
            summary = summarize_chunk(prompt , block['file'])
            summaries.append(summary)
        full_summary = "\n\n".join(summaries)
    else:
        prompt = (
            f"Summarize the following code for semantic search. "
            f"Mention the component name or function, API endpoints if any, and what the code does. "
            f"File: {block['file']}:\n\n{code}"
        )
        full_summary =  summarize_chunk( prompt , block['file'] )

    updated_summary_cache[code_hash] = full_summary
    return full_summary, block["source"], block["file"], block['name']
    # return None

def summarize_all(code_blocks):
    encoding = tiktoken.get_encoding("cl100k_base")
    
    def process_block(block):
        try:
            return summarize_code(block, encoding)
        except Exception as e:
            print(f"Error summarizing {block.get('file', '<unknown>')}: {e}")
            return None

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_block, code_blocks), total=len(code_blocks), desc="Summarizing Files"))

    return [r for r in results if r is not None]

def build_vector_db(summaries_data):
    with open(FILE_SUMMARY_PATH , "r") as f:
        file_summaries = json.load(f)
    
    summaries, codes, paths, names = zip(*[d for d in summaries_data if d[0]])

    combined_texts = []
    for summary, code, path in zip(summaries, codes, paths):
        file_summary = file_summaries.get(path, "")
        combined = f"File Summary: {file_summary}\n\nFunction/Class Summary: {summary}\n\nCode:\n{code}"
        combined_texts.append(combined)

    
    vectors = embedder.encode(combined_texts)
    index = faiss.IndexFlatL2(len(vectors[0]))
    index.add(vectors)

    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, f"{INDEX_DIR}/code_index.faiss")

    with open(f"{INDEX_DIR}/metadata.json", "w") as f:
        json.dump(list(zip(summaries, paths, codes, names)), f)

    with open(SUMMARY_CACHE, "w") as f:
        json.dump(updated_summary_cache, f)



def run_pipeline():
    repo_dir = input("Enter the path to the repository: ")
    build_call_graph(repo_dir, CALL_GRAPH_FILE , FILE_SUMMARY_PATH , IGNORED_DIRS , IGNORED_FILES , LLM_MODEL , api_keys[api_key_index])
    code_blocks = get_semantic_chunks_from_repo(repo_dir)
    print(len(code_blocks))
    summaries_data = summarize_all(code_blocks )
    build_vector_db(summaries_data)
    
if os.path.exists(UPLOADED_SUMMARY_CACHE):
    with open(UPLOADED_SUMMARY_CACHE, "r") as f:
        summary_cache = json.load(f)
        
updated_summary_cache = {}
        
count = 0
api_key_index = 4
run_pipeline()
import re
import os
import ast
import faiss
import json
import time
import hashlib
import tiktoken
from groq import Groq
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from openai import OpenAI

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_SUMMARY_MODEL = os.getenv("OPENAI_SUMMARY_MODEL")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL")
# GROQ_MODEL = "llama-3.3-70b-versatile"
IGNORED_DIRS = {'node_modules', 'venv', 'env', 'dist', 'build', '.git', '__pycache__'}
IGNORED_FILES = {'.gitignore', 'package-lock.json'}
SUMMARY_CACHE = "./summary_cache.json"
BASE_DIR = "./Build-it-master"
ENCODING = tiktoken.get_encoding("cl100k_base")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# client = Groq(
#         api_key= GROQ_API_KEY
# )
client = OpenAI(
        api_key=OPENAI_API_KEY
)
MAX_CONTEXT_TOKENS = 4000
system_prompt = """You are an expert developer helping to analyze a React-Django full-stack project."""
summary_cache = {}
count = 0

def build_vector_db(summaries_data):
    summaries, codes, paths , name = zip(*[d for d in summaries_data if d[0]])
    # print(name)
    vectors = embedder.encode(summaries)
    
    index = faiss.IndexFlatL2(len(vectors[0]))
    index.add(vectors)

    os.makedirs(BASE_DIR, exist_ok=True)
    faiss.write_index(index, f"{BASE_DIR}/code_index.faiss")

    with open(f"{BASE_DIR}/metadata.json", "w") as f:
        json.dump(list(zip(summaries, paths, codes , name)), f)

    with open(SUMMARY_CACHE, "w") as f:
        json.dump(summary_cache, f)

def convert_chunks_to_context_format(chunks):
    context_chunks = []
    for chunk in chunks:
        if len(chunk) >= 4:
            summary, file, source, name = chunk
            context_chunks.append({
                "name": name,
                "file": file,
                "source": source
            })
    return context_chunks

def convert_referenced_items(chunks):
    referenced_chunks = []
    for chunk in chunks :
        summary, file, source, name = chunk
        referenced_chunks.append(
            {
                "name": name,
                "file": file,
            }
        )
    return referenced_chunks

def merge_context_chunks(chunks , complete_cnt = 20):
    full = convert_chunks_to_context_format(chunks[:complete_cnt])
    refs = convert_referenced_items(chunks[complete_cnt : ])

    return full + refs

def search(query, top_k=3):
    index = faiss.read_index(f"{BASE_DIR}/code_index.faiss")
    with open(f"{BASE_DIR}/metadata.json", "r") as f:
        metadata = json.load(f)

    q_vec = embedder.encode([query])
    _, I = index.search(q_vec, top_k)
    from chat_workflow import get_used_chunks
    used_chunks = get_used_chunks()
    for id in I[0][:20] :
        if id not in used_chunks :
            used_chunks.add(id)
            
    return [metadata[i] for i in I[0]]

def chunk_context_by_token_limit(context_chunks, max_tokens=4000):
    batches = []
    current_batch = []
    current_tokens = 0
    for chunk in context_chunks:
        # print(chunk)
        global count
        chunk_tokens = count_tokens(chunk['file']) + count_tokens(chunk['name'])
        if 'source' in chunk.keys():
            chunk_tokens += count_tokens(chunk['source'])

        count += chunk_tokens
        if current_tokens + chunk_tokens > max_tokens:
            if current_batch:
                batches.append(current_batch)
            current_batch = [chunk]
            current_tokens = chunk_tokens
        else:
            current_batch.append(chunk)
            current_tokens += chunk_tokens

    if current_batch:
        batches.append(current_batch)

    return batches

def extract_json_from_response(text):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    return json.loads(text)

def ask_llm_to_refine_context_batch(query, chunk_batch):
    system_prompt = """You are an expert developer helping to analyze a React-Django full-stack project.
    Your task is to:
    1. From the provided code context, extract all the code relevant to the query solution.
    2. Do not remove the given context chunks too much.
    3. Identify function/class names that are needed to understand the code but are not included yet. These could be:
    - Functions/classes called or instantiated in the context.
    - Functions/classes mentioned but their code is not given.

    Respond strictly in the following JSON format without any other sentences:
    {
    "relevant_chunks": [
        {
        "name": "...",
        "file": "...",
        "trimmed_source": "..."
        },
        ...
    ],
    "needed_definitions": [
        { "name": "...", "file": "..." },
        ...
    ]
    }
    Be conservative — only include names in needed definitions if you are confident are required."""

    context_input = {
        "query": query,
        "context/extra chunks": chunk_batch
    }

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(context_input, indent=2)}
    ]

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0
    )
    # print(chunk_batch)
    # print('================================')
    print(response.choices[0].message.content)
    text = extract_json_from_response(response.choices[0].message.content)
    return text


def process_all_batches(query, context_chunks, token_limit=4000):
    batches = chunk_context_by_token_limit(context_chunks, max_tokens=token_limit)

    all_relevant = []
    all_needed = []

    for i, batch in enumerate(batches):
        print(f"Processing batch {i+1}/{len(batches)}...")
        result = ask_llm_to_refine_context_batch(query, batch)
        all_relevant.extend(result.get("relevant_chunks", []))
        all_needed.extend(result.get("needed_definitions", []))

    return {
        "relevant_chunks": all_relevant,
        "needed_definitions": all_needed
    }

def get_needed_functions(name , file) :
    global metadata
    from chat_workflow import get_used_chunks
    used_chunks = get_used_chunks()
    functions = []
    for i in range(len(metadata)) :
        if i in used_chunks :
            continue
        if (metadata[i][1] == file or metadata[i][1] == 'unknown') :
            if name in metadata[i][2] and name != 'imports' :
                functions.append({
                    'name': name , 
                    'file' : file ,
                    'source' : metadata[i][2]
                })
                used_chunks.add(i)

    return functions

def merge_chunks_to_string(relevant_chunks):
    merged = []
    for chunk in relevant_chunks:
        name = chunk.get("name", "unknown")
        file = chunk.get("file", "unknown")
        source = chunk.get("trimmed_source", "").strip()
        merged.append(f"// File: {file} | Chunk: {name}\n{source}\n")
    return "\n".join(merged)

def iterative_context_expansion(query, context_chunks, max_rounds=1):
    final_chunks = []
    for round_no in range(max_rounds):
        result = process_all_batches(query, context_chunks , token_limit=4000)

        final_chunks.extend(result["relevant_chunks"])
        needed_chunks = result['needed_definitions']

        if not len(needed_chunks) or round_no == max_rounds - 1:
            break

        new_chunks = []
        for chunk in needed_chunks :
            functions = get_needed_functions(chunk['name'] , chunk['file'])
            new_chunks.extend(functions)

        context_chunks = new_chunks

    context = merge_chunks_to_string(final_chunks)
    
    return context

def count_tokens(text):
    cnt = len(ENCODING.encode(text))
    return cnt

def chunk_context(text, max_tokens=MAX_CONTEXT_TOKENS):
    print(f'chunking {count_tokens(text)}')
    lines = text.split('\n')
    chunks, current_chunk, token_count = [], [], 0

    for line in lines:
        line_tokens = count_tokens(line)
        if token_count + line_tokens > max_tokens - 200:
            chunks.append('\n'.join(current_chunk))
            current_chunk, token_count = [], 0
            print("New chunk" ,count_tokens(chunks[-1]))
        current_chunk.append(line)
        token_count += line_tokens

    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    return chunks

def recursive_qa(context, query, ask_fn, max_tokens=MAX_CONTEXT_TOKENS):
    if count_tokens(context) <= max_tokens:
        return ask_fn(context, query)

    chunks = chunk_context(context, max_tokens)
    answers = [recursive_qa(chunk, query, ask_fn, max_tokens) for chunk in chunks]

    combined = "\n\n".join(answers)
    print('combined : ' , count_tokens(combined))
    return recursive_qa(combined, query , ask_fn, max_tokens)

def ask_fn(context, query):
    user_prompt = f"""Here is the user question: "{query}"
    
    Relevant code context:
    {context}
    """

    client = OpenAI(
        api_key=OPENAI_API_KEY
    )
    
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
        model="gpt-4-turbo", 
    )
    
    return chat_completion.choices[0].message.content

def summarize_chunk(prompt, file ):
    for attempt in range(3):
        global api_key
        try:
            # print(f"Summarising {file} using api key{GROQ_API_KEY}")
            # groq_client = Groq(api_key= GROQ_API_KEY)
            # response = groq_client.chat.completions.create(
            #     model=GROQ_MODEL,
            #     messages=[{"role": "user", "content": prompt}],
            # )
            # return response.choices[0].message.content.strip()
            openai_client = OpenAI(api_key=OPENAI_API_KEY)
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo", 
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error: {e}")
            wait_time = 10 * (attempt + 1)
            print(f"Retrying in {wait_time}s... for {file}")
            time.sleep(wait_time)
    print(f"Failed {file}")
    return "Failed to summarize chunk."

def chunk_code(code, encoding, max_tokens=6000):
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

def summarize_code( block, encoding):
    global count
    code_hash = hashlib.md5(block["source"].encode()).hexdigest()
    if code_hash in summary_cache:
        return summary_cache[code_hash], block["source"], block["file"] , block['name']

    code = block["source"]
    tokens = encoding.encode(code)
    # TODO : check if the code is too short
    if len(tokens) > 10000 :
        return None
    count += len(tokens)
    if len(tokens) > 6000:
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

    summary_cache[code_hash] = full_summary
    return full_summary, block["source"], block["file"] , block['name']

def is_ignored(path):
    return any(ignored in path for ignored in IGNORED_DIRS)

def summarize_all(code_blocks ):
    encoding = tiktoken.get_encoding("cl100k_base")
    results = []

    for block in tqdm(code_blocks, desc="Summarizing"):
            try:
                result = summarize_code(block, encoding)
                if result is not None:
                    # print(len(result))
                    results.append(result)
            except Exception as e:
                print(f"Error summarizing {block.get('path', '<unknown>')}: {e}")
                continue

    return results

def extract_chunks(file_path):
    # print(file_path)
    with open(file_path, 'r', encoding='utf-8') as f:
        source = f.read()

    if len(source) > 100000:
        return None

    chunks = []

    # 1. Extract imports
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

    # 2. Remove imports for chunking
    source_wo_imports = '\n'.join(
        line for i, line in enumerate(source_lines) if i not in used_lines
    )

    # 3. Match function/class declarations
    pattern = re.compile(
        r'^(?P<full>'
        r'(?P<export>export\s+)?(?P<async>async\s+)?function\s+(?P<funcname>\w+)'        # regular function
        r'|export\s+default\s+function\s+(?P<defaultname>\w+)?'                         # export default
        r'|class\s+(?P<classname>\w+)'                                                 # class
        r'|(?P<var_decl>(const|let|var)\s+(?P<varname>\w+)\s*=\s*(async\s+)?(\([^)]*\)|\w+)\s*=>)'  # arrow functions
        r'|(?P<anonfunc>(const|let|var)\s+(?P<anonname>\w+)\s*=\s*function)'           # const name = function() {}
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

        # Determine the correct name
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

    # 4. Add leftover unmatched code
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

def get_semantic_chunks_from_repo(repo_path):
    all_chunks = []
    for root, _, files in os.walk(repo_path):
        if is_ignored(root):
            continue
        for file in files:
            try :
                file_path_temp = os.path.join(root, file)
                if file in IGNORED_FILES or 'scripts' in file_path_temp or 'bootstrap' in file or 'fonts' in file_path_temp:
                    # print(file)
                    continue
                # elif file.endswith('.js') or file.endswith('.tsx') :
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

def run_pipeline():
    repo_dir = BASE_DIR
    code_blocks = get_semantic_chunks_from_repo(repo_dir)
    # print(len(code_blocks))
    summaries_data = summarize_all(code_blocks )
    build_vector_db(summaries_data)

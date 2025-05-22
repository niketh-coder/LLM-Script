import ast
import json
import os
from tree_sitter_languages import get_language, get_parser
from tqdm import tqdm
import re

from build_file_summary import build_file_summary, build_file_summary_prompt

IGNORED_DIRS = {'repos' , 'node_modules', 'venv','myenv', 'env', 'dist', 'build', '.git', '__pycache__' , '.github' , 'lib', 'bin', 'include', 'share', 'tests', 'test' , '.idea' , '.vscode' , '.pytest_cache' , '.mypy_cache' , '.coverage' , '.tox' , '.eggs' , '.hypothesis' , '.pytest' }
IGNORED_FILES = {'.gitignore', 'package-lock.json'}

TS_LANGUAGE = get_language('typescript')
ts_parser = get_parser('typescript')

def is_ignored(path):
    return any(ignored in path for ignored in IGNORED_DIRS)

def extract_calls_from_python_code(code):
    class CallGraphVisitor(ast.NodeVisitor):
        def __init__(self):
            self.names = {}

        def visit_FunctionDef(self, node):
            calls = []
            for n in ast.walk(node):
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                    calls.append(n.func.id)
            self.names[node.name] = {
                "calls": calls,
                "type": "function"
            }
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node):
            self.visit_FunctionDef(node)  

        def visit_ClassDef(self, node):
            method_names = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
            self.names[node.name] = {
                "calls": method_names,
                "type": "class"
            }
            self.generic_visit(node)

    try:
        tree = ast.parse(code)      
        
    except SyntaxError:
        return {"names": {}}

    lines = code.splitlines()
    
    import_lines = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            start = node.lineno - 1
            end = getattr(node, "end_lineno", node.lineno)
            for i in range(start, end):
                import_lines.append(lines[i])

    imports = '\n'.join(import_lines)

    visitor = CallGraphVisitor()
    visitor.visit(tree)
    return {
        "names": visitor.names
    } ,  imports

def extract_calls_from_typescript_code(code):
    
    import_pattern = r'^\s*import\s.+?;?\s*$'
    import_lines = re.findall(import_pattern, code, re.MULTILINE)
    imports = '\n'.join(import_lines).strip()
    
    tree = ts_parser.parse(bytes(code, "utf8"))
    root = tree.root_node

    result = {
        "names": {}
    }

    def walk(node, parent_func=None):
        if node.type in ['function_declaration', 'method_definition']:
            name = None
            for child in node.children:
                if child.type == 'identifier':
                    name = code[child.start_byte:child.end_byte]
            if name:
                result["names"].setdefault(name, {"calls": [], "type": "function"})
                parent_func = name

        elif node.type == 'class_declaration':
            class_name = None
            methods = []
            for child in node.children:
                if child.type == 'identifier':
                    class_name = code[child.start_byte:child.end_byte]
                if child.type == 'class_body':
                    for sub in child.children:
                        if sub.type == 'method_definition':
                            for item in sub.children:
                                if item.type == 'property_identifier':
                                    methods.append(code[item.start_byte:item.end_byte])
            if class_name:
                result["names"][class_name] = {"calls": methods, "type": "class"}

        elif node.type == 'call_expression' and parent_func:
            func_name_node = node.child_by_field_name("function")
            if func_name_node and func_name_node.type == 'identifier':
                func_name = code[func_name_node.start_byte:func_name_node.end_byte]
                result["names"].setdefault(parent_func, {"calls": [], "type": "function"})
                result["names"][parent_func]["calls"].append(func_name)

        for child in node.children:
            walk(child, parent_func)

    walk(root)
    return result , imports


def process_directory(base_dir , file_summary_path ,api_key , LLM_MODEL):
    call_graph = {}
    with open(file_summary_path, 'r', encoding='utf-8') as f:
        file_summaries = json.load(f)
    
    all_files = []
    for root, _, files in os.walk(base_dir):
        if is_ignored(root):
            continue
        for file in files:
            if file in IGNORED_FILES:
                continue
            if file.endswith('.py') or file.endswith('.ts'):
                all_files.append((root, file))

    for root, file in tqdm(all_files, desc="Processing files"):
        path = os.path.join(root, file)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                code = f.read()
                if file.endswith('.py'):
                    graph , imports = extract_calls_from_python_code(code)
                elif file.endswith('.ts'):
                    graph , imports = extract_calls_from_typescript_code(code)
                else:
                    continue
                
                call_graph[path] = graph
                prompt = build_file_summary_prompt(path, graph, imports)
                if path not in file_summaries.keys() : 
                    summary = build_file_summary(prompt , api_key , LLM_MODEL)
                    file_summaries[path] = summary
                
        except Exception as e:
            print(f"Error processing {file}: {e}")

    return call_graph , file_summaries

def build_call_graph(base_dir, code_graph_path ,file_summaries_path ,ignored_dirs , ignored_files , LLM_MODEL , api_key):
    global IGNORED_DIRS, IGNORED_FILES
    IGNORED_DIRS = set(ignored_dirs)
    IGNORED_FILES = set(ignored_files)

    call_graph , file_summaries = process_directory(base_dir ,file_summaries_path , api_key , LLM_MODEL)

    with open(code_graph_path, 'w', encoding='utf-8') as f:
        json.dump(call_graph, f, indent=2)
    with open(file_summaries_path, 'w', encoding='utf-8') as f:
        json.dump(file_summaries, f, indent=2)


import ast
import json
import os
from tree_sitter_languages import get_language, get_parser
from tqdm import tqdm

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
            self.names[node.name] = []
            for n in ast.walk(node):
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                    self.names[node.name].append(n.func.id)
            self.generic_visit(node)

        def visit_ClassDef(self, node):
            method_names = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
            self.names[node.name] = method_names
            self.generic_visit(node)

    tree = ast.parse(code)
    visitor = CallGraphVisitor()
    visitor.visit(tree)
    return {
        "names": visitor.names
    }

def extract_calls_from_typescript_code(code):
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
                result["names"].setdefault(name, [])

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
                result["names"][class_name] = methods

        elif node.type == 'call_expression':
            if parent_func:
                func_name_node = node.child_by_field_name("function")
                if func_name_node and func_name_node.type == 'identifier':
                    func_name = code[func_name_node.start_byte:func_name_node.end_byte]
                    result["names"].setdefault(parent_func, []).append(func_name)

        for child in node.children:
            walk(child, parent_func)

    walk(root)
    return result

def process_directory(base_dir):
    call_graph = {}

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
                    graph = extract_calls_from_python_code(code)
                elif file.endswith('.ts'):
                    graph = extract_calls_from_typescript_code(code)
                else:
                    continue
                
                call_graph[path] = graph
        except Exception as e:
            print(f"Error processing {file}: {e}")

    return call_graph

def build_call_graph(base_dir, output_file , ignored_dirs , ignored_files):
    global IGNORED_DIRS, IGNORED_FILES
    IGNORED_DIRS = set(ignored_dirs)
    IGNORED_FILES = set(ignored_files)

    result = process_directory(base_dir)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

    print(f"Call graph saved to {output_file}")

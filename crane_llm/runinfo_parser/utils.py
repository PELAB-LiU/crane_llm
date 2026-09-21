import ast
from typing import Dict, Any, List, Tuple
import tokenize
from io import StringIO
import re

def get_function_sources(code: str) -> Dict[str, str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}
    return {
        node.name: remove_comments(ast.get_source_segment(code, node))
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }

def remove_comments(source_code):
    tokens = tokenize.generate_tokens(StringIO(source_code).readline)
    result = []
    last_lineno = -1
    last_col = 0

    for token in tokens:
        tok_type = token.type
        tok_string = token.string
        start_line, start_col = token.start

        if tok_type == tokenize.COMMENT:
            continue

        if start_line > last_lineno:
            result.append('\n' * (start_line - last_lineno - 1))
            last_col = 0

        if start_col > last_col:
            result.append(' ' * (start_col - last_col))

        result.append(tok_string)
        last_lineno, last_col = token.end

    cleaned_code = ''.join(result)
    # Optional: strip trailing spaces from each line
    cleaned_code = '\n'.join(line.rstrip() for line in cleaned_code.splitlines())
    return cleaned_code #normalize_whitespace(cleaned_code)
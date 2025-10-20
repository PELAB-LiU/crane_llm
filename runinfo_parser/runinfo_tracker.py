# variable_tracker.py

import ast
from typing import List, Tuple

class RuninfoTracker(ast.NodeVisitor):
    """
    Extracts variable, function, and class definitions with their line numbers and cell numbers.
    """

    def __init__(self):
        self.definitions: List[Tuple[str, int]] = []  # (name, lineno)

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.definitions.append((target.id, node.lineno))
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        # Handles annotated assignments like `x: int = 1`
        if isinstance(node.target, ast.Name):
            self.definitions.append((node.target.id, node.lineno))
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self.definitions.append((node.name, node.lineno))
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        self.definitions.append((node.name, node.lineno))
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self.definitions.append((node.name, node.lineno))
        self.generic_visit(node)

    def get_definitions(self, cell_code: str) -> List[Tuple[str, int]]:
        """
        Parse code and return list of (name, lineno) of definitions.
        """
        self.definitions.clear()
        tree = ast.parse(cell_code)
        self.visit(tree)
        return self.definitions

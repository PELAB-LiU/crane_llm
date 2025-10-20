import ast
from typing import Tuple, Set

# Summary of What It Does
# Feature	              Tracked?	Where?
# Global variable uses	    ✅	global_vars
# Method calls (e.g. fit)	✅	method_calls
# Attribute chains       	✅	attr_accesses
# Subscript accesses	    ✅	subscript_vars
# Locals inside functions	❌	skipped via scope counter
# Class body locals	        ❌	skipped via scope counter

class DependencyVisitor(ast.NodeVisitor):
    def __init__(self):
        self.global_vars = set()            # e.g. 'model', 'train_ds'
        self.method_calls = set()           # e.g. ('model', 'fit')
        self.attr_accesses = set()          # e.g. ('model', 'layers'), ('model.layers[0]', 'name')
        self.subscript_exprs = set()        # e.g. 'model.layers'
        self._inside_function_or_class = 0

    def visit_FunctionDef(self, node):
        self._inside_function_or_class += 1
        # Don't visit inside function body for dependency purposes
        self._inside_function_or_class -= 1

    def visit_ClassDef(self, node):
        self._inside_function_or_class += 1
        self._inside_function_or_class -= 1

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and self._inside_function_or_class == 0:
            self.global_vars.add(node.id)

    def visit_Call(self, node):
        if self._inside_function_or_class == 0:
            func = node.func
            if isinstance(func, ast.Attribute):
                base = self._get_expr_repr(func.value)
                if base:
                    self.method_calls.add((base, func.attr))
                    self.attr_accesses.add((base, func.attr))
            elif isinstance(func, ast.Name):
                self.global_vars.add(func.id)
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if self._inside_function_or_class == 0:
            base = self._get_expr_repr(node.value)
            if base:
                self.attr_accesses.add((base, node.attr))
        self.generic_visit(node)

    def visit_Subscript(self, node):
        if self._inside_function_or_class == 0:
            base = self._get_expr_repr(node.value)
            if base:
                self.subscript_exprs.add(base)
        self.generic_visit(node)

    def _get_expr_repr(self, node):
        """Try to statically stringify simple expressions like a.b or a[0]."""
        try:
            return ast.unparse(node)
        except Exception:
            return None

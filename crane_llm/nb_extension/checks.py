"""The built-in checker: crashes it can report as certain, without the LLM.

Before the model is asked, the target cell is walked statement by statement,
in the order Python would evaluate it, against the live kernel namespace.
Every operation the walk reaches is offered to the rules in
``check_rules.py``, with the real values it will be performed on, and a crash
is reported when a rule finds the operation will certainly raise. The walk
itself catches the errors that come from Python's own rules: undefined names,
missing attributes, unpacking the wrong number of values, syntax errors.

What makes the verdict certain is where the walk stops. Anything that could
run code whose effect is unknown -- a call to a user function, a loop, a
``try`` block, an operation on a value the walk did not compute -- ends it,
and the cell goes to the model instead. So every value a rule looks at is
exactly the value the cell would see when it runs. The walk continues only
past operations that change nothing (listed at the end of
``check_rules.py``); if one of them raised instead, the cell would still
crash, only earlier.

The checker therefore never says a cell is safe. It either reports a certain
crash or says nothing.

Nothing here executes the target cell. Values are looked up, never mutated;
the only calls made are to known side-effect-free functions, such as ``len``
of a list or constructing a scikit-learn estimator, whose ``__init__`` only
stores its parameters.
"""

from __future__ import annotations

import ast
import builtins
import importlib.util
import inspect
import sys
import types
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from . import check_rules
from .check_helpers import (
    MISSING,
    OPAQUE,
    RULES,
    SCALARS,
    CONTAINERS,
    MAX_INSPECTED,
    TRUSTED,
    BinOp,
    Call,
    Compare,
    Delete,
    Iterate,
    Opaque,
    RuleCrash,
    Store,
    Subscript,
    Truth,
    UnaryOp,
    Unpack,
    bound_receiver,
    is_data,
    is_frame,
    is_hashable_key,
    is_label,
    is_ndarray,
    is_one_of,
    is_series,
    is_sklearn_estimator,
    is_sklearn_estimator_class,
    known,
    numpy,
    scan_budget,
)
from .texts import text


@dataclass
class CheckFinding:
    """A crash the target cell is certain to hit."""

    rule: str
    exception: str
    message: str
    line: int
    snippet: str
    # Notebook variables whose current state causes the crash. They are what
    # the origin locator traces back to the cells that produced them.
    variables: List[str] = field(default_factory=list)
    detail: str = ""

    def reasoning(self) -> str:
        sentence = text(
            "checker.reasoning",
            line=self.line,
            snippet=self.snippet,
            exception=self.exception,
            message=self.message,
        ).rstrip()
        if self.detail:
            if not sentence.endswith((".", "!", "?")):
                sentence += "."
            sentence += f" {self.detail}"
        return sentence


def run_checks(source: str, namespace: Dict[str, Any], shell: Any = None) -> Optional[CheckFinding]:
    """Return the certain crash in ``source``, or None if none was found.

    Never raises: a check that fails for any unforeseen reason is simply a
    check that found nothing.
    """

    if not source or not source.strip():
        return None
    # AST transformers configured in IPython rewrite the code before it runs,
    # so what runs is not what the walk would read.
    if getattr(shell, "ast_transformers", None):
        return None

    try:
        tree, code = _parse(source, shell)
    except _SyntaxFinding as finding:
        return finding.finding
    except Exception:
        return None
    if tree is None:
        return None

    walker = _Walker(code, namespace)
    try:
        for statement in tree.body:
            walker.statement(statement)
    except _Crash as crash:
        return crash.finding
    except _Stop:
        return None
    except Exception:
        return None
    return None


# --- control flow of the walk --------------------------------------------


class _Stop(Exception):
    """The walk reached code whose effect cannot be known without running it."""


class _Crash(Exception):
    def __init__(self, finding: CheckFinding):
        super().__init__(finding.message)
        self.finding = finding


class _SyntaxFinding(Exception):
    def __init__(self, finding: CheckFinding):
        super().__init__(finding.message)
        self.finding = finding


def _parse(source: str, shell: Any) -> Tuple[Optional[ast.Module], str]:
    """Parse the cell the way IPython will run it.

    A cell that cannot be compiled is itself a certain crash, but only when
    IPython's own input transformer was available to rewrite magics and shell
    escapes first; without it, ``%matplotlib inline`` would look like one.
    """

    flags = ast.PyCF_ONLY_AST | getattr(ast, "PyCF_ALLOW_TOP_LEVEL_AWAIT", 0)

    transform = getattr(shell, "transform_cell", None)
    code = source
    if transform is not None:
        try:
            code = transform(source)
        except Exception:
            return None, source

    try:
        return compile(code, "<cell>", "exec", flags), code
    except SyntaxError as exc:
        if transform is None:
            return None, code
        line = exc.lineno or 1
        lines = code.splitlines()
        snippet = lines[line - 1].strip() if 0 < line <= len(lines) else ""
        raise _SyntaxFinding(
            CheckFinding(
                rule="syntax",
                exception=type(exc).__name__,
                message=exc.msg or "invalid syntax",
                line=line,
                snippet=snippet,
            )
        )


# Builtin types with a ``__getattribute__`` slot of their own that is the
# generic lookup, which ``object.__getattribute__`` identity does not reveal.
_GENERIC_BUILTINS = frozenset(
    {int, float, bool, complex, str, bytes, list, tuple, dict, set, frozenset, range, slice}
)

_MAX_BUILT_ARRAY = 1_000_000


class _Walker:
    """Evaluates a cell's statements in order, as far as that is safe.

    Expressions evaluate to ``(value, root)``. ``value`` is the real object,
    or an ``Opaque`` stand-in. ``root`` is the notebook variable the value was
    reached from, such as ``df`` for ``df.columns``, or None.
    """

    def __init__(self, code: str, namespace: Dict[str, Any]):
        self.code = code
        self.namespace = namespace
        # Names bound earlier in the cell itself shadow the namespace.
        self.local: Dict[str, Tuple[Any, Optional[str]]] = {}
        builtins_ns = namespace.get("__builtins__", builtins)
        self.builtins = builtins_ns if isinstance(builtins_ns, dict) else vars(builtins_ns)

    # --- reporting ------------------------------------------------------

    def crash(self, node: ast.AST, rule: str, exception: str, message: str,
              variables=(), detail: str = "") -> None:
        snippet = ast.get_source_segment(self.code, node) or ""
        snippet = " ".join(snippet.split())
        if len(snippet) > 120:
            snippet = snippet[:117] + "..."
        raise _Crash(
            CheckFinding(
                rule=rule,
                exception=exception,
                message=message,
                line=getattr(node, "lineno", 1),
                snippet=snippet,
                variables=[v for v in dict.fromkeys(variables) if v],
                detail=detail,
            )
        )

    def apply_rules(self, site) -> None:
        """Offer an operation to every rule registered for its kind of site."""

        for registered in RULES:
            if not isinstance(site, registered.site):
                continue
            try:
                registered.check(site)
            except RuleCrash as found:
                self.crash(site.node, registered.rule_id, found.exception, found.message,
                           found.variables, found.detail)
            except Exception:
                # A rule that fails has found nothing; it must never turn a
                # mistake of its own into a reported crash.
                continue

    # --- statements -----------------------------------------------------

    def statement(self, node: ast.stmt) -> None:
        if isinstance(node, ast.Expr):
            self.eval(node.value)
        elif isinstance(node, ast.Pass):
            pass
        elif isinstance(node, ast.Assign):
            value, root = self.eval(node.value)
            for target in node.targets:
                self.assign(target, value, root, node)
        elif isinstance(node, ast.Import):
            self.import_(node)
        elif isinstance(node, ast.ImportFrom):
            self.import_from(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self.define_function(node)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            # The iterable is evaluated, and iteration begins, before the body
            # runs at all. The body itself is not walked.
            value, root = self.eval(node.iter)
            if isinstance(node, ast.For) and not isinstance(value, Opaque):
                self.apply_rules(Iterate(node.iter, obj=value, obj_root=root, context="for"))
                self.unpack_first_item(node, value)
            raise _Stop
        elif isinstance(node, (ast.If, ast.While)):
            # The test always runs, and is tested for truth, before either
            # branch.
            value, root = self.eval(node.test)
            if not isinstance(value, Opaque):
                self.apply_rules(Truth(node.test, value=value, root=root))
            raise _Stop
        elif isinstance(node, ast.Delete):
            self.delete(node)
        elif isinstance(node, ast.Raise):
            self.raise_(node)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            # Only the first context expression: entering it runs code, after
            # which nothing further is known.
            self.eval(node.items[0].context_expr)
            raise _Stop
        else:
            # Try blocks, class bodies, deletions, annotated assignments:
            # whether and how their code runs depends on things the walk does
            # not model. A try block in particular may catch the very crash a
            # rule would report.
            raise _Stop

    def import_(self, node: ast.Import) -> None:
        # Only modules that are already imported are bound: importing anything
        # else runs that module's code. A module that cannot be found at all
        # is a certain ModuleNotFoundError.
        for alias in node.names:
            module = sys.modules.get(alias.name)
            if module is None:
                self.check_module_exists(node, alias.name)
                raise _Stop
            if alias.asname:
                self.local[alias.asname] = (module, None)
            else:
                top = alias.name.split(".")[0]
                top_module = sys.modules.get(top)
                if top_module is None:
                    raise _Stop
                self.local[top] = (top_module, None)

    def import_from(self, node: ast.ImportFrom) -> None:
        if node.level or not node.module:
            raise _Stop
        module = sys.modules.get(node.module)
        if module is None:
            self.check_module_exists(node, node.module)
            raise _Stop
        module_dict = vars(module)
        for alias in node.names:
            if alias.name == "*":
                raise _Stop
            if alias.name in module_dict:
                self.local[alias.asname or alias.name] = (module_dict[alias.name], None)
                continue
            self.check_name_importable(node, module, alias.name)
            raise _Stop

    def unpack_first_item(self, node, iterable) -> None:
        """``for a, b in items``: the first item is unpacked into the targets
        before the body runs, so a first item of the wrong shape is a certain
        crash. Taken only from a non-empty builtin sequence or dict, where
        getting the first item changes nothing."""

        if not isinstance(node.target, (ast.Tuple, ast.List)):
            return
        if type(iterable) not in (list, tuple, str, dict, range) or not len(iterable):
            return
        self.assign(node.target, next(iter(iterable)), None, node)

    def delete(self, node: ast.Delete) -> None:
        """``del name`` or ``del obj[key]``. Deleting changes the namespace or the
        object, so the walk stops after the first target either way."""

        target = node.targets[0]
        if isinstance(target, ast.Name):
            if target.id not in self.local and target.id not in self.namespace:
                self.crash(target, "undefined-name", "NameError",
                           f"name '{target.id}' is not defined", [target.id])
        elif isinstance(target, ast.Subscript):
            base, base_root = self.eval(target.value)
            key, key_root = self.eval(target.slice)
            if not isinstance(base, Opaque) and not isinstance(key, Opaque):
                self.apply_rules(Delete(node, obj=base, key=key, obj_root=base_root, key_root=key_root))
        raise _Stop

    def raise_(self, node: ast.Raise) -> None:
        """A ``raise`` the walk reaches always raises: the cell is certain to crash.

        The walk never enters a ``try`` block, so nothing reached can catch it.
        A bare ``raise`` is left alone, since whether there is an exception to
        re-raise depends on how the cell is run.
        """

        if node.exc is None:
            raise _Stop
        value, root = self.eval(node.exc)
        if isinstance(value, Opaque):
            raise _Stop
        if node.cause is not None:
            cause, _ = self.eval(node.cause)
            if isinstance(cause, Opaque):
                raise _Stop
            valid = cause is None or isinstance(cause, BaseException) or (
                isinstance(cause, type) and issubclass(cause, BaseException))
            if not valid:
                self.crash(node, "explicit-raise", "TypeError",
                           "exception causes must derive from BaseException", [root])
        if isinstance(value, BaseException):
            self.crash(node, "explicit-raise", type(value).__name__, str(value), [root])
        if isinstance(value, type) and issubclass(value, BaseException):
            # Raising a class instantiates it. A user-defined __init__ could
            # raise something else first, so name the exception only when the
            # constructor is a builtin one.
            builtin_init = all(
                klass.__module__ == "builtins"
                for klass in value.__mro__
                if "__init__" in vars(klass) or "__new__" in vars(klass)
            )
            if not builtin_init:
                raise _Stop
            self.crash(node, "explicit-raise", value.__name__, "", [root])
        self.crash(node, "explicit-raise", "TypeError", "exceptions must derive from BaseException", [root])

    def check_module_exists(self, node, name: str) -> None:
        """Report ``import name`` when no module of that name can be found.

        Only searched, never imported: finding a top-level module, or a
        submodule of an already imported package, runs none of its code. For a
        submodule of a package not yet imported, searching would import the
        package, so nothing is claimed.
        """

        top = name.split(".")[0]
        if top not in sys.modules:
            try:
                if importlib.util.find_spec(top) is None:
                    self.crash_missing_module(node, top)
            except _Crash:
                raise
            except Exception:
                pass
            return
        if name.rpartition(".")[0] not in sys.modules:
            return
        try:
            spec = importlib.util.find_spec(name)
        except ModuleNotFoundError as exc:
            self.crash(node, "missing-module", "ModuleNotFoundError", str(exc), [],
                       text("checker.module_missing", module=name))
        except Exception:
            return
        if spec is None:
            self.crash_missing_module(node, name)

    def crash_missing_module(self, node, name: str) -> None:
        self.crash(node, "missing-module", "ModuleNotFoundError", f"No module named '{name}'", [],
                   text("checker.module_missing", module=name))

    def check_name_importable(self, node, module, name: str) -> None:
        """Report ``from module import name`` when ``name`` is neither in the
        module nor one of its submodules."""

        if "__getattr__" in vars(module):
            # PEP 562: the module computes some names on request.
            return
        module_name = getattr(module, "__name__", "?")
        if hasattr(module, "__path__"):
            try:
                if importlib.util.find_spec(f"{module_name}.{name}") is not None:
                    return
            except Exception:
                return
        self.crash(node, "missing-module", "ImportError",
                   f"cannot import name '{name}' from '{module_name}'", [],
                   text("checker.name_not_in_module", name=name, module=module_name))

    def define_function(self, node) -> None:
        # Decorators are calls. Annotations are evaluated at definition time
        # before Python 3.14, and the walk does not evaluate them.
        if node.decorator_list or getattr(node, "type_params", None):
            raise _Stop
        arguments = node.args
        annotated = node.returns is not None or any(
            a.annotation is not None
            for a in arguments.posonlyargs + arguments.args + arguments.kwonlyargs
            + [a for a in (arguments.vararg, arguments.kwarg) if a is not None]
        )
        if annotated and sys.version_info < (3, 14):
            raise _Stop
        for default in arguments.defaults:
            self.eval(default)
        for default in arguments.kw_defaults:
            if default is not None:
                self.eval(default)
        self.local[node.name] = (_signature_stub(node), None)

    def assign(self, target: ast.expr, value: Any, root: Optional[str], statement: ast.stmt) -> None:
        if isinstance(target, ast.Name):
            # A name rebound by the cell is no longer the notebook variable.
            self.local[target.id] = (value, None)
            return

        if isinstance(target, (ast.Tuple, ast.List)):
            if any(isinstance(element, ast.Starred) for element in target.elts):
                raise _Stop
            if not isinstance(value, Opaque):
                self.apply_rules(Iterate(statement, obj=value, obj_root=root, context="unpack"))
            if type(value) not in (tuple, list, str) or isinstance(value, Opaque):
                raise _Stop
            expected, got = len(target.elts), len(value)
            if got != expected:
                message = (
                    f"too many values to unpack (expected {expected})"
                    if got > expected
                    else f"not enough values to unpack (expected {expected}, got {got})"
                )
                self.crash(statement, "unpack-count", "ValueError", message, [root])
            for element, item in zip(target.elts, value):
                self.assign(element, item, None, statement)
            return

        if isinstance(target, ast.Subscript):
            base, base_root = self.eval(target.value)
            key, _ = self.eval(target.slice)
            if not isinstance(base, Opaque) and not isinstance(key, Opaque):
                self.apply_rules(Store(statement, obj=base, key=key, value=value,
                                       obj_root=base_root, value_root=root))

        # Storing into an object changes it, so nothing known about it holds.
        raise _Stop

    # --- expressions ----------------------------------------------------

    def eval(self, node: ast.expr) -> Tuple[Any, Optional[str]]:
        method = getattr(self, "eval_" + type(node).__name__, None)
        if method is None:
            raise _Stop
        return method(node)

    def eval_Constant(self, node: ast.Constant):
        return node.value, None

    def eval_Name(self, node: ast.Name):
        if not isinstance(node.ctx, ast.Load):
            raise _Stop
        name = node.id
        if name in self.local:
            return self.local[name]
        if name in self.namespace:
            return self.namespace[name], name
        if name in self.builtins:
            return self.builtins[name], None
        self.crash(node, "undefined-name", "NameError", f"name '{name}' is not defined", [name])

    def eval_Tuple(self, node: ast.Tuple):
        return self._sequence(node, tuple)

    def eval_List(self, node: ast.List):
        return self._sequence(node, list)

    def _sequence(self, node, kind):
        if not isinstance(node.ctx, ast.Load) or any(isinstance(e, ast.Starred) for e in node.elts):
            raise _Stop
        items = [self.eval(element)[0] for element in node.elts]
        if any(isinstance(item, Opaque) for item in items):
            return (TRUSTED if all(is_data(item) for item in items) else OPAQUE), None
        return kind(items), None

    def eval_Dict(self, node: ast.Dict):
        result = {}
        complete = True
        for key_node, value_node in zip(node.keys, node.values):
            if key_node is None:
                raise _Stop
            key, _ = self.eval(key_node)
            value, _ = self.eval(value_node)
            if type(key) in (list, dict, set, bytearray):
                self.crash(key_node, "unhashable-key", "TypeError",
                           f"unhashable type: '{type(key).__name__}'", [])
            # Hashing anything but a builtin scalar could run user code.
            if not is_hashable_key(key):
                raise _Stop
            if isinstance(value, Opaque):
                complete = False
            else:
                result[key] = value
        return (result if complete else TRUSTED), None

    def eval_UnaryOp(self, node: ast.UnaryOp):
        operand, operand_root = self.eval(node.operand)
        if isinstance(node.op, ast.Not) and not isinstance(operand, Opaque):
            self.apply_rules(Truth(node.operand, value=operand, root=operand_root))
        symbol = {ast.USub: "-", ast.UAdd: "+", ast.Invert: "~"}.get(type(node.op))
        if symbol is not None and not isinstance(operand, Opaque):
            self.apply_rules(UnaryOp(node, op=symbol, operand=operand, root=operand_root))
        if type(operand) in (int, float, bool):
            if isinstance(node.op, ast.USub):
                return -operand, None
            if isinstance(node.op, ast.UAdd):
                return +operand, None
            if isinstance(node.op, ast.Not):
                return not operand, None
        if is_data(operand) and not isinstance(operand, Opaque) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return TRUSTED, None
        raise _Stop

    def eval_BinOp(self, node: ast.BinOp):
        left, left_root = self.eval(node.left)
        right, right_root = self.eval(node.right)
        op = check_rules.OPERATORS.get(type(node.op))
        if op is None:
            raise _Stop
        if not isinstance(left, Opaque) and not isinstance(right, Opaque):
            self.apply_rules(BinOp(node, op=op, left=left, right=right,
                                   left_root=left_root, right_root=right_root))

        if type(left) in SCALARS and type(right) in SCALARS:
            return self.builtin_value(op, left, right), None
        # NumPy and pandas arithmetic changes nothing. If it raises, the cell
        # crashes there, which is still a crash.
        if is_data(left) and is_data(right):
            return TRUSTED, None
        raise _Stop

    @staticmethod
    def builtin_value(op: str, left, right) -> Any:
        """The result of arithmetic on builtin scalars, when it is cheap to know."""

        compute = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b,
                   "/": lambda a, b: a / b, "//": lambda a, b: a // b, "%": lambda a, b: a % b}.get(op)
        for value in (left, right):
            if type(value) in (str, bytes) and len(value) > 10_000:
                return TRUSTED
            if type(value) is int and abs(value) > 10**12:
                return TRUSTED
        if compute is None or (op == "*" and (type(left) in (str, bytes) or type(right) in (str, bytes))):
            return TRUSTED
        try:
            return compute(left, right)
        except Exception:
            return TRUSTED

    def eval_Compare(self, node: ast.Compare):
        left, left_root = self.eval(node.left)
        right, right_root = self.eval(node.comparators[0])
        op = check_rules.COMPARISONS.get(type(node.ops[0]))
        if op is not None and not isinstance(left, Opaque) and not isinstance(right, Opaque):
            # Only the first comparison of a chain is certain to run.
            self.apply_rules(Compare(node, op=op, left=left, right=right,
                                     left_root=left_root, right_root=right_root))
        values = [left, right] + [self.eval(c)[0] for c in node.comparators[1:]]
        if all(type(v) in SCALARS for v in values) and all(
            isinstance(op, (ast.Eq, ast.NotEq, ast.Is, ast.IsNot)) for op in node.ops
        ):
            return TRUSTED, None
        if all(is_data(v) for v in values):
            return TRUSTED, None
        raise _Stop

    def eval_BoolOp(self, node: ast.BoolOp):
        # Which operands run depends on truthiness, known only for builtins.
        result = None
        for operand in node.values:
            result, result_root = self.eval(operand)
            if not isinstance(result, Opaque):
                self.apply_rules(Truth(operand, value=result, root=result_root))
            if type(result) not in SCALARS and type(result) not in CONTAINERS:
                raise _Stop
            truthy = bool(result)
            if isinstance(node.op, ast.And) and not truthy:
                return result, None
            if isinstance(node.op, ast.Or) and truthy:
                return result, None
        return result, None

    def eval_IfExp(self, node: ast.IfExp):
        test, test_root = self.eval(node.test)
        if not isinstance(test, Opaque):
            self.apply_rules(Truth(node.test, value=test, root=test_root))
        if type(test) not in SCALARS and type(test) not in CONTAINERS:
            raise _Stop
        return self.eval(node.body if test else node.orelse)

    def eval_JoinedStr(self, node: ast.JoinedStr):
        for part in node.values:
            if isinstance(part, ast.FormattedValue):
                value, root = self.eval(part.value)
                if not is_data(value):
                    raise _Stop
                if part.format_spec is not None:
                    self.eval(part.format_spec)
                spec = _constant_spec(part.format_spec)
                if part.conversion == -1 and spec is not None and not isinstance(value, Opaque):
                    # f"{x:spec}" is format(x, spec), so it gets the same rules.
                    self.apply_rules(Call(part, func=builtins.format, args=[value, spec],
                                          arg_roots=[root, None]))
        return TRUSTED, None

    # Comprehensions and generator expressions evaluate their first iterable,
    # and start iterating it, immediately. The rest runs per item.
    def eval_ListComp(self, node):
        return self.comprehension(node)

    eval_SetComp = eval_ListComp
    eval_DictComp = eval_ListComp
    eval_GeneratorExp = eval_ListComp

    def comprehension(self, node):
        first = node.generators[0]
        value, root = self.eval(first.iter)
        if not isinstance(value, Opaque):
            self.apply_rules(Iterate(first.iter, obj=value, obj_root=root, context="comprehension"))
        raise _Stop

    def eval_Lambda(self, node: ast.Lambda):
        for default in node.args.defaults + [d for d in node.args.kw_defaults if d is not None]:
            self.eval(default)
        return OPAQUE, None

    def eval_NamedExpr(self, node: ast.NamedExpr):
        value, root = self.eval(node.value)
        self.local[node.target.id] = (value, None)
        return value, root

    def eval_Slice(self, node: ast.Slice):
        parts = []
        for part in (node.lower, node.upper, node.step):
            if part is None:
                parts.append(None)
                continue
            value, _ = self.eval(part)
            if type(value) not in SCALARS:
                raise _Stop
            parts.append(value)
        return slice(*parts), None

    # Python 3.8 wraps subscripts in Index nodes.
    def eval_Index(self, node):
        return self.eval(node.value)

    # --- attribute access ------------------------------------------------

    def eval_Attribute(self, node: ast.Attribute):
        if not isinstance(node.ctx, ast.Load):
            raise _Stop
        base, root = self.eval(node.value)
        if isinstance(base, Opaque):
            raise _Stop
        return self.attribute(node, base, root, node.attr), root

    def attribute(self, node, obj: Any, root: Optional[str], name: str) -> Any:
        kind = type(obj)

        if kind is types.ModuleType:
            return self.module_attribute(node, obj, root, name)

        if isinstance(obj, type):
            # Class attribute access goes through the metaclass.
            meta = type(obj)
            if meta.__getattribute__ is not type.__getattribute__ or _static_lookup(meta, "__getattr__") is not MISSING:
                raise _Stop
            has_getattr = False
        else:
            if kind.__getattribute__ is not object.__getattribute__ and kind not in _GENERIC_BUILTINS:
                raise _Stop
            has_getattr = _static_lookup(kind, "__getattr__") is not MISSING

        try:
            raw = inspect.getattr_static(obj, name, MISSING)
        except Exception:
            raise _Stop

        if raw is MISSING:
            if not has_getattr:
                self.missing_attribute(node, obj, root, name)
            if (is_frame(obj) or is_series(obj)) and not name.startswith("_"):
                # pandas resolves unknown attributes to columns (or, for a
                # Series, index labels) and raises AttributeError otherwise.
                try:
                    holds = obj._info_axis._can_hold_identifiers_and_holds_name(name)
                except Exception:
                    raise _Stop
                if holds:
                    # The column itself, as for df["col"].
                    try:
                        return obj[name]
                    except Exception:
                        return TRUSTED
                self.missing_attribute(node, obj, root, name)
            raise _Stop

        return self.resolve_attribute(node, obj, root, name, raw)

    def resolve_attribute(self, node, obj, root, name, raw) -> Any:
        """The value of an attribute known to exist, if reading it is pure."""

        if is_frame(obj) or is_series(obj):
            if name in ("str", "dt", "cat") and len(obj) > scan_budget():
                # Building an accessor reads every value to check the dtype.
                return TRUSTED
            if name in check_rules.PANDAS_CHEAP_ATTRIBUTES or isinstance(raw, types.FunctionType):
                return self.safe_getattr(node, obj, root, name)
            return TRUSTED

        if isinstance(raw, (types.FunctionType, staticmethod, classmethod, types.BuiltinFunctionType,
                            types.MethodDescriptorType, types.WrapperDescriptorType,
                            types.ClassMethodDescriptorType, types.GetSetDescriptorType,
                            types.MemberDescriptorType)):
            return self.safe_getattr(node, obj, root, name)

        if isinstance(obj, type) and isinstance(raw, property):
            return raw

        if not hasattr(type(raw), "__get__"):
            # A plain value: an instance attribute or a class constant.
            return self.safe_getattr(node, obj, root, name)

        if is_sklearn_estimator(obj) and type(raw).__module__.startswith("sklearn."):
            # ``available_if`` on meta-estimators: a pure check that raises
            # AttributeError when the wrapped estimator lacks the method.
            return self.safe_getattr(node, obj, root, name)

        raise _Stop

    def safe_getattr(self, node, obj, root, name) -> Any:
        try:
            return getattr(obj, name)
        except AttributeError as exc:
            self.crash(node, "missing-attribute", "AttributeError", str(exc), [root])
        except Exception:
            raise _Stop

    def missing_attribute(self, node, obj, root, name) -> None:
        if isinstance(obj, type):
            message = f"type object '{obj.__name__}' has no attribute '{name}'"
        else:
            message = f"'{type(obj).__name__}' object has no attribute '{name}'"
        detail = text("checker.is_none", name=root) if obj is None and root else ""
        self.crash(node, "missing-attribute", "AttributeError", message, [root], detail)

    def module_attribute(self, node, module, root, name) -> Any:
        module_dict = vars(module)
        if name in module_dict:
            return module_dict[name]

        module_name = getattr(module, "__name__", "?")
        hook = module_dict.get("__getattr__")
        if hook is None:
            self.crash(
                node, "missing-attribute", "AttributeError",
                f"module '{module_name}' has no attribute '{name}'", [root],
            )

        # A module-level __getattr__ can import submodules lazily. Only call
        # it for libraries whose hook is a plain lookup, and never for a name
        # that is a submodule.
        if module_name.split(".")[0] not in check_rules.TRUSTED_MODULE_GETATTR:
            raise _Stop
        if hasattr(module, "__path__"):
            try:
                if importlib.util.find_spec(f"{module_name}.{name}") is not None:
                    raise _Stop
            except _Stop:
                raise
            except Exception:
                raise _Stop
        try:
            return hook(name)
        except AttributeError as exc:
            self.crash(node, "missing-attribute", "AttributeError", str(exc).split("\n")[0], [root])
        except Exception:
            raise _Stop

    # --- subscripts -----------------------------------------------------

    def eval_Subscript(self, node: ast.Subscript):
        if not isinstance(node.ctx, ast.Load):
            raise _Stop
        base, root = self.eval(node.value)
        key, key_root = self.eval(node.slice)
        if isinstance(base, Opaque) or isinstance(key, Opaque):
            raise _Stop
        self.apply_rules(Subscript(node, obj=base, key=key, obj_root=root, key_root=key_root))
        return self.subscript_value(base, key), root

    @staticmethod
    def subscript_value(obj, key) -> Any:
        """What ``obj[key]`` evaluates to, once no rule found it raises."""

        kind = type(obj)
        if is_frame(obj):
            # A single column is a view, cheap to take for real, so that rules
            # see it: df["col"].astype(int).
            if is_label(key) and key in obj.columns:
                return obj[key]
            return TRUSTED
        if _is_pandas_indexer(obj):
            if is_data(key):
                return TRUSTED
            raise _Stop
        if is_series(obj) or is_ndarray(obj):
            if is_data(key):
                return TRUSTED
            raise _Stop
        if kind is dict and is_hashable_key(key) and key in obj:
            return obj[key]
        if kind in (list, tuple, str) and type(key) in (int, slice):
            if type(key) is slice and not all(type(p) in (int, bool, type(None)) for p in (key.start, key.stop, key.step)):
                raise _Stop
            try:
                return obj[key]
            except Exception:
                raise _Stop
        raise _Stop

    # --- calls ----------------------------------------------------------

    def eval_Call(self, node: ast.Call):
        func, func_root = self.eval(node.func)
        args, arg_roots = [], []
        for argument in node.args:
            if isinstance(argument, ast.Starred):
                self.unpacked_argument(argument.value, "*")
            value, value_root = self.eval(argument)
            args.append(value)
            arg_roots.append(value_root)
        kwargs, kwarg_roots = {}, {}
        for keyword in node.keywords:
            if keyword.arg is None:
                self.unpacked_argument(keyword.value, "**")
            value, value_root = self.eval(keyword.value)
            kwargs[keyword.arg] = value
            kwarg_roots[keyword.arg] = value_root
        if isinstance(func, Opaque):
            raise _Stop
        site = Call(node, func=func, args=args, kwargs=kwargs, func_root=func_root,
                    arg_roots=arg_roots, kwarg_roots=kwarg_roots)
        self.apply_rules(site)
        return self.call_value(site), None

    def unpacked_argument(self, node, kind: str) -> None:
        """``f(*x)`` or ``f(**x)``: check ``x`` can be unpacked, then stop, since
        what the call receives is no longer known argument by argument."""

        value, root = self.eval(node)
        if not isinstance(value, Opaque):
            self.apply_rules(Unpack(node, obj=value, obj_root=root, kind=kind))
        raise _Stop

    def call_value(self, site: Call) -> Any:
        """What a call evaluates to, once no rule found it raises.

        Only calls that change nothing are walked past. Anything else stops
        the walk, since the call may change any value a later rule would see.
        """

        func, args, kwargs = site.func, site.args, site.kwargs
        plain = all(is_data(a) for a in args) and all(is_data(v) for v in kwargs.values())

        result = self.builtin_value_of_call(site)
        if result is not MISSING:
            return result

        receiver = bound_receiver(func)
        if is_frame(receiver) or is_series(receiver):
            if site.method not in check_rules.PANDAS_PURE_METHODS or not site.is_genuine_method():
                raise _Stop
            if kwargs.get("inplace", False) is not False or not plain:
                raise _Stop
            return TRUSTED
        if is_ndarray(receiver):
            if site.method not in check_rules.NUMPY_PURE_METHODS or not plain:
                raise _Stop
            return TRUSTED

        np = numpy()
        name = getattr(func, "__name__", None)
        if np is not None and isinstance(name, str) and getattr(np, name, None) is func:
            exact = all(known(a) for a in args) and all(known(v) for v in kwargs.values())
            if name in check_rules.NUMPY_BUILT_FOR_REAL and plain and exact and _small_array(func, np, args):
                try:
                    return func(*args, **kwargs)
                except Exception:
                    raise _Stop
            if name in check_rules.NUMPY_PURE_FUNCTIONS and plain:
                return TRUSTED
            raise _Stop

        if (
            inspect.isclass(func)
            and issubclass(func, BaseException)
            and func.__module__ == "builtins"
            and all(type(a) in SCALARS for a in args)
            and not kwargs
        ):
            # ValueError("...") and friends: building one changes nothing.
            return func(*args)

        if inspect.isclass(func) and is_sklearn_estimator_class(func):
            # By scikit-learn's own rules ``__init__`` only stores its
            # parameters, so building the estimator is side-effect free, and
            # later rules can see that it is not fitted.
            if all(is_data(a) and known(a) for a in args) and all(is_data(v) and known(v) for v in kwargs.values()):
                try:
                    return func(*args, **kwargs)
                except Exception:
                    raise _Stop
        raise _Stop

    @staticmethod
    def builtin_value_of_call(site: Call) -> Any:
        """The result of a side-effect-free builtin call, or MISSING."""

        func, args, kwargs = site.func, site.args, site.kwargs
        if func is builtins.print or (
            getattr(func, "__name__", "") == "display"
            and getattr(func, "__module__", "").startswith("IPython.")
        ):
            if all(is_data(a) for a in args) and all(
                key in ("sep", "end") and type(value) is str for key, value in kwargs.items()
            ):
                return None
            raise _Stop

        if func is builtins.len and len(args) == 1 and not kwargs:
            value = args[0]
            if type(value) in (list, tuple, dict, set, frozenset, str, bytes, range):
                return len(value)
            if is_ndarray(value) and value.ndim:
                return value.shape[0]
            if is_frame(value) or is_series(value):
                return len(value.index)
            if isinstance(value, Opaque) and value.trusted:
                return TRUSTED
            raise _Stop

        if func is builtins.type and len(args) == 1 and not kwargs:
            return type(args[0]) if known(args[0]) else TRUSTED

        if is_one_of(func, builtins.int, builtins.float) and len(args) == 1 and not kwargs:
            if type(args[0]) in (str, int, float, bool):
                try:
                    return func(args[0])
                except Exception:
                    raise _Stop
            raise _Stop

        if func is builtins.str and len(args) <= 1 and not kwargs:
            if not args or type(args[0]) in SCALARS:
                return str(*args)
            raise _Stop

        if func is builtins.range and not kwargs and 1 <= len(args) <= 3 and all(type(a) is int for a in args):
            try:
                return range(*args)
            except Exception:
                raise _Stop

        if is_one_of(func, builtins.list, builtins.tuple) and len(args) == 1 and not kwargs:
            value = args[0]
            if type(value) in (list, tuple, str, range, dict) and len(value) <= MAX_INSPECTED:
                return func(value)
            raise _Stop

        return MISSING


def _small_array(func, np, args) -> bool:
    """True when an array constructor call will build at most a million items."""

    if not args:
        return False
    first = args[0]
    if is_one_of(func, np.array, np.asarray):
        # The data is a literal, already bounded by is_data.
        return not is_ndarray(first)
    if func is np.eye:
        return type(first) is int and 0 <= first * first <= _MAX_BUILT_ARRAY
    shape = (first,) if type(first) is int else first
    if type(shape) not in (tuple, list) or not all(type(n) is int and n >= 0 for n in shape):
        return False
    size = 1
    for n in shape:
        size *= n
    return size <= _MAX_BUILT_ARRAY


def _static_lookup(kind: type, name: str) -> Any:
    for klass in kind.__mro__:
        if name in vars(klass):
            return vars(klass)[name]
    return MISSING


def _is_pandas_indexer(value: Any) -> bool:
    """``df.loc`` or ``df.iloc``, which subscripting changes nothing through."""

    return type(value).__module__ == "pandas.core.indexing" and type(value).__name__ in (
        "_LocIndexer",
        "_iLocIndexer",
    )


def _constant_spec(spec: Optional[ast.AST]) -> Optional[str]:
    """The format spec of an f-string part, when it is written out literally."""

    if spec is None:
        return ""
    if isinstance(spec, ast.JoinedStr) and all(
        isinstance(part, ast.Constant) and isinstance(part.value, str) for part in spec.values
    ):
        return "".join(part.value for part in spec.values)
    return None


def _signature_stub(node) -> Any:
    """A function with the same parameters as the one the cell defines, and an
    empty body, so that calls to it later in the cell can be checked against
    its signature. It is never called."""

    arguments = node.args
    stub_arguments = ast.arguments(
        posonlyargs=[ast.arg(a.arg) for a in arguments.posonlyargs],
        args=[ast.arg(a.arg) for a in arguments.args],
        vararg=ast.arg(arguments.vararg.arg) if arguments.vararg else None,
        kwonlyargs=[ast.arg(a.arg) for a in arguments.kwonlyargs],
        kw_defaults=[None if d is None else ast.Constant(None) for d in arguments.kw_defaults],
        kwarg=ast.arg(arguments.kwarg.arg) if arguments.kwarg else None,
        defaults=[ast.Constant(None) for _ in arguments.defaults],
    )
    definition = ast.FunctionDef(
        name=node.name, args=stub_arguments, body=[ast.Pass()], decorator_list=[], returns=None
    )
    if sys.version_info >= (3, 12):
        definition.type_params = []
    module = ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[]))
    namespace: Dict[str, Any] = {}
    try:
        exec(compile(module, "<signature>", "exec"), namespace)
    except Exception:
        return OPAQUE
    return namespace[node.name]

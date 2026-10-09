"""The built-in checker's rules: the crashes it can report as certain.

Each rule is a function that looks at one operation the target cell is about
to perform, with the real values it will be performed on, and calls
``site.crash(...)`` when that operation will certainly raise. The walker
(``../walker.py``) goes through the cell in evaluation order and offers every
operation it reaches to the rules registered for that kind of site.

The rules are grouped by the library whose values they look at:
``python.py`` for Python's own types and builtins, then ``pandas.py``,
``numpy.py``, ``sklearn.py``, ``matplotlib.py`` and ``torch.py``.
``shapes.py`` holds the shape arithmetic the NumPy and PyTorch rules share.
Rules are tried in the order they are registered: module by module in the
order imported below, and from top to bottom within a module. When several
rules would report the same operation, the first one does.

Adding a rule
-------------

1. Pick the site: ``Subscript`` (``obj[key]``), ``Store`` (``obj[key] =
   value``), ``Call`` (``func(...)``, including methods), ``BinOp`` (``left +
   right`` and friends), ``Compare`` (``left < right``, ``x in y``),
   ``Iterate`` (``for x in obj``, comprehensions, unpacking), ``Truth``
   (``if value``, ``and``/``or``/``not``), ``UnaryOp`` (``-x``, ``~x``),
   ``Delete`` (``del obj[key]``) or ``Unpack`` (``f(*x)``, ``f(**x)``). Their
   fields are in ``../sites.py``.
2. In the module for the library, write a function decorated with
   ``@rule(<site>, "<rule-id>")``. Return when the rule does not apply; call
   ``site.crash(exception, message, variables, detail)`` when the operation
   will raise.
3. Only report what is certain. Check exact types (``is_frame``, ``type(x)
   is list``), never ``isinstance``, since a subclass can behave differently.
   Treat a value as known only if ``known(value)``. A rule that must read
   every value of a column or array checks its size against
   ``scan_budget()`` first and skips larger data: a skipped check costs a
   missed crash, while a guess could report a false one. Confirm the
   behaviour on the library itself, on every version you rely on.
4. Add the case to ``test_builtin_checks_are_certain`` in
   ``tests/test_nb_extension.py``, which runs it for real and checks that it
   raises what you said it would. If the rule could plausibly fire on code
   that works, add that code to ``test_builtin_checks_pass_working_code``.

A rule that raises anything other than through ``site.crash`` is treated as
having found nothing, so a mistake in a rule costs a missed crash, never a
false one.

Template::

    @rule(Call, "my-rule")
    def my_rule(site: Call) -> None:
        if site.method != "something" or not is_frame(site.receiver):
            return
        if <the call will certainly raise>:
            site.crash("ValueError", "<what Python would say>", [site.receiver_root])

The walker itself handles the errors that come from Python's own rules
rather than from a library: undefined names, missing attributes (including
on ``None`` and removed library APIs), imports of modules or names that do
not exist, unhashable dict keys, unpacking the wrong number of values, and
syntax errors. Which operations it may walk past, and so how far into a cell
the rules can see, is listed in ``../pure.py``.
"""

# Importing a module registers its rules, in this order.
from . import python, pandas, numpy, sklearn, matplotlib, torch  # noqa: F401

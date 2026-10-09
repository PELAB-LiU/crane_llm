"""The operations the checker may walk past.

After an operation, the checker either continues to the next one or stops
and leaves the cell to the LLM. It may continue only past operations that
change nothing, so that every later value is still exactly what the cell
will see. If one of these raised instead, the cell would still crash, only
earlier, so the verdict stays right.

Adding a name here lets the rules see further into cells that use it.
Only add operations that never modify their arguments, the receiver, or
anything else, for any input.
"""

import ast

# pandas methods that return a new object and change nothing, as long as
# ``inplace`` is not set. Their result is not computed.
PANDAS_PURE_METHODS = frozenset(
    {
        "head", "tail", "describe", "info", "nunique", "unique", "value_counts",
        "isnull", "isna", "notnull", "notna", "copy", "sum", "mean", "median",
        "min", "max", "std", "var", "count", "corr", "astype", "to_numpy",
        "drop", "dropna", "fillna", "rename", "reset_index", "sort_values",
        "sort_index", "set_index", "groupby", "select_dtypes", "duplicated",
        "drop_duplicates", "memory_usage", "abs", "round", "nlargest", "nsmallest",
        "idxmax", "idxmin", "any", "all", "get_dummies", "transpose", "reindex",
    }
)

# Attributes of pandas objects that are cheap to read and whose real value is
# kept. Other pandas attributes are side-effect free too, but may copy data,
# so they are left uncomputed.
PANDAS_CHEAP_ATTRIBUTES = frozenset(
    {"shape", "columns", "index", "dtypes", "dtype", "ndim", "size", "empty", "name", "names",
     "loc", "iloc",
     # Reading an accessor checks the dtype and raises AttributeError when it
     # does not fit, as .str does on a numeric column.
     "str", "dt", "cat"}
)

# NumPy array methods that return a new array and change nothing.
NUMPY_PURE_METHODS = frozenset(
    {
        "reshape", "astype", "copy", "sum", "mean", "min", "max", "std", "var",
        "flatten", "ravel", "transpose", "squeeze", "any", "all", "argmax",
        "argmin", "round", "tolist", "cumsum", "clip",
    }
)

# NumPy functions (``np.<name>``) that return a new array and change nothing.
NUMPY_PURE_FUNCTIONS = frozenset(
    {
        "array", "asarray", "zeros", "ones", "full", "arange", "linspace", "mean",
        "sum", "sqrt", "log", "exp", "abs", "max", "min", "unique", "argmax",
        "argmin", "std", "var", "median", "round", "isnan", "transpose", "squeeze",
        "expand_dims", "stack", "where", "cumsum", "clip", "eye", "dot", "matmul",
        "concatenate", "vstack", "hstack", "reshape",
    }
)

# NumPy constructors that are run for real when they build at most a million
# items, so that later rules see the actual shape, as in
# ``model.fit(X, np.ones(4))``.
NUMPY_BUILT_FOR_REAL = frozenset({"zeros", "ones", "full", "eye", "array", "asarray"})

# Libraries whose module-level ``__getattr__`` is a plain lookup, so that
# ``np.float`` can be checked by calling it.
TRUSTED_MODULE_GETATTR = frozenset({"numpy", "pandas", "sklearn", "scipy", "matplotlib", "seaborn"})

# How the checker spells each operator in ``BinOp.op``.
OPERATORS = {
    ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.FloorDiv: "//",
    ast.Mod: "%", ast.Pow: "**", ast.MatMult: "@", ast.BitAnd: "&", ast.BitOr: "|",
    ast.BitXor: "^", ast.LShift: "<<", ast.RShift: ">>",
}

# How the checker spells each comparison in ``Compare.op``.
COMPARISONS = {
    ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=",
    ast.Is: "is", ast.IsNot: "is not", ast.In: "in", ast.NotIn: "not in",
}

"""Rules for scikit-learn: estimators used before fitting, data that does not
match what the estimator was fitted on or can take, and arguments outside
a function's declared constraints.
"""

from __future__ import annotations

import inspect
import sys
from typing import List, NamedTuple, Optional

from ..sites import (
    Call,
    SCALARS,
    is_frame,
    is_ndarray,
    is_series,
    is_sklearn_estimator,
    is_sklearn_estimator_class,
    numpy,
    pandas,
    plain_axis,
    rule,
    sample_count,
    scan_budget,
    signature_target,
)
from ...texts import text


# --- what an estimator takes -------------------------------------------------


# Methods that scikit-learn's own estimator checks require to raise on an
# estimator that has not been fitted. ``transform`` is not among them:
# stateless transformers such as ``Normalizer`` transform without fitting.
_PREDICT_METHODS = frozenset({"predict", "predict_proba", "predict_log_proba", "decision_function"})


# Methods that scikit-learn's estimator checks require to reject input with a
# different number of features than the estimator was fitted on.
_FEATURE_CHECKED_METHODS = _PREDICT_METHODS | {"transform", "score"}


_DATA_METHODS = frozenset({"fit", "predict", "predict_proba", "predict_log_proba", "decision_function",
                           "transform", "fit_transform", "fit_predict", "score"})


class _InputTags(NamedTuple):
    # scikit-learn's own estimator checks hold the estimator to validating
    # 2-D input, and so to rejecting the wrong number of features.
    validates_2d: bool
    # It takes no text, categories or dicts: encoders such as OneHotEncoder do.
    numbers_only: bool
    allow_nan: bool


def _input_tags(estimator) -> Optional[_InputTags]:
    """What scikit-learn's tags say about the input an estimator takes.

    The tags changed form in 1.6, so they are read the way the installed
    version stores them. None when they cannot be read: then claim nothing.
    """

    utils = sys.modules.get("sklearn.utils")
    get_tags = getattr(utils, "get_tags", None) if utils is not None else None
    try:
        if get_tags is not None:
            tags = get_tags(estimator)
            inputs = tags.input_tags
            return _InputTags(
                validates_2d=bool(inputs.two_d_array) and not inputs.one_d_array and not tags.no_validation,
                numbers_only=not inputs.string and not inputs.categorical and not inputs.dict,
                allow_nan=bool(inputs.allow_nan),
            )
        tags = estimator._get_tags()
        two_d_only = tags.get("X_types") == ["2darray"]
        return _InputTags(
            validates_2d=two_d_only and not tags.get("no_validation", False),
            numbers_only=two_d_only,
            allow_nan=bool(tags.get("allow_nan", False)),
        )
    except Exception:
        return None


def _validates_plain_input(estimator) -> bool:
    """A scikit-learn estimator, not a pipeline or meta-estimator, that its own
    estimator checks hold to validating numeric 2-D input."""

    base = sys.modules.get("sklearn.base")
    metaestimators = sys.modules.get("sklearn.utils.metaestimators")
    if base is None or not is_sklearn_estimator(estimator):
        return False
    if isinstance(estimator, base.MetaEstimatorMixin):
        return False
    composition = getattr(metaestimators, "_BaseComposition", None)
    if composition is not None and isinstance(estimator, composition):
        return False
    tags = _input_tags(estimator)
    return tags is not None and tags.validates_2d


def _sklearn_at_least(major: int, minor: int) -> bool:
    module = sys.modules.get("sklearn")
    try:
        parts = str(module.__version__).split(".")
        return (int(parts[0]), int(parts[1])) >= (major, minor)
    except Exception:
        return False


# --- fitting and predicting --------------------------------------------------


@rule(Call, "not-fitted")
def estimator_not_fitted(site: Call) -> None:
    """``model.predict(X)`` before ``model.fit``."""

    if site.method not in _PREDICT_METHODS or not is_sklearn_estimator(site.receiver):
        return
    validation = sys.modules.get("sklearn.utils.validation")
    exceptions = sys.modules.get("sklearn.exceptions")
    if validation is None or exceptions is None:
        return
    try:
        validation.check_is_fitted(site.receiver)
    except exceptions.NotFittedError as exc:
        root = site.receiver_root
        site.crash("NotFittedError", str(exc).split(".")[0] + ".", [root],
                   text("checker.not_fitted", name=root) if root else "")


@rule(Call, "feature-names")
def estimator_feature_names(site: Call) -> None:
    """``model.predict(new_df)`` where ``new_df``'s columns are not the ones
    the estimator was fitted on, in the same order. An error from
    scikit-learn 1.2 on; before that only a warning.

    Not for composite estimators: a ColumnTransformer picks its columns from
    a DataFrame by name, in any order.
    """

    estimator, data = site.receiver, site.arg(0, "X")
    if site.method not in _FEATURE_CHECKED_METHODS or not _validates_plain_input(estimator) or not is_frame(data):
        return
    fitted = vars(estimator).get("feature_names_in_")
    if fitted is None or not _sklearn_at_least(1, 2):
        return
    columns = list(data.columns)
    if not all(type(c) is str for c in columns):
        return
    expected = [str(c) for c in fitted]
    if columns != expected:
        data_root, root = site.arg_root(0, "X"), site.receiver_root
        site.crash("ValueError", "The feature names should match those that were passed during fit.",
                   [data_root, root],
                   text("checker.feature_names", data=data_root, columns=_listed(columns),
                        model=root, expected=_listed(expected)) if data_root and root else "")


@rule(Call, "feature-count")
def estimator_feature_count(site: Call) -> None:
    """``model.predict(X)`` where ``X`` has a different number of columns than
    the data ``model`` was fitted on.

    Not for composite estimators: a ColumnTransformer picks its columns from
    a DataFrame by name, and ignores any others.
    """

    estimator = site.receiver
    if site.method not in _FEATURE_CHECKED_METHODS or not _validates_plain_input(estimator) or not site.args:
        return
    expected = vars(estimator).get("n_features_in_")
    data, data_root = site.args[0], site.arg_roots[0]
    if type(expected) is not int:
        return
    if is_ndarray(data) and data.ndim == 2:
        columns = data.shape[1]
    elif is_frame(data):
        columns = data.shape[1]
    else:
        return
    if columns != expected:
        root = site.receiver_root
        site.crash(
            "ValueError",
            f"X has {columns} features, but {type(estimator).__name__} is expecting "
            f"{expected} features as input.",
            [data_root, root],
            text("checker.feature_count", data=data_root, columns=columns, model=root, expected=expected)
            if data_root and root else "",
        )


@rule(Call, "sample-count")
def estimator_fit_sample_count(site: Call) -> None:
    """``model.fit(X, y)`` for a classifier or regressor, with ``X`` and ``y``
    of different lengths."""

    base = sys.modules.get("sklearn.base")
    estimator = site.receiver
    if site.method != "fit" or not is_sklearn_estimator(estimator) or base is None:
        return
    if not (base.is_classifier(estimator) or base.is_regressor(estimator)):
        return
    x_len, y_len = sample_count(site.arg(0, "X")), sample_count(site.arg(1, "y"))
    if x_len is not None and y_len is not None and x_len != y_len:
        site.crash("ValueError",
                   f"Found input variables with inconsistent numbers of samples: [{x_len}, {y_len}]",
                   [site.arg_root(0, "X"), site.arg_root(1, "y")])


@rule(Call, "sample-count")
def train_test_split_sample_count(site: Call) -> None:
    """``train_test_split(X, y)`` with arrays of different lengths."""

    selection = sys.modules.get("sklearn.model_selection")
    if selection is None or site.func is not getattr(selection, "train_test_split", None):
        return
    lengths = [sample_count(a) for a in site.args]
    if len(lengths) >= 2 and all(n is not None for n in lengths) and len(set(lengths)) > 1:
        site.crash("ValueError", f"Found input variables with inconsistent numbers of samples: {lengths}",
                   site.arg_roots)


@rule(Call, "too-few-samples")
def kneighbors_more_than_fitted(site: Call) -> None:
    """``knn.predict(X)`` where ``n_neighbors`` is larger than the number of
    samples ``knn`` was fitted on."""

    estimator = site.receiver
    neighbors = sys.modules.get("sklearn.neighbors")
    if neighbors is None or type(estimator) not in (neighbors.KNeighborsClassifier, neighbors.KNeighborsRegressor):
        return
    if site.method not in ("predict", "predict_proba", "score") or not site.args:
        return
    fitted, wanted = vars(estimator).get("n_samples_fit_"), estimator.n_neighbors
    if type(fitted) is int and type(wanted) is int and wanted > fitted:
        site.crash("ValueError", f"Expected n_neighbors <= n_samples,  but n_samples = {fitted}, "
                   f"n_neighbors = {wanted}", [site.receiver_root])


def _listed(names: List[str], limit: int = 8) -> str:
    shown = ", ".join(names[:limit])
    return shown + (", ..." if len(names) > limit else "")


# --- targets -----------------------------------------------------------------


@rule(Call, "continuous-target")
def classification_metric_on_continuous(site: Call) -> None:
    """``f1_score(y_true, scores)`` where ``scores`` are continuous, such as
    a regressor's predictions. Reads every value, within the scan limit."""

    metrics = sys.modules.get("sklearn.metrics")
    if metrics is None or not any(site.func is getattr(metrics, n, None) for n in _CLASSIFICATION_METRICS):
        return
    y_true, y_pred = site.arg(0, "y_true"), site.arg(1, "y_pred")
    if not _small_targets(y_true, y_pred):
        return
    checker = getattr(sys.modules.get("sklearn.metrics._classification"), "_check_targets", None)
    if checker is None:
        return
    try:
        checker(y_true, y_pred)
    except ValueError as exc:
        site.crash("ValueError", str(exc), [site.arg_root(0, "y_true"), site.arg_root(1, "y_pred")])


@rule(Call, "continuous-target")
def classifier_fit_on_continuous(site: Call) -> None:
    """``LogisticRegression().fit(X, y)`` where ``y`` is continuous. Reads
    every value of ``y``, within the scan limit."""

    base = sys.modules.get("sklearn.base")
    estimator = site.receiver
    if site.method != "fit" or not is_sklearn_estimator(estimator) or base is None or not base.is_classifier(estimator):
        return
    y = site.arg(1, "y")
    if not _small_targets(y):
        return
    check = getattr(sys.modules.get("sklearn.utils.multiclass"), "check_classification_targets", None)
    if check is None:
        return
    try:
        check(y)
    except ValueError as exc:
        site.crash("ValueError", str(exc), [site.arg_root(1, "y"), site.receiver_root])


# Metrics that check their targets with ``_check_targets`` before anything else.
_CLASSIFICATION_METRICS = (
    "accuracy_score", "balanced_accuracy_score", "f1_score", "fbeta_score", "precision_score",
    "recall_score", "precision_recall_fscore_support", "confusion_matrix", "classification_report",
    "jaccard_score", "matthews_corrcoef", "hamming_loss", "zero_one_loss", "cohen_kappa_score",
)


def _small_targets(*values) -> bool:
    """Plain arrays of targets, within the scan limit together."""

    total = 0
    for value in values:
        if is_ndarray(value):
            if value.dtype.kind == "O":
                return False
        elif not (is_series(value) or (type(value) is list and all(type(v) in SCALARS for v in value))):
            return False
        total += len(value)
    return total <= scan_budget()


# --- values the estimator cannot take ----------------------------------------


@rule(Call, "non-finite-data")
def estimator_rejects_nan(site: Call) -> None:
    """``model.fit(X, y)`` or ``model.predict(X)`` where ``X`` holds NaN or
    infinity, for an estimator that does not accept missing values. Reads
    every value, within the scan limit."""

    estimator, data = site.receiver, site.arg(0, "X")
    if site.method not in _DATA_METHODS or not _validates_plain_input(estimator):
        return
    values = _numeric_values(data)
    tags = _input_tags(estimator)
    if values is None or tags is None or tags.allow_nan:
        return
    np = numpy()
    if np.isfinite(values).all():
        return
    message = (
        "Input X contains NaN." if np.isnan(values).any()
        else "Input X contains infinity or a value too large for dtype('float64')."
    )
    site.crash("ValueError", message, [site.arg_root(0, "X")], text("checker.contains_nan", name=site.arg_root(0, "X"))
               if site.arg_root(0, "X") else "")


@rule(Call, "bad-value")
def estimator_rejects_text(site: Call) -> None:
    """``model.fit(df, y)`` where a column holds text that is not a number,
    such as ``"male"``, for an estimator that only takes numbers. Reads every
    value of the text columns, within the scan limit.

    Each suspect value is confirmed by scikit-learn's own input validation,
    on a one-row copy of the data in the same dtype, so the check follows
    however the installed versions convert text.
    """

    estimator, data = site.receiver, site.arg(0, "X")
    if site.method not in _DATA_METHODS or not _validates_plain_input(estimator):
        return
    tags = _input_tags(estimator)
    if tags is None or not tags.numbers_only:
        return
    validation = sys.modules.get("sklearn.utils.validation")
    check_array = getattr(validation, "check_array", None)
    if check_array is None:
        return
    np = numpy()
    if is_frame(data):
        if not plain_axis(data.columns):
            return
        text_columns = [i for i in range(data.shape[1]) if _is_text_dtype(data.dtypes.iloc[i])]
        if not text_columns or sum(len(data) for _ in text_columns) > scan_budget():
            return
        candidates = (
            (value, i) for i in text_columns for value in data.iloc[:, i].array
        )
    elif is_ndarray(data) and data.dtype.kind in "US":
        if data.size > scan_budget():
            return
        candidates = ((v.item(), None) for v in data.ravel())
    else:
        return
    for value, column in candidates:
        if type(value) is not str:
            if type(value) not in SCALARS:
                return
            continue
        try:
            float(value)
            continue
        except ValueError:
            pass
        if column is None:
            sample = np.array([[value]], dtype=data.dtype)
        else:
            sample = pandas().DataFrame(
                {data.columns[column]: pandas().Series([value], dtype=data.dtypes.iloc[column])}
            )
        try:
            check_array(sample)
        except ValueError as exc:
            site.crash("ValueError", str(exc), [site.arg_root(0, "X")],
                       text("checker.text_in_numbers", value=repr(value)))
        return


def _is_text_dtype(dtype) -> bool:
    """object, or one of pandas' string dtypes (the default for text from
    pandas 3). Not category, which scikit-learn converts differently."""

    pd = pandas()
    if str(dtype) == "object":
        return True
    string_dtype = getattr(pd, "StringDtype", None) if pd is not None else None
    return string_dtype is not None and isinstance(dtype, string_dtype)


def _numeric_values(data):
    """All values of a numeric array or DataFrame, within the scan limit, else None."""

    np = numpy()
    if is_ndarray(data):
        if data.dtype.kind not in "fiub" or data.size > scan_budget():
            return None
        return data
    if is_frame(data):
        if data.size > scan_budget() or not all(dt.kind in "fiub" for dt in data.dtypes):
            return None
        return data.to_numpy(dtype=np.float64)
    return None


# --- arguments ---------------------------------------------------------------


@rule(Call, "bad-argument")
def estimator_bad_argument(site: Call) -> None:
    """``LogisticRegression(n_estimators=3)``: an argument the estimator does
    not take. scikit-learn estimators declare every parameter explicitly."""

    if not is_sklearn_estimator_class(site.func):
        return
    try:
        inspect.signature(site.func).bind(*site.args, **site.kwargs)
    except TypeError as exc:
        site.crash("TypeError", f"{site.func.__name__}.__init__() {exc}", [])


@rule(Call, "bad-argument")
def sklearn_parameter_constraints(site: Call) -> None:
    """``train_test_split(X, test_size=1.5)`` and any other scikit-learn
    function that validates its parameters (from 1.2 on): a plain value
    outside the function's declared constraints. Checked with scikit-learn's
    own validator, which the function runs before anything else."""

    func = site.func
    constraints = getattr(func, "_skl_parameter_constraints", None)
    validator = getattr(sys.modules.get("sklearn.utils._param_validation"), "validate_parameter_constraints", None)
    sklearn = sys.modules.get("sklearn")
    if not isinstance(constraints, dict) or validator is None or sklearn is None:
        return
    try:
        if sklearn.get_config().get("skip_parameter_validation"):
            return
        target = signature_target(func)
        bound = inspect.signature(target or func, follow_wrapped=False).bind_partial(*site.args, **site.kwargs)
    except Exception:
        return
    # Only plain values: checking an array or estimator against a constraint
    # would look at objects whose meaning the checker does not model.
    plain = {k: v for k, v in bound.arguments.items() if type(v) in SCALARS and k in constraints}
    if not plain:
        return
    errors = sys.modules.get("sklearn.utils._param_validation")
    invalid = getattr(errors, "InvalidParameterError", ValueError)
    try:
        validator({k: constraints[k] for k in plain}, plain, caller_name=getattr(func, "__qualname__", "function"))
    except invalid as exc:
        site.crash(type(exc).__name__, str(exc), [])


@rule(Call, "bad-argument")
def train_test_split_sizes(site: Call) -> None:
    """``train_test_split(X, test_size=1.5)``: a test or train size that is
    neither a fraction in (0, 1) nor a count below the number of samples."""

    split = sys.modules.get("sklearn.model_selection._split")
    selection = sys.modules.get("sklearn.model_selection")
    if split is None or selection is None or site.func is not getattr(selection, "train_test_split", None):
        return
    validate = getattr(split, "_validate_shuffle_split", None)
    lengths = [sample_count(a) for a in site.args]
    if validate is None or not lengths or any(n is None for n in lengths) or len(set(lengths)) != 1:
        return
    test_size, train_size = site.kwargs.get("test_size"), site.kwargs.get("train_size")
    if not all(v is None or type(v) in (int, float) for v in (test_size, train_size)):
        return
    try:
        validate(lengths[0], test_size, train_size, default_test_size=0.25)
    except ValueError as exc:
        site.crash("ValueError", str(exc), [])

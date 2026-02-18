import inspect
import re
import site
import os, json
from collections import defaultdict
import config

config_path = config.sum_rule_config_path
with open(config_path) as f:
    config = json.load(f)

# Feature categorization for ablation study
FEATURE_CATEGORIES = {
    # Structural features
    # Features that describe the size, shape, or count of objects/datasets.
    'structural': {
        'shape', 'length', 'n_samples', 'dataset_shape', 'data_shape', 'target_shape',
        'sample_shapes', 'batch_shapes', 'image_shape', 'target_size', 'n_examples',
        'num_batches', 'n_epochs', 'samples', 'batch_size'
    },
    
    # Representation features
    # Features that describe type, dtype, module, or device-level representation of objects.  
    'representation': {
        'type', 'dtype', 'dtypes_summary', 'key_type', 'value_type', 'classes_dtype',
        'module', 'device', 'requires_grad', 'element_spec'
    },
    
    # Value features
    # Features that describe the actual content, values, and task-related properties.
    'value': {
        'value_info', 'has_nan', 'value_range', 'num_unique', 'unique_values', 
        'example_values', 'detected', 'metrics', 'value_repr', 'preview', 
        'class_mode', 'num_classes', 'n_classes', 'class_name', 'final_values',
        'keys', 'per_column', 'value'
    },
    
    # State features 
    # internal configuration of runtime objects
    # -> now we set it to always belong to Representation features
    'state': {
        'is_fitted', 'fitted_attributes'
    }
}

# Create reverse mapping for faster lookups
FEATURE_TO_CATEGORY = {}
for category, features in FEATURE_CATEGORIES.items():
    for feature in features:
        FEATURE_TO_CATEGORY[feature] = category

def get_excluded_categories():
    """Get list of excluded feature categories from config."""
    ablation_settings = config.get("ablation_settings", {})
    excluded_categories = []
    for category in ['structural', 'representation', 'value']:
        if not ablation_settings.get(category, True):  # Default to True (enabled), so False means excluded
            excluded_categories.append(category)
    
    # state follows representation
    if not ablation_settings.get('representation', True):
        excluded_categories.append('state')
    
    return excluded_categories

def filter_features_by_category(summary_dict, excluded_categories=None):
    """
    Filter summary dictionary by excluding specified feature categories.
    Only filters actual feature keys (second-level), not variable names (top-level).
    
    Args:
        summary_dict: Dictionary containing extracted features
        excluded_categories: List of categories to exclude. If None, uses config.
        
    Returns:
        Filtered dictionary excluding features from specified categories.
        Empty nested dictionaries are removed entirely.
    """
    if summary_dict is None:
        return summary_dict
        
    exclude_categories = excluded_categories if excluded_categories is not None else get_excluded_categories()
    
    # If nothing should be excluded, return original dict as-is
    if not exclude_categories:
        return summary_dict
    
    # If all categories are excluded, return None
    if set(exclude_categories) >= {'structural', 'representation', 'value', 'state'}:
        return None
    
    def _recursive_filter(obj, is_top_level=True):
        if not isinstance(obj, dict):
            return obj
            
        filtered_dict = {}
        
        for key, value in obj.items():
            # Always preserve execution metadata regardless of level
            if key == 'execution_cell_source':
                filtered_dict[key] = value
                continue
                
            if is_top_level:
                # Top-level keys are variable names - always include them but filter their contents
                if isinstance(value, dict):
                    filtered_nested = _recursive_filter(value, is_top_level=False)
                    # Include variable if it has ANY content after filtering (be more permissive)
                    if filtered_nested:
                        filtered_dict[key] = filtered_nested
                else:
                    # Include non-dict values at top level
                    filtered_dict[key] = value
            else:
                # Second-level and deeper - apply category filtering to actual feature keys
                # Handle dynamic per-column keys that start with 'per_column'
                lookup_key = key
                if key.startswith('per_column'):
                    lookup_key = 'per_column'
                    
                category = FEATURE_TO_CATEGORY.get(lookup_key)
                
                if category and category in exclude_categories:
                    # Feature is in an excluded category - skip it
                    pass
                elif category is None:
                    # Unknown feature key - could be a nested structure, explore it
                    if isinstance(value, dict):
                        # Recursively filter unknown nested structures
                        filtered_nested = _recursive_filter(value, is_top_level=False)
                        # Only include if it contains valid features after filtering
                        if filtered_nested:
                            filtered_dict[key] = filtered_nested
                    else:
                        # Unknown scalar - include by default (more permissive for unknown features)
                        filtered_dict[key] = value
                else:
                    # Known feature NOT in excluded categories - include it
                    if isinstance(value, dict):
                        # For nested dicts, recursively filter
                        filtered_nested = _recursive_filter(value, is_top_level=False)
                        if filtered_nested:
                            filtered_dict[key] = filtered_nested
                    else:
                        # Include scalar values directly
                        filtered_dict[key] = value
                        
        return filtered_dict
    
    result = _recursive_filter(summary_dict, is_top_level=True)
    return result if result else None

def filter_all_category_combinations(summary_dict):
    """
    Generate all possible category combinations (s, r, v, s_r, s_v, r_v) from a summary dict.
    This gives 100% accuracy by working directly with the dictionary instead of parsing text.
    
    Args:
        summary_dict: Dictionary containing extracted features from notebook execution
        
    Returns:
        Dictionary with keys like 's', 'r', 'v', 's_r', 's_v', 'r_v' containing filtered results
    """
    if summary_dict is None:
        return {}
    
    # Define all category combinations by what to EXCLUDE
    # For 's' (structural only): exclude representation, value, state
    # For 'r' (representation + state only): exclude structural, value  
    # etc.
    exclude_combinations = {
        's': ['representation', 'value', 'state'],           # keep only structural
        'r': ['structural', 'value'],                        # keep representation + state
        'v': ['structural', 'representation', 'state'],      # keep only value
        's_r': ['value'],                                    # keep structural + representation + state
        's_v': ['representation', 'state'],                  # keep structural + value
        'r_v': ['structural']                                # keep representation + value + state
    }
    
    results = {}
    
    # Generate filtered version for each combination
    for abbrev, excluded_categories in exclude_combinations.items():
        filtered_dict = filter_features_by_category(summary_dict, excluded_categories=excluded_categories)
        # print(f"DEBUG: {abbrev} (exclude {excluded_categories}) -> {type(filtered_dict)} with {len(filtered_dict) if filtered_dict else 0} items")
        # Include result if it's not None and has actual content
        if filtered_dict is not None and len(filtered_dict) > 0:
            results[abbrev] = filtered_dict
    
    return results

# rules about callable objects ----------- start here -----------------
def _is_user_defined_function(func):
    """Check if a function is defined in notebook or outside site-packages."""
    try:
        src_file = inspect.getsourcefile(func)

        # Notebook cell: e.g., "<ipython-input-5-abc123>"
        if (src_file is None) or re.match(r"<ipython-input-\d+-.*>", src_file):
            return True

        # Check for local script (non-site-packages)
        src_file = os.path.abspath(src_file)
        for site_path in site.getsitepackages() + [site.getusersitepackages()]:
            if src_file.startswith(os.path.abspath(site_path)):
                return False
        return True
    except Exception:
        return False

# test user defined function calls: torch_2/torch_2_reproduced.ipynb
def summarize_callable(val, namespace=None, name=None):
    if not callable(val):
        return None
    summary = {}
    try:
        # if (inspect.isfunction(val) or inspect.ismethod(val)):
        if config["other"].get("summarize_callable_signature", False):
            sig = get_signature(val)
            if sig:
                summary['signature'] = serialize_signature(sig)
        if config["other"].get("summarize_callable_doc_all", False):
            summary['doc'] = get_full_doc_structured(val)
        else:
            if config["other"].get("summarize_callable_doc_summary", False):
                summary['doc_summary'] = get_short_doc(val, config["other"].get("summarize_callable_doc_summary_maxchars", 500))
            if config["other"].get("summarize_callable_doc_importants", False):
                important_secs = get_important_doc(val)
                if important_secs:
                    summary['doc_importants'] = important_secs
        if inspect.ismethod(val) and config["other"].get("summarize_callable_context_attributes", False):
            summary["context"] = get_method_context(val)
        if config["other"].get("summarize_callable_sourcecode_all", False): # should not be useful
                summary['source'] = inspect.getsource(val)
        elif config["other"].get("summarize_callable_sourcecode_userdefined", False): # should not be useful
            if _is_user_defined_function(val):
                summary['source'] = inspect.getsource(val)
    except Exception as e:
        if config["other"].get("summarize_callable_sourcecode", False):
            if name and namespace and f'__source__{name}' in namespace:
                summary['source'] = namespace[f'__source__{name}']
    return summary

def get_signature(callable_obj):
    try:
        return inspect.signature(callable_obj)
    except (ValueError, TypeError):
        return None

def serialize_signature(sig: inspect.Signature):
    params = []
    for p in sig.parameters.values():
        params.append({
            "name": p.name,
            "kind": str(p.kind),  # POSITIONAL_ONLY, VAR_KEYWORD, etc.
            "required": p.default is inspect._empty,
            "default": None if p.default is inspect._empty else repr(p.default),
            "annotation": (
                None if p.annotation is inspect._empty else repr(p.annotation)
            )
        })
    return {
        "parameters": params,
        "return_annotation": (
            None if sig.return_annotation is inspect._empty
            else repr(sig.return_annotation)
        )
    }

def get_short_doc(obj, max_chars=300):
    doc = parse_numpy_doc(obj)
    if not doc["summary"]:
        return None
    if isinstance(max_chars, int) and len(doc["summary"]) > max_chars:
        return doc["summary"][:max_chars] + "..."
    return doc["summary"]

def get_important_doc(obj):
    doc = parse_numpy_doc(obj)
    return {
        k: v
        for k, v in doc["sections"].items()
        if k in config["IMPORTANT_DOC_SECTIONS"]
    }

def get_full_doc_structured(obj):
    return parse_numpy_doc(obj)

def is_section_header(lines, i):
    if i + 1 >= len(lines):
        return False

    title = lines[i].strip()
    underline = lines[i + 1].strip()

    return (
        title
        and underline
        and all(c == "-" for c in underline)
        and len(underline) >= len(title)
    )

def parse_numpy_doc(obj):
    doc = inspect.getdoc(obj)
    if not doc:
        return {"summary": None, "sections": {}}

    lines = doc.splitlines()
    n = len(lines)
    i = 0

    summary_lines = []
    sections = {}

    # ---- summary ----
    while i < n:
        if is_section_header(lines, i):
            break
        if lines[i].strip():
            summary_lines.append(lines[i].strip())
        i += 1

    summary = " ".join(summary_lines) if summary_lines else None

    # ---- sections ----
    while i < n:
        if not is_section_header(lines, i):
            i += 1
            continue

        title = lines[i].strip().lower()
        i += 2  # skip title + underline
        body = []

        while i < n and not is_section_header(lines, i):
            body.append(lines[i])
            i += 1

        # normalize whitespace but preserve full sentences
        text = " ".join(
            line.strip() for line in body if line.strip()
        )

        sections[title] = text

    return {
        "summary": summary,
        "sections": sections
    }


def get_method_context(method):
    if not inspect.ismethod(method):
        return None

    self_obj = method.__self__
    cls = type(self_obj)

    return {
        # "self_type": f"{cls.__module__}.{cls.__name__}",
        "public_attributes": sorted(
            name for name in vars(self_obj).keys()
            if not name.startswith("_")
        )
    }
# rules about callable objects ----------- end here -----------------

def summarize_collection(val):
    if isinstance(val, (list, tuple, dict, set)):
        summary = {'length': len(val)}
        return filter_features_by_category(summary)
    return None

def summarize_primitive(val):
    if isinstance(val, (int, float, str, bool)):
        summary = {'value': val}
        return filter_features_by_category(summary)
    return None

# 1d array / series
def summarize_value_range(val):
    import numpy as np
    import pandas as pd

    try:
        # Convert tensors or lists to numpy array
        if hasattr(val, "detach"):  # torch.Tensor
            val = val.detach().cpu().numpy()
        elif hasattr(val, "numpy"):  # numpy array or pandas series
            val = val.numpy()
        elif isinstance(val, list):
            val = np.array(val)

        # Flatten if possible
        if hasattr(val, "ndim") and val.ndim > 1:
            return None  # Too ambiguous to summarize multidimensional arrays

        # Handle pandas Series and numpy arrays
        if isinstance(val, (pd.Series, np.ndarray)):
            if isinstance(val, pd.Series):
                data = val.dropna()
            else:
                data = pd.Series(val).dropna()
            summary = {"value_info":{}}
            if data.empty:
                summary["value_info"] = {"value_type": "empty or all NaN"}

            unique_vals = data.unique()
            num_unique = len(unique_vals)

            if pd.api.types.is_numeric_dtype(data):
                # Binary (e.g. only 0 and 1)
                if num_unique == 2 and set(unique_vals).issubset({0, 1}):
                    summary["value_info"] = {
                        "value_type": "binary",
                        "unique_values": sorted(unique_vals.tolist())
                    }

                # Categorical (numeric with small number of unique values)
                elif num_unique <= 10 and pd.api.types.is_integer_dtype(data):
                    summary["value_info"] = {
                        "value_type": "categorical numeric(no more than 10 unique values)",
                        "num_unique": num_unique,
                    }
                    if num_unique <= 5:
                        summary["value_info"]["unique_values"] = sorted(unique_vals.tolist())
                else:
                    summary["value_info"] = {
                        "value_type": "continuous",
                        "value_range": (data.min(), data.max())
                    }

            else:  # Non-numeric categorical
                summary["value_info"] = {
                    "value_type": "categorical or object",
                    "num_unique": len(unique_vals),
                }
                if len(unique_vals) <= 5:
                    summary["value_info"]["unique_values"] = unique_vals.tolist()
            return filter_features_by_category(summary)

    except Exception:
        pass

    return None

# test pandas series: statsmodels_2_reproduced
def summarize_pandas_series(val):
    import pandas as pd
    import numpy as np

    if not isinstance(val, pd.Series):
        return None

    cls = type(val)
    mod = cls.__module__
    summary = {
        "type": f"{mod}.{cls.__name__}",
        "dtype": str(val.dtype),
        "length": len(val),
    }

    # Check for missing values
    summary["has_nan"] = bool(val.isna().any())

    return filter_features_by_category(summary)

# test np.array: tensorflow_4/tensorflow_4_reproduced.ipynb
def summarize_numpy_array(val):
    import numpy as np

    if not isinstance(val, np.ndarray):
        return None

    cls = type(val)
    mod = cls.__module__
    summary = {
        "type": f"{mod}.{cls.__name__}",
        "shape": val.shape,
        "dtype": str(val.dtype),

    }
    try:
        summary["has_nan"] = bool(np.isnan(val).any())
    except TypeError:
        pass
    if np.issubdtype(val.dtype, np.number):
        if val.size > 0:
            summary["value_range"] = (np.nanmin(val), np.nanmax(val))
        else:
            summary["value_range"] = "empty array"
    return filter_features_by_category(summary)

# test pd.series: pandas_3/pandas_3_reproduced.ipynb
# test pd.dataframe: pandas_3/pandas_3_reproduced.ipynb
def _truncate(val, max_len=20):
    if isinstance(val, str) and len(val) > max_len:
        return val[:max_len] + "..."
    return val
def summarize_dataframe(df):
    import pandas as pd
    import numpy as np

    if not isinstance(df, pd.DataFrame):
        return None

    cls = type(df)
    mod = cls.__module__
    summary = {
        "type": f"{mod}.{cls.__name__}",
        "shape": df.shape,
        "has_nan": df.isnull().values.any()
    }
    if config["other"].get("summarize_dataframe_per_column", False) == False:
        dtypes = df.dtypes.astype(str)
        summary["dtypes_summary"] = dtypes.value_counts().to_dict()
    else:
        max_columns = config["other"].get("summarize_dataframe_per_column_maxcolumns", 20)
        if df.shape[1] > max_columns:
            sum_col_name = "per_column_max_{}".format(max_columns)
        else:
            sum_col_name = "per_column"
        summary[sum_col_name] = {}
        for col in df.columns[:max_columns]:
            col_data = df[col].dropna()
            unique_vals = col_data.unique()
            num_unique = len(unique_vals)
            dtype_str = str(df[col].dtype)

            if pd.api.types.is_numeric_dtype(col_data):
                is_integer = pd.api.types.is_integer_dtype(col_data)
                if num_unique == 2 and set(unique_vals).issubset({0, 1}):
                    col_type = "binary"
                elif is_integer and num_unique <= 10:
                    col_type = "categorical_numeric"
                else:
                    col_type = "continuous"

                summary[sum_col_name][col] = {
                    "dtype": dtype_str,
                    "type": col_type,
                    "num_unique": num_unique,
                    "value_range": (col_data.min(), col_data.max()),
                }
            else:
                col_type = "categorical"
                col_summary = {
                    "dtype": dtype_str,
                    "type": col_type,
                    "num_unique": num_unique
                }

                if num_unique <= 5:
                    col_summary["unique_values"] = [_truncate(v) for v in unique_vals.tolist()[:5]]
                else:
                    # Sample up to 5 random unique values for illustration
                    sampled_vals = np.random.choice(unique_vals, size=min(5, num_unique), replace=False)
                    col_summary["example_values"] = [_truncate(v) for v in sampled_vals]
                summary[sum_col_name][col] = col_summary

    return filter_features_by_category(summary)

# test keras history: tensorflow_6/tensorflow_6_reproduced.ipynb
def summarize_dict(val):
    if not isinstance(val, dict):
        return None

    summary = {
        "type": str(type(val)),
        "length": len(val),
    }

    keys = list(val.keys())

    # Heuristic 1: tf.keras History dict (metric name → list of floats)
    try:
        values = list(val.values())
        if (
            values and
            all(isinstance(v, list) and all(isinstance(x, (int, float)) for x in v) for v in values)
        ):
            n_epochs = len(values[0])
            if all(len(v) == n_epochs for v in values):
                summary["detected"] = "training_history_dict"
                summary["metrics"] = keys
                summary["n_epochs"] = n_epochs
                # summary["final_values"] = {k: v[-1] for k, v in val.items()}
                return filter_features_by_category(summary)
    except Exception:
        pass

    # Heuristic 2: sklearn.datasets.load_* output
    if {"data", "target"}.issubset(val.keys()):
        summary["detected"] = "sklearn_dataset_dict"
        summary["keys"] = keys
        summary["data_shape"] = getattr(val.get("data"), "shape", None)
        summary["target_shape"] = getattr(val.get("target"), "shape", None)
        return filter_features_by_category(summary)

    # Heuristic 3: flat scalar metrics dict
    if all(isinstance(v, (int, float)) for v in val.values()):
        summary["detected"] = "scalar_metrics_dict"
        summary["metrics"] = keys
        return filter_features_by_category(summary)

    # Generic shallow preview
    preview = []
    try:
        for i, (k, v) in enumerate(val.items()):
            if i >= 5:
                break
            try:
                key_str = str(k)
                val_repr = repr(v)
                preview.append({
                    'key': key_str[:30] + '...' if len(key_str) > 30 else key_str,
                    'key_type': str(type(k)),
                    'value_repr': val_repr[:50] + '...' if len(val_repr) > 50 else val_repr,
                    'value_type': str(type(v)),
                })
            except Exception:
                continue
        summary["preview"] = preview
    except Exception:
        pass

    return filter_features_by_category(summary)

#--------------------------------tensorflow/keras----------------------------------
# test: tensorflow_2/tensorflow_2_reproduced.ipynb
def summarize_directory_iterator(val):
    cls = type(val)
    if cls.__name__ == "DirectoryIterator" and "keras" in cls.__module__: # if isinstance(val, DirectoryIterator):
        summary = {
            'n_samples': getattr(val, 'n', None),
            'num_classes': getattr(val, 'num_classes', None),
            'batch_size': getattr(val, 'batch_size', None),
            'image_shape': getattr(val, 'image_shape', None),
            'target_size': getattr(val, 'target_size', None)
            # 'class_indices': val.class_indices,
            # 'shuffle': val.shuffle,
            # 'color_mode': val.color_mode,
            # 'directory': val.directory,
            # 'example_filenames': val.filenames[:5]  # show first 5
        }
        return filter_features_by_category(summary)
    return None

# test: tensorflow_1/tensorflow_1_reproduced.ipynb
def summarize_tf_dataset(val):
    cls = type(val)
    if cls.__name__.endswith("Dataset") and cls.__module__.startswith("tensorflow."): # if isinstance(val, ImageDataGenerator): # if isinstance(val, tf.data.Dataset):
        summary = {
            "type": f"{cls.__module__}.{cls.__name__}",
        }
        try:
            summary["element_spec"] = repr(val.element_spec)
        except Exception:
            pass
        return filter_features_by_category(summary)
    return None

# test: tensorflow_6/tensorflow_6_reproduced.ipynb
# def summarize_tensorflow_history(val):
#     cls = type(val)
#     if cls.__name__ == "History" and cls.__module__.startswith("keras.callbacks"): # if not isinstance(val, tf.keras.callbacks.History):
#         try:
#             history = getattr(val, 'history', {})
#             return {
#                 'type': f"{cls.__module__}.{cls.__name__}",
#                 'metrics': list(history.keys())
#             }
#         except Exception:
#             pass
#     return None

# test: tensorflow_11/tensorflow_11_reproduced.ipynb
def summarize_dataframe_iterator(val, name=None):
    # Check by class name since importing DataFrameIterator may fail
    if type(val).__name__ != "DataFrameIterator":
        return None

    summary = {
        "type": f"{type(val).__module__}.{type(val).__name__}",
        "n_samples": getattr(val, "n", None),             # sometimes 'n' or 'samples' attribute
        "samples": getattr(val, "samples", None),
        "batch_size": getattr(val, "batch_size", None),
        "image_shape": getattr(val, "image_shape", None),
        "class_mode": getattr(val, "class_mode", None),
        "num_classes": len(getattr(val, "class_indices", {})) if hasattr(val, "class_indices") else None,
    }

    return filter_features_by_category(summary)

#--------------------------------torch----------------------------------
# def summarize_torch_tensor(val):
#     if hasattr(val, 'device'):
#         return {
#             'device': str(getattr(val, 'device', None)),
#             'requires_grad': getattr(val, 'requires_grad', None)
#         }
#     return None

# test: torch_7/torch_7_reproduced.ipynb
def summarize_torch_tensor(val):
    cls = type(val)
    if cls.__name__ != "Tensor" or "torch" not in cls.__module__:
        return None

    summary = {
        "type": str(cls),
        "shape": tuple(val.shape) if hasattr(val, "shape") else None,
        "dtype": str(getattr(val, "dtype", None)),
        "device": str(getattr(val, "device", None)),
        "requires_grad": getattr(val, "requires_grad", None),
    }

    # check for NaNs or Infs (can be slow on large tensors)
    try:
        if hasattr(val, "isnan") and callable(val.isnan):
            summary["has_nan"] = bool(val.isnan().any().item())
        # if hasattr(val, "isinf") and callable(val.isinf):
        #     summary["has_inf"] = bool(val.isinf().any().item())
    except Exception:
        pass

    return filter_features_by_category(summary)

# test: torch_2/torch_2_reproduced.ipynb
# no shape, dtype info without executing: next(iter(val))
def summarize_pytorch_dataloader(val, name=None):
    cls = type(val)
    if cls.__name__ == "DataLoader" and cls.__module__.startswith("torch.utils.data"): # if not isinstance(val, DataLoader):
        summary = {
            "type": f"{cls.__module__}.{cls.__name__}",
            "num_batches": len(val),
            "num_examples": len(val.dataset) if hasattr(val, "dataset") else None,
            "batch_size": getattr(val, "batch_size", None),
            "dataset": _pytorch_dataloader_dataset(val.dataset if hasattr(val, "dataset") else None, val.collate_fn if hasattr(val, "collate_fn") else None)
            # "shuffle": getattr(val, "shuffle", None),
            # "num_workers": getattr(val, "num_workers", None),
            # "pin_memory": getattr(val, "pin_memory", None),
            # "drop_last": getattr(val, "drop_last", None),
            # "dataset_type": str(type(val.dataset)) if hasattr(val, "dataset") else None,
        }
        return filter_features_by_category(summary)
    return None

# test subset: torchvision_1_reproduced
def summarize_pytorch_subset(val, name=None, num_samples=10):
    from torch.utils.data import Subset

    if not isinstance(val, Subset):
        return None

    cls = type(val)
    mod = cls.__module__

    summary = {
        "type": f"{mod}.{cls.__name__}",
        "length": len(val),
        # "indices_count": len(val.indices),
        # "indices_range": (min(val.indices), max(val.indices)) if val.indices else None,
        "underlying_dataset": _pytorch_dataloader_dataset(val.dataset),
    }

    return filter_features_by_category(summary)

from collections import defaultdict

def _pytorch_dataloader_dataset(dataset, collate_fn=None, num_samples=10):
    if dataset is None:
        return

    # ---- Dataset-level info from vars() ----
    dataset_vars = vars(dataset)
    field_keys = list(dataset_vars.keys())
    dataset_info = {}

    for key in field_keys:
        val = dataset_vars[key]
        cls = type(val)
        if (cls.__name__ == "Tensor") and ("torch" in cls.__module__):
            dataset_info[key] = {
                "dataset_shape": tuple(val.shape),
                "dtype": val.dtype,
            }
        else:
            dataset_info[key] = {"type": cls.__name__}

    # ---- Sample-level shape summary ----
    shape_summary = defaultdict(set)
    collected_samples = []

    for i in range(min(num_samples, len(dataset))):
        try:
            sample = dataset[i]
            cls = type(sample)
            collected_samples.append(sample)
        except Exception as e:
            print(f"Skipping index {i}: {e}")
            continue

        # Handle tuple (common in __getitem__)
        if isinstance(sample, (tuple, list)):
            for j, value in enumerate(sample):
                key = "input" if j == 0 else "label" if j == 1 else f"field[{j}]"
                if (type(value).__name__ == "Tensor") and ("torch" in type(value).__module__): # if isinstance(value, torch.Tensor):
                    shape_summary[key].add(value.shape)
                elif hasattr(value, 'size') and callable(value.size): # PIL image
                    shape_summary[key].add(f"PIL({value.size[0]}x{value.size[1]})")
                else:
                    shape_summary[key].add(type(value).__name__)

        # Handle dict outputs
        elif isinstance(sample, dict):
            for key, value in sample.items():
                shape = getattr(value, 'shape', type(value).__name__)
                shape_summary[key].add(shape)

        # Handle single tensors
        elif (cls.__name__ == "Tensor") and ("torch" in cls.__module__): #elif isinstance(sample, torch.Tensor):
            shape_summary["data"].add(sample.shape)

        # Handle objects with attributes
        elif hasattr(sample, "__dict__"):
            for key, value in vars(sample).items():
                shape = getattr(value, 'shape', type(value).__name__)
                shape_summary[key].add(shape)

        else:
            shape_summary["unknown"].add(type(sample).__name__)

    # ---- Optional: Simulate batch via collate_fn ----
    batch_shapes = {}
    if collate_fn and collected_samples:
        try:
            batch = collate_fn(collected_samples)
            if isinstance(batch, dict):
                for k, v in batch.items():
                    shape = getattr(v, 'shape', type(v).__name__)
                    batch_shapes[f"batch.{k}"] = shape
            elif isinstance(batch, (tuple, list)):
                for i, v in enumerate(batch):
                    shape = getattr(v, 'shape', type(v).__name__)
                    batch_shapes[f"batch[{i}]"] = shape
            elif (type(batch).__name__ == "Tensor") and ("torch" in type(batch).__module__): #isinstance(batch, torch.Tensor):
                batch_shapes["batch"] = batch.shape
            else:
                batch_shapes["batch"] = type(batch).__name__
        except Exception as e:
            batch_shapes["error"] = f"collate_fn failed: {e}"

    return {
        "dataset_info": dataset_info,
        "sample_shapes": shape_summary,
        "batch_shapes": batch_shapes
    }
  

#--------------------------------sklearn----------------------------------
# test: pandas_3/pandas_3_reproduced.ipynb
def summarize_label_encoder(val):
    cls = type(val)
    if cls.__name__ == "LabelEncoder" and "sklearn.preprocessing" in cls.__module__:
        summary = {
            "type": f"{cls.__module__}.{cls.__name__}"
        }
        if hasattr(val, "classes_"):
            classes = val.classes_
            summary["n_classes"] = len(classes)
            summary["classes_dtype"] = getattr(classes, "dtype", None)
            # summary["classes"] = classes.tolist()[:10]  # show up to 10 classes
        return filter_features_by_category(summary)
    return None

# test: sklearn_2_reproduced.ipynb
def summarize_sklearn_model(model):
    cls = type(model)
    # Check if it is a sklearn estimator by module name
    if not hasattr(cls, "__module__") or not cls.__module__.startswith("sklearn."):
        return None
    from sklearn.utils.validation import check_is_fitted
    import sklearn
    
    summary = {
        "type": str(cls),
        "class_name": cls.__name__,
        "module": cls.__module__,
        "is_fitted": False,
    }
    
    # Check if fitted
    try:
        check_is_fitted(model)
        summary["is_fitted"] = True
    except Exception:
        summary["is_fitted"] = False
        
    # If fitted, gather key learned info for tree models
    if summary["is_fitted"]:
        summary["fitted_attributes"] = {}
        try:
            summary["fitted_attributes"]["n_features_in_"] = getattr(model, "n_features_in_", None)
            summary["fitted_attributes"]["n_outputs_"] = getattr(model, "n_outputs_", None)
        except Exception:
            # In case something fails, just skip
            pass
            
    return filter_features_by_category(summary)




# def filter_runinfo_files(input_folder="executed_code_runinfo_full"):
#     """
#     Filter extracted runinfo text files based on excluded feature categories.
#     Reads files from input_folder, filters the runinfo content, and writes 
#     filtered versions to folders named with category abbreviations.
    
#     Args:
#         input_folder: Folder containing the original runinfo .txt files
#     """
    
#     # Get excluded categories and create output folder names
#     excluded_categories = get_excluded_categories()
#     if set(excluded_categories) >= {'structural', 'representation', 'value', 'state'}:
#         print("All categories excluded, skipping filtering.")
#         return
    
#     # Create category abbreviations for what's ENABLED (not excluded)
#     category_abbrev = {
#         'structural': 's',
#         'representation': 'r', 
#         'value': 'v',
#         'state': 'st'
#     }
    
#     # Create output folder name based on what's NOT excluded
#     all_categories = {'structural', 'representation', 'value'}
#     enabled_categories = list(all_categories - set(excluded_categories))
#     # Sort in desired order: structural, representation, value
#     category_order = ['structural', 'representation', 'value']
#     enabled_categories.sort(key=lambda x: category_order.index(x))
#     abbrev_list = [category_abbrev[cat] for cat in enabled_categories if cat in category_abbrev]
#     output_suffix = '_'.join(abbrev_list)
#     output_folder = input_folder.replace('_full', f'_{output_suffix}')
    
#     # Create output directory
#     os.makedirs(output_folder, exist_ok=True)
    
#     # Process each file in input folder recursively
#     if not os.path.exists(input_folder):
#         print(f"Input folder {input_folder} does not exist.")
#         return
        
#     processed_count = 0
#     for root, dirs, files in os.walk(input_folder):
#         for filename in files:
#             if not filename.endswith('.txt'):
#                 continue
                
#             input_path = os.path.join(root, filename)
            
#             # Preserve the relative directory structure
#             rel_path = os.path.relpath(root, input_folder)
#             if rel_path == '.':
#                 output_dir = output_folder
#             else:
#                 output_dir = os.path.join(output_folder, rel_path)
            
#             # Create output directory if it doesn't exist
#             os.makedirs(output_dir, exist_ok=True)
#             output_path = os.path.join(output_dir, filename)
            
#             try:
#                 with open(input_path, 'r', encoding='utf-8') as f:
#                     content = f.read()
                
#                 # Extract runinfo section
#                 # Convert excluded categories back to enabled for text filtering (legacy compatibility)
#                 all_categories = {'structural', 'representation', 'value', 'state'}
#                 enabled_categories = list(all_categories - set(excluded_categories))
#                 filtered_content = _filter_runinfo_text(content, enabled_categories)
                
#                 # Write filtered content
#                 with open(output_path, 'w', encoding='utf-8') as f:
#                     f.write(filtered_content)
                    
#                 processed_count += 1
                    
#             except Exception as e:
#                 print(f"Error processing {input_path}: {e}")
#                 continue
    
#     print(f"Filtered {processed_count} files from {input_folder} to {output_folder}")

# def _filter_runinfo_text(content, enabled_categories):
#     """
#     Filter the runinfo section of a text file based on enabled categories.
    
#     Args:
#         content: Full text content of the file
#         enabled_categories: List of enabled feature categories
        
#     Returns:
#         Filtered text content with only enabled category features
#     """
    
#     # Find the runinfo section
#     start_marker = "# Current relevent runtime information:"
#     end_marker = "# Target Cell:"
    
#     start_idx = content.find(start_marker)
#     end_idx = content.find(end_marker)
    
#     if start_idx == -1 or end_idx == -1:
#         return content  # Return original if markers not found
    
#     # Extract parts
#     before_runinfo = content[:start_idx + len(start_marker)]
#     runinfo_section = content[start_idx + len(start_marker):end_idx].strip()
#     after_runinfo = content[end_idx:]
    
#     # Try to parse runinfo as JSON/dict structure
#     try:
#         # Look for JSON-like structures in the runinfo section
#         filtered_runinfo = _filter_runinfo_content(runinfo_section, enabled_categories)
#     except Exception as e:
#         # If parsing fails, return original content
#         print(f"Failed to filter runinfo: {e}")
#         return content
    
#     # Reconstruct the content
#     if filtered_runinfo.strip():
#         return before_runinfo + "\n" + filtered_runinfo + "\n" + after_runinfo
#     else:
#         return before_runinfo + "\n\n" + after_runinfo

# def _filter_runinfo_content(runinfo_text, enabled_categories):
#     """
#     Filter runinfo content by parsing the dictionary and filtering based on categories.
    
#     Args:
#         runinfo_text: The runinfo section text (should be a Python dictionary)
#         enabled_categories: List of enabled feature categories
        
#     Returns:
#         Filtered runinfo text
#     """
#     import ast
#     import json
#     import numpy as np
    
#     try:
#         # Create a safe evaluation environment with common imports
#         safe_globals = {
#             '__builtins__': {},
#             'nan': float('nan'),
#             'inf': float('inf'),
#             'True': True,
#             'False': False,
#             'None': None,
#         }
        
#         # Add numpy types that might appear
#         try:
#             import numpy as np
#             safe_globals.update({
#                 'dtype': np.dtype,
#                 'int64': np.int64,
#                 'float64': np.float64,
#                 'object': object,
#                 'str': str,
#             })
#         except ImportError:
#             pass
        
#         # Use eval with safe globals
#         runinfo_dict = eval(runinfo_text.strip(), safe_globals, {})
        
#         if isinstance(runinfo_dict, dict):
#             # Filter each variable in the dictionary
#             filtered_dict = {}
            
#             for var_name, var_data in runinfo_dict.items():
#                 if isinstance(var_data, dict):
#                     # Filter the variable's data recursively
#                     filtered_var_data = _recursive_filter_dict(var_data, enabled_categories)
                    
#                     # Only include the variable if it has remaining data after filtering
#                     if filtered_var_data:
#                         filtered_dict[var_name] = filtered_var_data
#                 else:
#                     # For non-dict values, include them as-is (they're usually simple metadata)
#                     filtered_dict[var_name] = var_data
            
#             # Convert back to formatted string
#             if filtered_dict:
#                 # Format the dictionary nicely with proper indentation
#                 json_str = json.dumps(filtered_dict, indent=4, default=str)
#                 # Convert to Python syntax
#                 python_str = json_str.replace('"', "'").replace('null', 'None').replace('true', 'True').replace('false', 'False')
#                 return python_str
#             else:
#                 return ""  # No variables left after filtering
#         else:
#             # Not a dictionary, try manual filtering as fallback
#             return _filter_content_manually_basic(runinfo_text, enabled_categories)
            
#     except Exception as e:
#         # If parsing fails, try basic manual filtering instead of returning original
#         print(f"Failed to parse runinfo as dictionary: {e}")
#         return _filter_content_manually_basic(runinfo_text, enabled_categories)

# def _filter_content_manually_basic(content, enabled_categories):
#     """
#     Basic text-based filtering when dictionary parsing fails.
#     More permissive approach that preserves variable structures.
#     """
#     lines = content.split('\n')
#     filtered_lines = []
    
#     # Create list of prohibited features based on enabled categories
#     prohibited_features = set()
    
#     for feature, category in FEATURE_TO_CATEGORY.items():
#         if category not in enabled_categories:
#             prohibited_features.add(feature)
    
#     # Convert to patterns for regex matching
#     prohibited_patterns = []
#     for feature in prohibited_features:
#         prohibited_patterns.append(f"'{feature}'\\s*:")
#         prohibited_patterns.append(f'"{feature}"\\s*:')
    
#     import re
    
#     skip_mode = False
#     brace_level = 0
#     skip_start_level = 0
#     in_variable_block = False
#     current_variable_lines = []
    
#     i = 0
#     while i < len(lines):
#         line = lines[i]
#         original_line = line
#         stripped = line.strip()
        
#         # Track brace levels for nested structures
#         old_brace_level = brace_level
#         brace_level += stripped.count('{') - stripped.count('}')
        
#         # Detect variable declarations (top-level keys)
#         variable_match = re.match(r"^\\s*'([^']+)'\\s*:\\s*\\{", stripped)
#         if variable_match and brace_level > old_brace_level:
#             # Starting a new variable block
#             if current_variable_lines:
#                 # Process previous variable block
#                 variable_result = _process_variable_block(current_variable_lines, enabled_categories, prohibited_patterns)
#                 if variable_result:
#                     filtered_lines.extend(variable_result)
#                 current_variable_lines = []
            
#             # Start collecting new variable
#             current_variable_lines = [original_line]
#             in_variable_block = True
#             skip_mode = False
#             i += 1
#             continue
        
#         if in_variable_block:
#             current_variable_lines.append(original_line)
            
#             # Check if we're at the end of this variable (back to top level)
#             if brace_level == 0 or (brace_level == 1 and stripped.endswith(',')):
#                 # Process this variable block
#                 variable_result = _process_variable_block(current_variable_lines, enabled_categories, prohibited_patterns)
#                 if variable_result:
#                     filtered_lines.extend(variable_result)
#                 current_variable_lines = []
#                 in_variable_block = False
#                 skip_mode = False
#         else:
#             # Not in a variable block - include structural lines
#             filtered_lines.append(original_line)
        
#         i += 1
    
#     # Handle any remaining variable block
#     if current_variable_lines:
#         variable_result = _process_variable_block(current_variable_lines, enabled_categories, prohibited_patterns)
#         if variable_result:
#             filtered_lines.extend(variable_result)
    
#     return '\\n'.join(filtered_lines)

# def _process_variable_block(variable_lines, enabled_categories, prohibited_patterns):
#     """
#     Process a single variable block and filter its content.
#     Returns the filtered lines if the variable should be kept, None otherwise.
#     """
#     if not variable_lines:
#         return None
    
#     import re
    
#     # Check if this variable contains any features from enabled categories
#     content_text = '\\n'.join(variable_lines)
    
#     # Look for any enabled features in this variable
#     has_enabled_features = False
#     for feature, category in FEATURE_TO_CATEGORY.items():
#         if category in enabled_categories:
#             pattern1 = f"'{feature}'\\s*:"
#             pattern2 = f'"{feature}"\\s*:'
#             if re.search(pattern1, content_text) or re.search(pattern2, content_text):
#                 has_enabled_features = True
#                 break
    
#     if not has_enabled_features:
#         # Variable has no enabled features - exclude it entirely
#         return None
    
#     # Variable has some enabled features - filter out individual prohibited features
#     filtered_lines = []
#     skip_mode = False
#     brace_level = 0
#     skip_start_level = 0
    
#     for line in variable_lines:
#         original_line = line
#         stripped = line.strip()
        
#         # Track brace levels
#         old_brace_level = brace_level
#         brace_level += stripped.count('{') - stripped.count('}')
        
#         # If we're in skip mode, check if we should exit
#         if skip_mode:
#             if brace_level <= skip_start_level:
#                 skip_mode = False
#             else:
#                 continue  # Skip this line
        
#         # Check if this line contains any prohibited features
#         should_skip = False
#         for pattern in prohibited_patterns:
#             if re.search(pattern, stripped):
#                 should_skip = True
#                 break
        
#         if should_skip:
#             # Start skipping this feature and its nested content
#             skip_mode = True
#             skip_start_level = brace_level - stripped.count('{')
#             continue
        
#         # If we reach here and not in skip mode, include the line
#         filtered_lines.append(original_line)
    
#     return filtered_lines if filtered_lines else None

# def _recursive_filter_dict(obj, enabled_categories):
#     """
#     Recursively filter dictionary based on enabled categories.
#     More permissive filtering that preserves nested structures with any valid features.
#     """
#     if not isinstance(obj, dict):
#         return obj
        
#     filtered_dict = {}
    
#     for key, value in obj.items():
#         # Always preserve execution metadata
#         if key == 'execution_cell_source':
#             filtered_dict[key] = value
#             continue
            
#         # Handle dynamic per-column keys that start with 'per_column'
#         lookup_key = key
#         if key.startswith('per_column'):
#             lookup_key = 'per_column'
            
#         category = FEATURE_TO_CATEGORY.get(lookup_key)
        
#         if category and category in enabled_categories:
#             # Feature is in enabled category - include it
#             if isinstance(value, dict):
#                 # Recursively filter nested dictionaries
#                 filtered_nested = _recursive_filter_dict(value, enabled_categories)
#                 if filtered_nested:
#                     filtered_dict[key] = filtered_nested
#             else:
#                 # Include non-dict values directly
#                 filtered_dict[key] = value
#         elif category is None:
#             # Unknown feature - could be nested structure, explore it
#             if isinstance(value, dict):
#                 # Recursively filter unknown nested structures 
#                 filtered_nested = _recursive_filter_dict(value, enabled_categories)
#                 # Include if it contains any valid features after filtering
#                 if filtered_nested:
#                     filtered_dict[key] = filtered_nested
#             # Unknown scalars are excluded for safety
#         # else: known feature not in enabled categories - exclude
                
#     return filtered_dict

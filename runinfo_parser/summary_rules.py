import inspect
import re
import site
import os, json
from collections import defaultdict
import config

# import numpy as np
# import pandas as pd
# import torch
# from torch.utils.data import DataLoader
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import DirectoryIterator
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

config_path = config.sum_rule_config_path
with open(config_path) as f:
    config = json.load(f)

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
        if config["other"].get("summarize_callable_signature", False):
            summary['signature'] = str(inspect.signature(val))
        if config["other"].get("summarize_callable_doc", False):
            summary['doc'] = inspect.getdoc(val)
        if config["other"].get("summarize_callable_sourcecode", False):
            if (inspect.isfunction(val) or inspect.ismethod(val)) and _is_user_defined_function(val):
                summary['source'] = inspect.getsource(val)
    except Exception as e:
        if config["other"].get("summarize_callable_sourcecode", False):
            if name and namespace and f'__source__{name}' in namespace:
                summary['source'] = namespace[f'__source__{name}']
    return summary

def summarize_collection(val):
    if isinstance(val, (list, tuple, dict, set)):
        return {'length': len(val)}
    return None

def summarize_primitive(val):
    if isinstance(val, (int, float, str, bool)):
        return {'value': val}
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
            return summary

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

    return summary

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
    return summary

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
        summary["per_column"] = {}
        for col in df.columns:
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

                summary["per_column"][col] = {
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
                summary["per_column"][col] = col_summary

    return summary

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
                return summary
    except Exception:
        pass

    # Heuristic 2: sklearn.datasets.load_* output
    if {"data", "target"}.issubset(val.keys()):
        summary["detected"] = "sklearn_dataset_dict"
        summary["keys"] = keys
        summary["data_shape"] = getattr(val.get("data"), "shape", None)
        summary["target_shape"] = getattr(val.get("target"), "shape", None)
        return summary

    # Heuristic 3: flat scalar metrics dict
    if all(isinstance(v, (int, float)) for v in val.values()):
        summary["detected"] = "scalar_metrics_dict"
        summary["metrics"] = keys
        return summary

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

    return summary

#--------------------------------tensorflow/keras----------------------------------
# test: tensorflow_2/tensorflow_2_reproduced.ipynb
def summarize_directory_iterator(val):
    cls = type(val)
    if cls.__name__ == "DirectoryIterator" and "keras" in cls.__module__: # if isinstance(val, DirectoryIterator):
        return {
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
        return summary
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

    return summary

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

    return summary

# test: torch_2/torch_2_reproduced.ipynb
# no shape, dtype info without executing: next(iter(val))
def summarize_pytorch_dataloader(val, name=None):
    cls = type(val)
    if cls.__name__ == "DataLoader" and cls.__module__.startswith("torch.utils.data"): # if not isinstance(val, DataLoader):
        return {
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

    return summary

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
        return summary
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
            
    return summary
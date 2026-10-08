# CRANE-LLM: Runtime-Augmented LLMs for Crash Prediction and Diagnosis in ML Notebooks

**Predict whether an ML notebook cell will crash — before you run it.**

The repository holds two things: the JupyterLab extension that anyone can install and use, and the artefacts that reproduce the paper.

> **Looking for the artefacts from our SCAM 2026 paper "*[CRANE-LLM: Runtime-Augmented LLMs for Crash Prediction and Diagnosis in ML Notebooks](https://arxiv.org/abs/2602.18537)*"?** The repository is on the [`paper_scam26`](../../tree/paper_scam26) branch. That branch is frozen; development continues here.

## Quick start

```bash
pip install "crane-llm[lab]"
jupyter lab
```

Set an API key once, in any notebook cell:

```python
import crane_llm
crane_llm.set_api_key("sk-...")
```

Then run a few cells, select the cell you want to check, and click **CRANE-LLM** in the toolbar. You can also have every cell checked before it runs, so that a cell that would certainly crash is not run at all.

Kaggle and Colab cannot load the toolbar button. There, you use the `%%crane_llm` cell magic instead and keep the key in a notebook secret; the [user guide](./README_PYPI.md) has the steps. It also covers providers other than OpenAI, where the extension looks for its settings, and the runtime-information switch.

## Documentation

| If you want to | Read |
|---|---|
| use the extension | [README_PYPI.md](./README_PYPI.md) — the user guide, which is also the [PyPI project page](https://pypi.org/project/crane-llm/) |
| see which crashes are detected for certain, and what is sent to the model | [Built-in checks and runtime information](#built-in-checks-and-runtime-information), below |
| change or build the extension | [crane_llm/nb_extension/README.md](./crane_llm/nb_extension/README.md) |
| reproduce the paper | [PAPER.md](./PAPER.md) |

## Built-in checks and runtime information

CRANE-LLM answers in two ways. A built-in check reports a crash it can detect for certain from the live kernel state, without calling the model. Otherwise the model predicts, from the executed cells, the target cell and a summary of the runtime state. This section lists both: which crashes the built-in checks detect, and what the runtime summary contains. How to use them is in the [user guide](./README_PYPI.md).

### Built-in checks

A check reports a crash only when it is certain, and never declares a cell safe: when it finds nothing, the model is asked. It reads the cell in the order Python runs it and stops at code whose effect it cannot know, such as a call to your own function or the body of a loop.

Past such code it carries on, with every check below, but only on what that code cannot have changed: which variables exist, and values that cannot change at all, such as numbers, strings and `None`, under names that code cannot reassign. A cell that ends with `data1.head()` is therefore a certain `NameError` however much runs before it, as long as nothing in the cell and none of the notebook's functions can define `data1`. Values that can change, such as DataFrames and lists, and the world outside the kernel (files, installed packages) are not checked past that point, since the code before may have changed them.

With *Check cells before they run* switched on, the same checks run on every cell just before it executes, and a cell they find will crash is not run.

| Library | The cell... | Raises |
|---|---|---|
| Python | uses a name that is not defined | `NameError` |
| Python | reads an attribute that does not exist, including on `None` (`df = df.dropna(inplace=True)` leaves `df` as `None`) and APIs a library has removed, such as `DataFrame.append` or `np.float` | `AttributeError` |
| Python | imports a module that is not installed, or a name a module does not have, such as `from sklearn import cross_validation` | `ModuleNotFoundError`, `ImportError` |
| Python | calls a function with arguments its signature does not accept: a missing or extra argument, a positional argument where only keywords are allowed, or a keyword a library has removed, such as `read_csv(error_bad_lines=...)` | `TypeError` |
| Python | calls something that cannot be called, such as `df.shape()` or `df.columns()` | `TypeError` |
| Python | opens a file that does not exist, with `open`, `pd.read_csv`, `read_excel`, `read_parquet` and similar, `np.load` or `os.listdir` | `FileNotFoundError` |
| Python | loops over, or unpacks, something that cannot be iterated, such as `for x in None` | `TypeError` |
| Python | tests membership in something that is not a container (`x in 5`), or orders values that cannot be compared (`None > 0`, `1 < "a"`) | `TypeError` |
| Python | uses a list, dict or set as a dict key | `TypeError` |
| Python | assigns into a tuple, a string or `None` | `TypeError` |
| Python | formats a string with too few or too many arguments (`"{} {}".format(x)`, `"%d %d" % (x,)`) or a format code the value does not support (`f"{1.5:d}"`) | `IndexError`, `KeyError`, `ValueError`, `TypeError` |
| Python | applies `-`, `+` or `~` to a value that does not support it, such as `-"a"` or `-None` | `TypeError` |
| Python | deletes from a tuple or `None`, or deletes a missing key or index (`del d["x"]`, `del lst[5]`) | `TypeError`, `KeyError`, `IndexError` |
| Python | passes `*x` where `x` cannot be iterated, or `**x` where `x` is not a mapping | `TypeError` |
| Python | loops with `for a, b in ...` over items that do not unpack into two, such as the keys of a dict | `ValueError`, `TypeError` |
| Python | slices a list, tuple or string with bounds that are not integers, or a step of zero | `TypeError`, `ValueError` |
| Python | reaches a `raise` outside any `try`, including `raise` of something that is not an exception | the raised exception, `TypeError` |
| Python | reads an environment variable that is not set with `os.environ["NAME"]`, or calls `os.getenv` with a name that is not a string | `KeyError`, `TypeError` |
| Python | instantiates a class that still has abstract methods | `TypeError` |
| Python | reads a missing dict key, or a list or tuple index out of range | `KeyError`, `IndexError` |
| Python | divides by zero, adds incompatible types such as `None + 1`, unpacks the wrong number of values, or cannot be parsed | `ZeroDivisionError`, `TypeError`, `ValueError`, `SyntaxError` |
| pandas | selects, drops, groups, sorts or indexes by a column that does not exist | `KeyError` |
| pandas | uses `.loc` with a row or column label that does not exist | `KeyError` |
| pandas | uses `.iloc` with a position past the end, such as `df.iloc[100]` on a frame of 10 rows | `IndexError` |
| pandas | converts a column with `astype(int)` that holds a value which cannot become an integer, such as the text `"n/a"`, `None`, or NaN | `ValueError`, `IntCastingNaNError`, `TypeError` |
| pandas | assigns a list of the wrong length as a column | `ValueError` |
| pandas | uses `.str`, `.dt` or `.cat` on a column of another type, such as `.str` on numbers | `AttributeError` |
| pandas | adds, multiplies or compares a Series with a list or array of a different length | `ValueError` |
| pandas | merges or joins on a key column one of the frames does not have, or concatenates nothing | `KeyError`, `ValueError` |
| pandas, NumPy | tests a DataFrame, a Series or an array of more than one element for truth: `if df:`, `not arr` | `ValueError` |
| NumPy | multiplies, adds, stacks or reshapes arrays whose shapes do not fit, or indexes past the end | `ValueError`, `IndexError` |
| NumPy | reduces along an axis the array does not have, such as `a.sum(axis=3)` on a 2-D array | `AxisError` |
| NumPy | indexes a plain array with a string (`arr["price"]`), or a list with an array of several values (`items[np.array([0, 2])]`) | `IndexError`, `TypeError` |
| NumPy | converts text that is not a number with `astype(int)`, or passes a function data it rejects for its number of dimensions or its missing values | `ValueError` |
| scikit-learn | calls `predict` on a model that has not been fitted | `NotFittedError` |
| scikit-learn | passes a model a different number of features, or columns with different names or order, than it was fitted on | `ValueError` |
| scikit-learn | fits a model, or calls `train_test_split`, with `X` and `y` of different lengths | `ValueError` |
| scikit-learn | fits a classifier on a continuous target, or scores continuous predictions with a classification metric such as `accuracy_score` | `ValueError` |
| scikit-learn | creates an estimator with an argument it does not take, or calls `train_test_split` with a `test_size` that is neither a fraction nor a count below the number of samples | `TypeError`, `ValueError` |
| scikit-learn | passes data with NaN or infinity to an estimator that does not accept missing values, or text such as `"male"` to one that only takes numbers | `ValueError` |
| scikit-learn | predicts with k-nearest neighbours where `n_neighbors` is larger than the number of samples fitted | `ValueError` |
| matplotlib | shows an array with `plt.imshow` that is not an image: not 2-D, or 3-D without 1, 3 or 4 channels | `TypeError` |
| PyTorch | passes an `nn.Linear` layer input whose last dimension is not its `in_features` | `RuntimeError` |
| PyTorch | multiplies (`@`, `torch.matmul`), adds, concatenates (`torch.cat`) or reshapes (`view`, `reshape`) tensors whose shapes do not fit | `RuntimeError` |
| PyTorch | calls `.numpy()` on a tensor that requires grad, or `.item()` on a tensor of more than one value | `RuntimeError` |
| Keras | reads a metric the training history did not record, or calls a method a model does not have | `KeyError`, `AttributeError` |

### Runtime information sent to the LLM

Only the variables the target cell uses are included: every name it reads that exists in the kernel, plus the attributes and methods it uses on them.

| Object | What the prompt includes |
|---|---|
| `int`, `float`, `str`, `bool` | the value itself, including the full text of a string |
| `list`, `tuple`, `set` | length. A flat list also gets the value summary below |
| `dict` | length, and depending on the contents: the metric names and epoch count of a Keras training history; the keys and data/target shapes of a scikit-learn dataset; the keys of a dict of numbers; otherwise the first 5 entries, with each value shown up to 50 characters |
| NumPy array | shape, dtype, whether it contains NaN, minimum and maximum |
| pandas Series | dtype, length, whether it contains NaN |
| pandas DataFrame | shape, whether it contains NaN, and for each of the first 20 columns: dtype, number of distinct values, and either the minimum and maximum (numeric columns) or up to 5 values, each shortened to 20 characters (other columns) |
| PyTorch tensor | shape, dtype, device, `requires_grad`, whether it contains NaN |
| PyTorch `DataLoader` and `Subset` | number of batches and examples, batch size, the dataset's fields, and the shapes of its first 10 samples and of a batch built from them |
| TensorFlow `tf.data` dataset | its element spec |
| Keras `DirectoryIterator` and `DataFrameIterator` | number of samples and classes, batch size, image shape |
| scikit-learn estimator | class, whether it has been fitted, and once fitted, the number of input features and outputs. A fitted `LabelEncoder` adds its number of classes |
| functions, methods, classes, modules | the type only |

**Value summary.** A 1-D array, a Series or a flat list is also described by its values: *binary* with the two values, *categorical* with the number of distinct values (listed when there are 5 or fewer), or *continuous* with its minimum and maximum.

## License

This project is licensed under the terms of the BSD 3-Clause License.

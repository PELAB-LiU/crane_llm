from collections.abc import Sequence
import importlib
import json
import os
from pathlib import Path
from types import ModuleType

# Common ML/Data Science packages - you can modify this list as needed
DEFAULT_PACKAGES = [
    'numpy', 'pandas', 'sklearn', 'scipy', 'matplotlib', 'seaborn',
    'torch', 'tensorflow', 'keras', 'xgboost', 'lightgbm', 'plotly', 'statsmodels', 'torchvision',
    'transformers', 'datasets', 'PIL', 'cv2', 'nltk', 'spacy', 'gensim', 'fastai',
    'tqdm', 'joblib', 'h5py', 'pydantic'
]


def get_supported_modules(packages=None):
    """Gets module names to process. Uses default list if none provided."""
    if packages is None:
        packages = DEFAULT_PACKAGES

    # Filter to only packages that are actually installed
    available_packages = []
    for package in packages:
        try:
            importlib.import_module(package)
            available_packages.append(package)
        except ImportError:
            print(f"Package {package} not available, skipping")

    return available_packages


def extract_public_aliases(module_name: str):
    """Extracts all aliases in a particular module."""
    public_aliases = dict()
    passed_modules = set()

    def _extract_public_aliases(module):
        for attr_name in dir(module):
            if attr_name.startswith("_"):
                continue

            try:
                attr = getattr(module, attr_name)
            except Exception:
                continue

            if isinstance(attr, ModuleType):
                if attr.__name__ not in passed_modules:
                    passed_modules.add(attr.__name__)
                    _extract_public_aliases(attr)
                continue

            try:
                internal_mod = attr.__module__
                qualname = attr.__qualname__
                internal_path = f"{internal_mod}.{qualname}"
                public_path = f"{module.__name__}.{attr_name}"
                public_aliases[internal_path] = public_path
            except AttributeError:
                continue

    try:
        module = importlib.import_module(module_name)
        _extract_public_aliases(module)
    except ImportError:
        print(f"Could not import module: {module_name}")

    return public_aliases


def build_module_aliases_file(output_path: Path, packages=None):
    """
    Loads the specified modules, extracts their aliases,
    and writes all of that to a JSON file.
    """
    packages = get_supported_modules(packages)
    print(f'Processing {len(packages)} packages: {packages}')

    package_aliases = dict()
    for package in packages:
        aliases = extract_public_aliases(package)
        package_aliases[package] = aliases
        print(f'Loaded {len(aliases)} aliases for {package}')

    os.makedirs(output_path.parent, exist_ok=True)
    with open(output_path, 'w+') as output_file:
        j_data = json.dumps(package_aliases, indent=4)
        output_file.write(j_data)

    print(f'Aliases written to {output_path}')


def build_module_aliases(output_path=None, packages=None):
    """
    Does the same as `build_module_aliases_file`
    but uses a default output path if none provided.
    """
    if output_path is None:
        output_path = Path("alias_mapping.json")

    build_module_aliases_file(output_path=output_path, packages=packages)


# build_module_aliases()

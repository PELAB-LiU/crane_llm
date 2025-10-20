import re
from typing import Dict
from pathlib import Path
import os, ast
from llms.config_llms import config as llmconfig
import sas.type_convert.type_convertor as type_convertor

def clean_type(type_str: str) -> str:
    """Clean and validate type string, return None if invalid"""
    if not type_str or not isinstance(type_str, str):
        return None
    
    type_str = type_str.strip()
    
    # Handle <class '...'> format
    if type_str.startswith("<class '") and type_str.endswith("'>"):
        cleaned = type_str[len("<class '"):-len("'>")]
        # Validate that the cleaned type is reasonable
        if cleaned and not cleaned.startswith("<") and not cleaned.endswith(">"):
            return cleaned
        return None
    
    # Handle already clean type strings
    if type_str and not type_str.startswith("<") and not type_str.endswith(">"):
        return type_str
    
    # Invalid or malformed type string
    return None


def parse_element_spec(el_spec: str) -> str:
    if not el_spec:
        return ""
    shape_match = re.search(r"shape=\(([^)]*)\)", el_spec)
    dtype_match = re.search(r"dtype=([\w\.]+)", el_spec)
    shape = shape_match.group(1) if shape_match else ""
    dtype = dtype_match.group(1) if dtype_match else ""
    parts = []
    if shape:
        parts.append(f"shape=({shape})")
    if dtype:
        parts.append(f"dtype={dtype}")
    return ", ".join(parts)


def _extract_imports_and_aliases(code: str) -> Dict[str, str]:
    """Extract import statements and their aliases from the code.
    
    Returns a dict mapping full module names to their aliases.
    For example: {'numpy': 'np', 'tensorflow': 'tf', 'torch.nn': 'nn'}
    
    IMPORTANT: Only includes modules that are directly importable for type annotations.
    'from module.submodule import something' does NOT make 'module' available for type annotations.
    """
    import_map = {}
    
    for line in code.split('\n'):
        line = line.strip()
        
        # Match "import module.submodule as alias" - highest priority
        # This makes the full module path available under the alias
        match = re.match(r'^import\s+(\w+(?:\.\w+)*)\s+as\s+(\w+)', line)
        if match:
            full_module, alias = match.groups()
            import_map[full_module] = alias
            # Also add the root module if not already present (unless it's aliased differently)
            root_module = full_module.split(".")[0]
            if root_module not in import_map:
                import_map[root_module] = root_module
            continue
            
        # Match "import module" or "import module.submodule"
        # This makes the module available for type annotations
        match = re.match(r'^import\s+(\w+(?:\.\w+)*)', line)
        if match:
            module = match.group(1)
            if '.' not in module:
                # Direct module import like 'import keras'
                if module not in import_map:
                    import_map[module] = module
            else:
                # Submodule import like 'import torch.nn' or 'import matplotlib.pyplot'
                root_module = module.split(".")[0]
                # Both the full path and root module become available
                import_map[module] = module
                if root_module not in import_map:
                    import_map[root_module] = root_module
            continue
            
        # NOTE: We do NOT include "from module import ..." patterns here
        # because they don't make the module itself available for type annotations
        # For example: "from keras.models import Sequential" makes Sequential available,
        # but does NOT make keras.Sequential available for type annotations
    
    return import_map


def _convert_to_aliased_type(vtype: str, import_map: Dict[str, str]) -> str:
    """Convert a full module type to use aliases found in the code.
    
    For example: 'numpy.ndarray' -> 'np.ndarray' if numpy is imported as np
    """
    # Handle builtin types
    builtin_types = {"str", "int", "float", "bool", "list", "dict", "tuple", "set"}
    if vtype in builtin_types:
        return vtype
    
    # Find the best matching import (longest match first for specificity)
    best_match = None
    best_match_len = 0
    
    for full_module, alias in import_map.items():
        # Check for exact module match
        if vtype == full_module:
            return alias
        
        # Check for prefix match (module.something)
        if vtype.startswith(f"{full_module}."):
            # Prefer longer matches for more specific imports
            # e.g., torch.nn should match before torch if both are imported
            if len(full_module) > best_match_len:
                best_match = (full_module, alias)
                best_match_len = len(full_module)
    
    if best_match:
        full_module, alias = best_match
        # Replace the module part with the alias
        return vtype.replace(f"{full_module}.", f"{alias}.", 1)
    
    # If no matching import found, return as-is
    return vtype


def _needs_import(vtype: str, existing_imports: Dict[str, str], builtin_types: set) -> bool:
    """Check if a type needs an import statement to be added.
    
    Returns True if the type requires a module that's not already imported.
    """
    # Skip builtin types
    if vtype in builtin_types or vtype.startswith("builtins."):
        return False
    
    # Skip if it's a single identifier (no dots)
    if "." not in vtype:
        return False
    
    # Check if the root module is already imported
    root_module = vtype.split(".")[0]
    
    # Check exact match first
    if root_module in existing_imports:
        return False
    
    # Check if any existing import starts with the root module
    # (e.g., sklearn.ensemble already imports sklearn)
    for imported_module in existing_imports.keys():
        if imported_module.startswith(f"{root_module}."):
            return False
    
    return True


def annotate_source_code(code: str, runtime_info: Dict[str, Dict], executed_code: str = "") -> str:
    var_info = {}
    imports_needed = set()

    # Extract existing imports and aliases from the executed code (not just the target code)
    code_to_analyze = executed_code if executed_code else code
    existing_imports = _extract_imports_and_aliases(code_to_analyze)

    for var, info in runtime_info.items():
        if var.startswith("__method__"):
            continue
        raw_type = info.get("type", "Any")
        
        # Clean the type string first (remove <class '...'> wrapper)
        cleaned_type = clean_type(raw_type)
        
        # Skip variables with invalid or unparseable types
        if cleaned_type is None:
            print(f"Skipping variable '{var}' with invalid type: {raw_type}")
            continue
            
        vtype = type_convertor.map_to_public_alias(cleaned_type)
        
        # Handle special cases for better type annotations
        if vtype == "pandas.io.stata.DataFrame":  # Fix incorrect DataFrame mapping
            vtype = "pandas.DataFrame"
        elif vtype == "pandas.io.stata.Series":  # Fix incorrect Series mapping
            vtype = "pandas.Series"
        elif vtype == "torch.fft.Tensor":  # Fix incorrect Tensor mapping
            vtype = "torch.Tensor"
        elif cleaned_type == "torch.Tensor":  # Direct torch.Tensor case
            vtype = "torch.Tensor"
        elif cleaned_type.startswith("transformers."):
            # For transformers types, use the simplified public API
            if "BatchEncoding" in cleaned_type:
                vtype = "transformers.BatchEncoding"
            else:
                vtype = cleaned_type  # Keep as-is for other transformers types
        
        # Skip annotations for modules, types, and other uninformative annotations
        skip_types = {
            "module", "ModuleType", "type", "function", "method", 
            "builtin_function_or_method", "abc.ABCMeta"
        }
        if vtype in skip_types or cleaned_type in skip_types:
            continue
            
        # Skip __main__ types since they're not importable by static analyzers
        if vtype.startswith("__main__."):
            continue

        # Handle builtin types
        builtin_types = {"str", "int", "float", "bool", "list", "dict", "tuple", "set", 
                        "type", "function", "module", "NoneType", "object"}
        if vtype.startswith("builtins."):
            # Use the simple type name for builtins
            vtype = vtype.split(".")[-1]  # builtins.str -> str
        elif vtype not in builtin_types:
            # Convert to aliased type if available in existing imports
            original_vtype = vtype
            vtype = _convert_to_aliased_type(vtype, existing_imports)
            
            # Check if we need to add an import for this type
            if _needs_import(original_vtype, existing_imports, builtin_types):
                root_module = original_vtype.split(".")[0]
                if (root_module.isidentifier() and 
                    root_module not in ("typing", "builtins") and
                    root_module not in builtin_types):
                    imports_needed.add(root_module)

        var_info[var] = vtype

    out_lines = []

    # Add missing imports first
    if imports_needed:
        for module in sorted(imports_needed):
            out_lines.append(f"import {module}")
        out_lines.append("")

    # Add variable annotations using real types (no quotes, no TYPE_CHECKING)
    for var, vtype in var_info.items():
        out_lines.append(f"{var}: {vtype}")

    out_lines.append("")
    out_lines.append(code)

    return "\n".join(out_lines)


def run(input_path: Path, target_path: Path):
    source_code = input_path.read_text(encoding='utf-8')
    runinfo_marker_start = "# Current relevent runtime information:\n"
    runinfo_marker_end = "# Target Cell:\n"
    runinfo_idx_start = source_code.find(runinfo_marker_start)
    if runinfo_idx_start == -1:
        print(f"Current relevant runtime information marker not found in {input_path.name}.")
        return
    runinfo_idx_end = source_code.find(runinfo_marker_end)
    if runinfo_idx_end == -1:
        print(f"Target cell marker not found in {input_path.name}.")
        return
    executed_code = source_code[:runinfo_idx_start]
    runtime_info = source_code[runinfo_idx_start+len(runinfo_marker_start):runinfo_idx_end]
    target_code = source_code[runinfo_idx_end+len(runinfo_marker_end):]

    runtime_info = re.sub(r"dtype\('([^']+)'\)", r"'\1'", runtime_info)
    
    # Handle only the most problematic patterns - be conservative
    # Count and replace object references
    object_count = len(re.findall(r'<[^>]+object at 0x[a-fA-F0-9]+>', runtime_info))
    if object_count > 0:
        print(f"Found {object_count} object references, replacing with placeholders...")
        runtime_info = re.sub(r'<[^>]+object at 0x[a-fA-F0-9]+>', '"<object>"', runtime_info)
    
    # Handle torch.Size which is very common and problematic
    torch_size_count = len(re.findall(r'torch\.Size\([^)]*\)', runtime_info))
    if torch_size_count > 0:
        print(f"Found {torch_size_count} torch.Size patterns, replacing...")
        runtime_info = re.sub(r'torch\.Size\([^)]*\)', '"<torch.Size>"', runtime_info)
    
    # Handle defaultdict patterns
    defaultdict_count = len(re.findall(r'defaultdict\(', runtime_info))
    if defaultdict_count > 0:
        print(f"Found {defaultdict_count} defaultdict patterns, replacing...")
        runtime_info = re.sub(r'defaultdict\([^)]*\)', '{}', runtime_info)
    
    try:
        runtime_info_dict = ast.literal_eval(runtime_info)
    except (SyntaxError, ValueError) as e:
        print(f"Error parsing runtime info in {input_path.name}: {e}")
        print("Runtime info preview:", runtime_info[:200] + "..." if len(runtime_info) > 200 else runtime_info)
        print("Attempting fallback parsing...")
        
        # Fallback: try to extract just the type information which is what we need
        try:
            # Enhanced regex-based extraction for type information
            runtime_info_dict = {}
            
            # Method 1: Standard pattern 'var': {'type': '...', ...}
            # Only match complete, well-formed type strings
            var_pattern1 = r"'([^']+)':\s*\{[^{}]*'type':\s*[\"']([^\"'<>]+)[\"'][^{}]*\}"
            matches1 = re.findall(var_pattern1, runtime_info)
            for var_name, type_str in matches1:
                if not var_name.startswith("__method__") and type_str.strip():
                    # Validate the type string before adding
                    if not type_str.startswith("<") and not type_str.endswith(">"):
                        runtime_info_dict[var_name] = {'type': type_str}
            
            # Method 2: Handle properly formatted class patterns like "type": "<class 'module.Class'>"
            var_pattern2 = r"'([^']+)':\s*\{[^{}]*'type':\s*[\"']<class\s+'([^\"'>]+)'>[\"'][^{}]*\}"
            matches2 = re.findall(var_pattern2, runtime_info)
            for var_name, class_name in matches2:
                if not var_name.startswith("__method__") and class_name.strip():
                    # Only include if the class name looks valid (no incomplete fragments)
                    if "." in class_name or class_name.isidentifier():
                        runtime_info_dict[var_name] = {'type': f"<class '{class_name}'>"}
            
            # Method 3: Extract execution cell info if available
            cell_pattern = r"'([^']+)':\s*\{[^{}]*'execution_cell_source':\s*\{[^{}]*'cellno':\s*(\d+)[^{}]*\}[^{}]*\}"
            cell_matches = re.findall(cell_pattern, runtime_info)
            for var_name, cellno in cell_matches:
                if var_name in runtime_info_dict:
                    runtime_info_dict[var_name]['execution_cell_source'] = {'cellno': int(cellno), 'lineno': 1}
            
            print(f"Fallback extraction methods found: {len(matches1)} standard + {len(matches2)} class patterns = {len(runtime_info_dict)} total variables")
            
            if not runtime_info_dict:
                print(f"Could not parse runtime info in {input_path.name}, skipping type annotation")
                # Still create the file with just the original code
                if target_path.parent != Path("."):
                    os.makedirs(target_path.parent, exist_ok=True)
                target_path.write_text(f"{executed_code}# Annotated Target Code:\n{target_code}", encoding='utf-8')
                return
            else:
                print(f"Fallback parsing succeeded, extracted {len(runtime_info_dict)} variables")
                
        except Exception as fallback_error:
            print(f"Fallback parsing also failed for {input_path.name}: {fallback_error}")
            # Still create the file with just the original code
            if target_path.parent != Path("."):
                os.makedirs(target_path.parent, exist_ok=True)
            target_path.write_text(f"{executed_code}# Annotated Target Code:\n{target_code}", encoding='utf-8')
            return

    typed_code = annotate_source_code(target_code, runtime_info_dict, executed_code)
    # print("Target code:\n", target_code)
    # print("Runinfo:\n", runtime_info_dict)
    annotated_code = f"{executed_code}# Annotated Target Code:\n{typed_code}"
    # print(annotated_code)

    # Create directory if needed
    if target_path.parent != Path("."):
        os.makedirs(target_path.parent, exist_ok=True)
    
    target_path.write_text(annotated_code, encoding='utf-8')
    print(f"Runtime type annotated code saved in {target_path}")


def generate_runinfo_injected_version(dst_path: Path, lib_name: str, case_names: list = None):
    os.makedirs(dst_path, exist_ok=True)
    src_path = llmconfig.path_input_executed_code_runinfo.joinpath(lib_name)
    if case_names:
        for case_name in case_names:
            for version in ["fixed", "reproduced"]:
                filename = f"{case_name}_{version}.txt"
                if os.path.exists(src_path / filename):
                    run(Path(os.path.join(src_path, filename)), dst_path / lib_name / filename)
                else:
                    print(f"File {filename} not found in {src_path}, skipping...")
    else:
        for filename in os.listdir(src_path):
            if filename.endswith(".txt"):
                case_name = os.path.splitext(filename)[0]
                run(Path(os.path.join(src_path, filename)), dst_path / lib_name / filename)


# print(run("tensorflow_1_reproduced.txt"))
# generate_runinfo_injected_version(Path("sas/sas_inputs/executed_code_runinfo"), "tensorflow")

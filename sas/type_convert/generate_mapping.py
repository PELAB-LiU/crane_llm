#!/usr/bin/env python3
"""
Complete alias mapping generator for all ML/Data Science packages
Includes comprehensive error handling for GPU/CUDA issues
"""
import os
import sys
import warnings
import importlib
from pathlib import Path
import json
import mapping_internal_types

def setup_cpu_only_environment():
    """Configure environment for CPU-only operation"""
    # Disable CUDA completely
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TF logs
    
    # PyTorch CPU settings
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

def extract_keras_aliases():
    """Extract Keras aliases safely, including functional models"""
    print("Extracting Keras aliases...")
    aliases = {}
    
    try:
        import keras
        
        # Key Keras modules to extract
        keras_modules = [
            ('keras', keras),
            ('keras.models', keras.models),
            ('keras.layers', keras.layers),
            ('keras.optimizers', keras.optimizers),
            ('keras.losses', keras.losses),
            ('keras.metrics', keras.metrics),
            ('keras.callbacks', keras.callbacks),
        ]
        
        for module_name, module in keras_modules:
            try:
                # Extract public attributes
                for attr_name in dir(module):
                    if attr_name.startswith('_'):
                        continue
                        
                    try:
                        attr = getattr(module, attr_name)
                        
                        # Get the internal path
                        if hasattr(attr, '__module__') and hasattr(attr, '__qualname__'):
                            internal_path = f"{attr.__module__}.{attr.__qualname__}"
                            public_path = f"{module_name}.{attr_name}"
                            aliases[internal_path] = public_path
                        elif hasattr(attr, '__module__') and hasattr(attr, '__name__'):
                            internal_path = f"{attr.__module__}.{attr.__name__}"
                            public_path = f"{module_name}.{attr_name}"
                            aliases[internal_path] = public_path
                            
                    except Exception:
                        continue
                        
            except Exception as e:
                print(f"  Warning: Could not process {module_name}: {e}")
                continue
        
        # ENHANCED: Create functional models to capture dynamic types
        print("  Creating functional models to capture dynamic types...")
        try:
            import tensorflow as tf
            
            # Create a simple functional model to capture the internal types
            inputs = keras.Input(shape=(10,))
            x = keras.layers.Dense(5, activation='relu')(inputs)
            outputs = keras.layers.Dense(1)(x)
            functional_model = keras.Model(inputs=inputs, outputs=outputs)
            
            # Extract the actual class of the functional model
            model_class = functional_model.__class__
            if hasattr(model_class, '__module__') and hasattr(model_class, '__qualname__'):
                internal_path = f"{model_class.__module__}.{model_class.__qualname__}"
                public_path = "keras.Model"
                aliases[internal_path] = public_path
                print(f"    ✓ {model_class.__qualname__} -> {public_path}")
            
            # Also try to capture other model types
            sequential_model = keras.Sequential([
                keras.layers.Dense(5, activation='relu'),
                keras.layers.Dense(1)
            ])
            
            seq_class = sequential_model.__class__
            if hasattr(seq_class, '__module__') and hasattr(seq_class, '__qualname__'):
                internal_path = f"{seq_class.__module__}.{seq_class.__qualname__}"
                public_path = "keras.Sequential"
                aliases[internal_path] = public_path
                print(f"    ✓ {seq_class.__qualname__} -> {public_path}")
                
        except Exception as e:
            print(f"    Warning: Could not create functional models: {e}")
        
        # ENHANCED: Extract preprocessing data generators to capture DirectoryIterator types
        print("  Extracting preprocessing data generators...")
        try:
            # Try both TensorFlow Keras and standalone Keras preprocessing
            preprocessing_modules = []
            
            # TensorFlow Keras preprocessing
            try:
                import tensorflow as tf
                if hasattr(tf.keras, 'preprocessing') and hasattr(tf.keras.preprocessing, 'image'):
                    preprocessing_modules.append(('tensorflow.keras.preprocessing.image', tf.keras.preprocessing.image))
            except Exception:
                pass
            
            # Standalone Keras preprocessing  
            try:
                if hasattr(keras, 'preprocessing') and hasattr(keras.preprocessing, 'image'):
                    preprocessing_modules.append(('keras.preprocessing.image', keras.preprocessing.image))
            except Exception:
                pass
            
            # Process each preprocessing module
            for module_name, module in preprocessing_modules:
                try:
                    # Extract ImageDataGenerator and DirectoryIterator classes
                    if hasattr(module, 'ImageDataGenerator'):
                        datagen = module.ImageDataGenerator(rescale=1./255)
                        
                        # Try to create a DirectoryIterator-like object
                        # Note: We can't actually create one without a directory, but we can access the class
                        if hasattr(module, 'DirectoryIterator'):
                            dir_iter_class = module.DirectoryIterator
                            if hasattr(dir_iter_class, '__module__') and hasattr(dir_iter_class, '__qualname__'):
                                internal_path = f"{dir_iter_class.__module__}.{dir_iter_class.__qualname__}"
                                public_path = f"{module_name}.DirectoryIterator"
                                aliases[internal_path] = public_path
                                print(f"    ✓ {dir_iter_class.__qualname__} -> {public_path}")
                        
                        # Extract ImageDataGenerator class
                        datagen_class = datagen.__class__
                        if hasattr(datagen_class, '__module__') and hasattr(datagen_class, '__qualname__'):
                            internal_path = f"{datagen_class.__module__}.{datagen_class.__qualname__}"
                            public_path = f"{module_name}.ImageDataGenerator"
                            aliases[internal_path] = public_path
                            print(f"    ✓ {datagen_class.__qualname__} -> {public_path}")
                            
                except Exception as e:
                    print(f"    - Failed to extract from {module_name}: {e}")
            
            # # CRITICAL: Add the specific legacy type mapping that's missing
            # # This handles the case where keras.src.legacy.preprocessing.image.DirectoryIterator appears
            # legacy_mappings = {
            #     "keras.src.legacy.preprocessing.image.DirectoryIterator": "keras.preprocessing.image.DirectoryIterator",
            #     "keras.src.legacy.preprocessing.image.ImageDataGenerator": "keras.preprocessing.image.ImageDataGenerator"
            # }
            
            # for internal_path, public_path in legacy_mappings.items():
            #     aliases[internal_path] = public_path
            #     print(f"    ✓ {internal_path} -> {public_path} (legacy mapping)")
                
        except Exception as e:
            print(f"    Warning: Could not extract preprocessing generators: {e}")
        
        # Return only keras-specific aliases (not tensorflow.keras ones)
        keras_only_aliases = {k: v for k, v in aliases.items() if not k.startswith('tensorflow.')}
        print(f"✓ Extracted {len(keras_only_aliases)} Keras aliases")
        return keras_only_aliases
        
        print(f"✓ Extracted {len(aliases)} Keras aliases")
        return aliases
        
    except Exception as e:
        print(f"✗ Keras extraction failed: {e}")
        return {}

def extract_tensorflow_aliases():
    """Extract TensorFlow aliases safely"""
    print("Extracting TensorFlow aliases...")
    aliases = {}
    
    try:
        # Suppress warnings
        warnings.filterwarnings('ignore')
        
        # Import TensorFlow modules
        import tensorflow as tf
        
        # Configure for CPU only
        tf.config.set_visible_devices([], 'GPU')
        
        # Key TensorFlow modules to extract
        tf_modules = [
            'tensorflow',
            'tensorflow.keras',
            'tensorflow.keras.layers',
            'tensorflow.keras.optimizers', 
            'tensorflow.keras.losses',
            'tensorflow.keras.metrics',
            'tensorflow.keras.callbacks',
            'tensorflow.nn',
            'tensorflow.math',
            'tensorflow.linalg',
            'tensorflow.image',
            'tensorflow.data',
            'tensorflow.train'
        ]
        
        for module_name in tf_modules:
            try:
                # Import the module
                module_parts = module_name.split('.')
                if len(module_parts) == 1:
                    module = tf
                else:
                    module = tf
                    for part in module_parts[1:]:
                        module = getattr(module, part)
                
                # Extract public attributes
                for attr_name in dir(module):
                    if attr_name.startswith('_'):
                        continue
                        
                    try:
                        attr = getattr(module, attr_name)
                        
                        # Skip if it's a module (to avoid recursion issues)
                        if hasattr(attr, '__module__') and hasattr(attr, '__name__'):
                            if hasattr(attr, '__qualname__'):
                                internal_path = f"{attr.__module__}.{attr.__qualname__}"
                            else:
                                internal_path = f"{attr.__module__}.{attr.__name__}"
                            public_path = f"{module_name}.{attr_name}"
                            aliases[internal_path] = public_path
                            
                    except Exception:
                        continue
                        
            except Exception as e:
                print(f"  Warning: Could not process {module_name}: {e}")
                continue
        
        # ENHANCED: Extract dynamically created TensorFlow data operation classes
        print("  Extracting TensorFlow data operation classes...")
        try:
            # Create sample datasets to capture dynamically created classes
            sample_data = [1, 2, 3, 4, 5]
            base_dataset = tf.data.Dataset.from_tensor_slices(sample_data)
            
            # Common data operations that create internal classes
            data_operations = [
                ('prefetch', lambda ds: ds.prefetch(1)),
                ('batch', lambda ds: ds.batch(2)),
                ('map', lambda ds: ds.map(lambda x: x * 2)),
                ('filter', lambda ds: ds.filter(lambda x: x > 0)),
                ('repeat', lambda ds: ds.repeat(2)),
                ('take', lambda ds: ds.take(3)),
                ('skip', lambda ds: ds.skip(1)),
                ('shuffle', lambda ds: ds.shuffle(10)),
            ]
            
            for op_name, op_func in data_operations:
                try:
                    result_dataset = op_func(base_dataset)
                    dataset_class = result_dataset.__class__
                    
                    if hasattr(dataset_class, '__module__') and hasattr(dataset_class, '__qualname__'):
                        internal_path = f"{dataset_class.__module__}.{dataset_class.__qualname__}"
                        # Map to the base Dataset class, not the operation method
                        public_path = "tensorflow.data.Dataset"
                        aliases[internal_path] = public_path
                        print(f"    ✓ {dataset_class.__qualname__} -> {public_path}")
                        
                except Exception as e:
                    print(f"    - Failed to extract {op_name}: {e}")
                    
        except Exception as e:
            print(f"  Warning: Could not extract data operation classes: {e}")
        
        # # CRITICAL: Add TensorFlow Keras legacy type mappings
        # print("  Adding TensorFlow Keras legacy type mappings...")
        # tf_legacy_mappings = {
        #     "tensorflow.keras.src.legacy.preprocessing.image.DirectoryIterator": "tensorflow.keras.preprocessing.image.DirectoryIterator",
        #     "tensorflow.keras.src.legacy.preprocessing.image.ImageDataGenerator": "tensorflow.keras.preprocessing.image.ImageDataGenerator"
        # }
        
        # for internal_path, public_path in tf_legacy_mappings.items():
        #     aliases[internal_path] = public_path
        #     print(f"    ✓ {internal_path} -> {public_path} (legacy mapping)")
        
        print(f"✓ Extracted {len(aliases)} TensorFlow aliases")
        return aliases
        
    except Exception as e:
        print(f"✗ TensorFlow extraction failed: {e}")
        return {}

def extract_torch_aliases():
    """Extract PyTorch aliases safely"""
    print("Extracting PyTorch aliases...")
    aliases = {}
    
    try:
        import torch
        
        # Force CPU mode
        torch.set_default_device('cpu')
        
        # Key PyTorch modules to extract
        torch_modules = [
            ('torch', torch),
            ('torch.nn', torch.nn),
            ('torch.nn.functional', torch.nn.functional),
            ('torch.optim', torch.optim),
            ('torch.utils', torch.utils),
            ('torch.utils.data', torch.utils.data),
        ]
        
        # Try to import additional modules safely
        optional_modules = [
            'torch.jit',
            'torch.autograd', 
            'torch.linalg',
            'torch.fft'
        ]
        
        for mod_name in optional_modules:
            try:
                mod = importlib.import_module(mod_name)
                torch_modules.append((mod_name, mod))
            except:
                continue
        
        for module_name, module in torch_modules:
            try:
                # Extract public attributes
                for attr_name in dir(module):
                    if attr_name.startswith('_'):
                        continue
                        
                    try:
                        attr = getattr(module, attr_name)
                        
                        # Get the internal path
                        if hasattr(attr, '__module__') and hasattr(attr, '__qualname__'):
                            internal_path = f"{attr.__module__}.{attr.__qualname__}"
                            public_path = f"{module_name}.{attr_name}"
                            aliases[internal_path] = public_path
                        elif hasattr(attr, '__module__') and hasattr(attr, '__name__'):
                            internal_path = f"{attr.__module__}.{attr.__name__}"
                            public_path = f"{module_name}.{attr_name}"
                            aliases[internal_path] = public_path
                            
                    except Exception:
                        continue
                        
            except Exception as e:
                print(f"  Warning: Could not process {module_name}: {e}")
                continue
        
        print(f"✓ Extracted {len(aliases)} PyTorch aliases")
        return aliases
        
    except Exception as e:
        print(f"✗ PyTorch extraction failed: {e}")
        return {}

def extract_torchvision_aliases():
    """Extract torchvision aliases safely"""
    print("Extracting torchvision aliases...")
    aliases = {}
    
    try:
        import torchvision
        
        # Key torchvision modules
        tv_modules = [
            ('torchvision', torchvision),
            ('torchvision.transforms', torchvision.transforms),
            ('torchvision.datasets', torchvision.datasets),
            ('torchvision.models', torchvision.models),
        ]
        
        for module_name, module in tv_modules:
            try:
                for attr_name in dir(module):
                    if attr_name.startswith('_'):
                        continue
                        
                    try:
                        attr = getattr(module, attr_name)
                        
                        if hasattr(attr, '__module__') and hasattr(attr, '__qualname__'):
                            internal_path = f"{attr.__module__}.{attr.__qualname__}"
                            public_path = f"{module_name}.{attr_name}"
                            aliases[internal_path] = public_path
                            
                    except Exception:
                        continue
                        
            except Exception as e:
                print(f"  Warning: Could not process {module_name}: {e}")
                continue
        
        print(f"✓ Extracted {len(aliases)} torchvision aliases")
        return aliases
        
    except Exception as e:
        print(f"✗ torchvision extraction failed: {e}")
        return {}

def generate_complete_mapping():
    """Generate comprehensive mapping for ALL supported packages"""
    print("Generating complete mapping for all ML/Data Science packages...")
    
    # Setup environment
    setup_cpu_only_environment()
    
    current_dir = Path(__file__).parent.resolve()
    mapping_path = current_dir.joinpath('resources').joinpath('internal_to_public_alias_mapping_raw.json')
    
    print(f"Output path: {mapping_path}")
    
    # Get all supported packages from mapping_internal_types
    print("\nScanning for available packages...")
    available_packages = mapping_internal_types.get_supported_modules()
    print(f"Found {len(available_packages)} available packages: {available_packages}")
    
    # Generate mappings for all packages
    print(f"\nGenerating mappings...")
    
    # Use the robust extraction from mapping_internal_types for most packages
    safe_packages = [pkg for pkg in available_packages if pkg not in ['torch', 'tensorflow', 'torchvision', 'keras']]
    
    package_aliases = {}
    
    # Process safe packages first
    if safe_packages:
        print(f"\nProcessing {len(safe_packages)} safe packages...")
        for package in safe_packages:
            try:
                aliases = mapping_internal_types.extract_public_aliases(package)
                package_aliases[package] = aliases
                print(f"✓ {package}: {len(aliases):,} aliases")
            except Exception as e:
                print(f"✗ {package}: Failed - {e}")
                package_aliases[package] = {}
    
    # Process deep learning packages with special handling
    dl_packages = [pkg for pkg in available_packages if pkg in ['torch', 'tensorflow', 'torchvision', 'keras']]
    if dl_packages:
        print(f"\nProcessing {len(dl_packages)} deep learning packages with enhanced handling...")
        
        # TensorFlow
        if 'tensorflow' in dl_packages:
            tf_aliases = extract_tensorflow_aliases()
            package_aliases['tensorflow'] = tf_aliases
        
        # Keras (standalone)
        if 'keras' in dl_packages:
            keras_aliases = extract_keras_aliases()
            package_aliases['keras'] = keras_aliases
        
        # PyTorch  
        if 'torch' in dl_packages:
            torch_aliases = extract_torch_aliases()
            package_aliases['torch'] = torch_aliases
        
        # torchvision
        if 'torchvision' in dl_packages:
            tv_aliases = extract_torchvision_aliases()
            package_aliases['torchvision'] = tv_aliases
    
    # Write the complete mapping
    try:
        os.makedirs(mapping_path.parent, exist_ok=True)
        with open(mapping_path, 'w') as f:
            json.dump(package_aliases, f, indent=4)
        
        print(f"\n✓ Complete mapping written to {mapping_path}")
        
        # Statistics
        total_aliases = sum(len(v) for v in package_aliases.values())
        successful_packages = sum(1 for v in package_aliases.values() if len(v) > 0)
        
        print(f"✓ Successful packages: {successful_packages}/{len(available_packages)}")
        print(f"✓ Total aliases: {total_aliases:,}")
        
        print(f"\nPackage breakdown:")
        for pkg, aliases in sorted(package_aliases.items()):
            status = "✓" if len(aliases) > 0 else "✗"
            print(f"  {status} {pkg}: {len(aliases):,} aliases")
        
        return True
        
    except Exception as e:
        print(f"✗ Error writing mapping: {e}")
        return False


def cleanup_generated_mapping():
    current_dir = Path(__file__).parent.resolve()
    mapping_path = current_dir.joinpath('resources').joinpath('internal_to_public_alias_mapping_raw.json')
    
    with open(mapping_path, 'r') as mapping_file:
        j_data = mapping_file.read()
        _internal_to_public_mapping = json.loads(j_data)
        # clean up by removing keys in values that are not start with the root key
        cleaned_mapping = {}
        for root_key, aliases in _internal_to_public_mapping.items():
            cleaned_aliases = {}
            for internal_alias, public_alias in aliases.items():
                if internal_alias.startswith(root_key + "."):
                    cleaned_aliases[internal_alias] = public_alias
            cleaned_mapping[root_key] = cleaned_aliases
        # save cleaned mapping to json file
        cleaned_mapping_path = current_dir.joinpath('resources').joinpath('internal_to_public_alias_mapping.json')
        with open(cleaned_mapping_path, 'w') as f:
            json.dump(cleaned_mapping, f, indent=4)
        print(f"\n✓ Cleaned mapping written to {cleaned_mapping_path}")


if __name__ == '__main__':
    success = generate_complete_mapping()
    if success:
        cleanup_generated_mapping()
        print(f"\n🎉 Complete mapping generated successfully!")
    else:
        print(f"\n❌ Failed to generate complete mapping")
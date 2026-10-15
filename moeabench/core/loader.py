# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from joblib import load
import zipfile
from io import BytesIO
import pathlib

class loader:
    @staticmethod
    def IPL_loader(target_obj, folder, mode='all'):
        path_z = pathlib.Path(folder)
        if path_z.suffix != '.zip':
            path_z = path_z.with_suffix('.zip')
            
        if not path_z.exists():
            raise FileNotFoundError(f"Experiment file not found: {path_z}")
       
        with zipfile.ZipFile(path_z, 'r') as zf:
            bytes_data = zf.read('Moeabench.joblib')
            # Older archives may not contain modern provenance metadata.
            archive_metadata = None
            if 'metadata.json' in zf.namelist():
                import json
                archive_metadata = json.loads(zf.read('metadata.json'))
            loaded_obj = load(BytesIO(bytes_data))

        # Prefer execution provenance captured on the object, falling back to
        # archive metadata for legacy exports (whose versions reflect save time).
        if not getattr(loaded_obj, "_execution_environment", None) and archive_metadata:
            provenance = archive_metadata.get("provenance", {})
            original = {
                "moeabench": provenance.get("moeabench"),
                "python": provenance.get("python"),
            }
            for dep in ("numpy", "scipy", "pymoo", "pandas", "moocore", "deap"):
                if dep + "_version" in provenance:
                    original[dep] = provenance[dep + "_version"]
            if original["moeabench"] is not None and original["python"] is not None:
                loaded_obj._execution_environment = original
        from ..environment_verification import verify_experiment

        # Handle selective loading based on mode
        if mode == 'config':
            # Keep existing runs if any, only update config
            # But usually load('config') means you want a clean slate with that config
            # However, the user said "load configuration only"
            # We'll update everything except runs
            original_runs = target_obj._runs
            target_obj.__dict__.update(loaded_obj.__dict__)
            target_obj._runs = original_runs
            
        elif mode == 'data':
            # Only update runs from the loaded object
            target_obj._runs = loaded_obj._runs
            # We might want to update the result pointer too if used
            if hasattr(loaded_obj, 'result'):
                target_obj.result = loaded_obj.result
        else:
            # Default 'all': full update
            target_obj.__dict__.update(loaded_obj.__dict__)
            
        verify_experiment(target_obj, show=False)
        return target_obj



       
      
 

        
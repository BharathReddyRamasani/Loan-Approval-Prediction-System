import os
import json
import joblib
import numpy as np

def ensure_directory_exists(directory_path):
    """Ensures a directory exists, creating it if necessary."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")

def save_artifact(artifact, filename, directory="models"):
    """Saves a Python object (e.g. model, preprocessor) using joblib."""
    ensure_directory_exists(directory)
    filepath = os.path.join(directory, filename)
    joblib.dump(artifact, filepath)
    print(f"Saved artifact to {filepath}")

def load_artifact(filename, directory="models"):
    """Loads a Python object using joblib."""
    filepath = os.path.join(directory, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Artifact not found: {filepath}")
    return joblib.load(filepath)

def save_metrics(metrics_dict, filename="metrics.json", directory="models"):
    """Saves model metrics to a JSON file for quick loading by UI."""
    ensure_directory_exists(directory)
    filepath = os.path.join(directory, filename)
    # Ensure nested numeric structures are JSON serializable
    clean_metrics = {}
    for k, v in metrics_dict.items():
        if isinstance(v, dict):
            clean_metrics[k] = {ik: (float(iv) if isinstance(iv, (int, float, np.integer, np.floating)) else iv) for ik, iv in v.items()}
        else:
            clean_metrics[k] = v
            
    with open(filepath, 'w') as f:
        json.dump(clean_metrics, f, indent=4)
    print(f"Saved training metrics to {filepath}")

def load_metrics(filename="metrics.json", directory="models"):
    """Loads training metrics from a JSON file."""
    filepath = os.path.join(directory, filename)
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r') as f:
        return json.load(f)

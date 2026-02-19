from pathlib import Path
import yaml

PROJECT_NAME = "."

# ----------------------------
# Define Structure
# ----------------------------

directories = [
    "data/raw",
    "data/interim",
    "data/processed",
    "notebooks",
    "src",
    "models",
    "configs",
    "mlruns"
]

files = {
    "src/data.py": "",
    "src/features.py": "",
    "src/train.py": "",
    "src/evaluate.py": "",
    "main.py": "",
    "requirements.txt": "",
    "configs/config.yaml": {
        "random_seed": 42,
        "data_path": "data/raw/train.csv",
        "target_column": "target",
        "model": {
            "name": "xgboost",
            "learning_rate": 0.1,
            "n_estimators": 100
        }
    }
}

# ----------------------------
# Create Project
# ----------------------------

base_path = Path(PROJECT_NAME)

for directory in directories:
    (base_path / directory).mkdir(parents=True, exist_ok=True)

for file_path, content in files.items():
    full_path = base_path / file_path
    full_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(content, dict):
        with open(full_path, "w") as f:
            yaml.dump(content, f)
    else:
        full_path.touch()

print(f"✅ Project '{PROJECT_NAME}' created successfully.")

import os
from pathlib import Path

# creates everything inside the current folder
root = Path(".")

dirs = [
    "data/raw",
    "data/processed",
    "data/features",
    "models",
    "src/data",
    "src/training",
    "src/api",
    "src/monitoring",
    "notebooks",
    "tests",
    ".github/workflows",
    "docker",
    "monitoring/grafana",
    "mlflow",
]

files = {
    # package inits
    "src/__init__.py":                       "",
    "src/data/__init__.py":                  "",
    "src/training/__init__.py":              "",
    "src/api/__init__.py":                   "",
    "src/monitoring/__init__.py":            "",
    "tests/__init__.py":                     "",

    # gitkeeps so empty dirs are tracked by git
    "data/raw/.gitkeep":                     "",
    "data/processed/.gitkeep":              "",
    "data/features/.gitkeep":               "",
    "models/.gitkeep":                       "",
    "notebooks/.gitkeep":                    "",
    "mlflow/.gitkeep":                       "",
    "monitoring/grafana/.gitkeep":           "",

    # src/data
    "src/data/preprocess.py":               "# preprocess.py\n",
    "src/data/feature_engineering.py":      "# feature_engineering.py\n",

    # src/training
    "src/training/train.py":                "# train.py\n",
    "src/training/ensemble.py":             "# ensemble.py\n",
    "src/training/hyperparameter_tuning.py":"# hyperparameter_tuning.py\n",

    # src/api
    "src/api/main.py":                      "# main.py\n",
    "src/api/schemas.py":                   "# schemas.py\n",
    "src/api/predict.py":                   "# predict.py\n",

    # src/monitoring
    "src/monitoring/drift_detector.py":     "# drift_detector.py\n",
    "src/monitoring/metrics.py":            "# metrics.py\n",

    # docker
    "docker/Dockerfile":                    "# Dockerfile\n",
    "docker/docker-compose.yml":            "# docker-compose.yml\n",

    # monitoring
    "monitoring/prometheus.yml":            "# prometheus.yml\n",

    # ci/cd
    ".github/workflows/retrain.yml":        "# retrain.yml\n",

    # root config files
    "dvc.yaml":                             "# dvc pipeline stages\n",
    "params.yaml":                          "# xgb and lgb hyperparameters\n",
}


def build():
    created_dirs  = 0
    created_files = 0
    skipped       = 0

    for d in dirs:
        path = root / d
        path.mkdir(parents=True, exist_ok=True)
        created_dirs += 1

    for rel_path, content in files.items():
        path = root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(content)
            created_files += 1
        else:
            skipped += 1

    print("\n✅  structure created inside current folder:")
    print(f"    {created_dirs} dirs  |  {created_files} files created  |  {skipped} already existed\n")

    # print tree
    for path in sorted(root.rglob("*")):
        parts = path.parts
        if any(p in ("__pycache__", ".git") for p in parts):
            continue
        rel   = path.relative_to(root)
        depth = len(rel.parts) - 1
        indent = "    " * depth
        name   = path.name + ("/" if path.is_dir() else "")
        print(f"{indent}├── {name}")


if __name__ == "__main__":
    build()
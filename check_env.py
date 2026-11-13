"""Checks that the laptop environment is ready. Run: .venv\\Scripts\\python.exe check_env.py"""
import importlib
import importlib.metadata as metadata
import sys

ALLOWED_PYTHON = ((3, 11), (3, 12))

PACKAGES = {  # distribution name -> module to import
    "pandas": "pandas",
    "scikit-learn": "sklearn",
    "tqdm": "tqdm",
    "psutil": "psutil",
    "python-dotenv": "dotenv",
    "torch": "torch",
    "transformers": "transformers",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "pydantic": "pydantic",
    "langgraph": "langgraph",
    "pytest": "pytest",
    "httpx": "httpx",
}


def main() -> int:
    failures = 0

    version = sys.version_info
    ok = (version.major, version.minor) in ALLOWED_PYTHON
    print(f"[{'OK' if ok else 'FAIL'}] python {version.major}.{version.minor}.{version.micro}  ({sys.executable})")
    failures += not ok

    for dist, module in PACKAGES.items():
        try:
            installed = metadata.version(dist)
            importlib.import_module(module)
            print(f"[OK] {dist} {installed}")
        except Exception as error:
            print(f"[FAIL] {dist}: {error!r}")
            failures += 1

    try:
        import torch

        result = float((torch.ones(2, 2) @ torch.ones(2, 2)).sum())
        ok = result == 8.0
        print(f"[{'OK' if ok else 'FAIL'}] torch matrix multiply on CPU (CUDA available: {torch.cuda.is_available()})")
        failures += not ok
    except Exception as error:
        print(f"[FAIL] torch CPU test: {error!r}")
        failures += 1

    print()
    print("Environment is ready." if failures == 0 else f"{failures} problem(s) found.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
from pathlib import Path


import classification

PACKAGE_ROOT = Path(classification.__file__).resolve().parent

print(f"Package root from core {PACKAGE_ROOT}")
from classification.config.core import PACKAGE_ROOT

# print(f"Package root from init {PACKAGE_ROOT}")

with open(PACKAGE_ROOT / "VERSION") as version_file:
    __version__ = version_file.read().strip()
    print(f"Version from init {__version__}")
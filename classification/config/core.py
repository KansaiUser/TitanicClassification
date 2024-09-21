from pathlib import Path

from typing import Optional

from pydantic import BaseModel
from strictyaml import YAML, load

import classification

PACKAGE_ROOT = Path(classification.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
DATASET_DIR = PACKAGE_ROOT / "datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"

# SOURCE_DATA = 

print(f"Package root from core {PACKAGE_ROOT}")
print(f"Config file path {CONFIG_FILE_PATH}")

class AppConfig(BaseModel):
    package_name: str
    data_source: str
    pipeline_save_file: str

class Config(BaseModel):
    app_config: AppConfig


def fetch_config_from_yaml(cfg_path:Optional[Path]= None)-> YAML:
    if not cfg_path:
        pass
    if cfg_path:
        print("Opening and reading Yaml")
        with open(cfg_path,"r")as conf_file:
            parsed_config = load(conf_file.read())
            print(f"parsed config type {type(parsed_config)}")
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")
    


def create_and_validate_config(parsed_config: YAML = None) -> Config:
    if parsed_config is None:
        print("Parsed config is None")
        # this is just temporal
        path_to_config = Path(CONFIG_FILE_PATH)
        parsed_config = fetch_config_from_yaml(path_to_config)

        print(parsed_config.data)

    _config = Config(
        app_config=AppConfig(**parsed_config.data)
    )
    return _config


config = create_and_validate_config()
print(f"Config: {config}")
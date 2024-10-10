from pathlib import Path

from typing import Optional, List

from pydantic import BaseModel
from strictyaml import YAML, load

import classification

PACKAGE_ROOT = Path(classification.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
DATASET_DIR = PACKAGE_ROOT / "datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"

# SOURCE_DATA = 

# print(f"Package root from core {PACKAGE_ROOT}")
# print(f"Config file path {CONFIG_FILE_PATH}")
# print(f"Data source to be saved in {DATASET_DIR}  ")

class AppConfig(BaseModel):
    package_name: str
    data_source: str
    saved_data: str
    pipeline_save_file: str

class ModelConfig(BaseModel):
    target: str
    random_state: int
    test_size: float
    categorical_variables : List[str]
    numerical_variables: List[str]
    cabin: List[str]


class Config(BaseModel):
    app_config: AppConfig
    themodel_config: ModelConfig


_config_instance = None

def find_config_file() -> Path:
    """Locate the configuration file."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")

def fetch_config_from_yaml(cfg_path:Optional[Path]= None)-> YAML:
    if not cfg_path:
        cfg_path = find_config_file()
    if cfg_path:
        # print("Opening and reading Yaml")
        with open(cfg_path,"r")as conf_file:
            parsed_config = load(conf_file.read())
            # print(f"parsed config type {type(parsed_config)}")
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")
    
def create_and_validate_config(parsed_config: YAML = None) -> Config:
    if parsed_config is None:
        # print("Parsed config is None")
        parsed_config = fetch_config_from_yaml()

        # print(parsed_config.data)

    _config = Config(
        app_config=AppConfig(**parsed_config.data),
        themodel_config = ModelConfig(**parsed_config.data)
    )
    return _config

def get_config() -> Config:
    global _config_instance
    if _config_instance is None:
        _config_instance = create_and_validate_config()
    return _config_instance


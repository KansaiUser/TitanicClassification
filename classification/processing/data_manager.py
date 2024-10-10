from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger

from classification.config.core import get_config,DATASET_DIR
from classification.processing.auxiliar import get_first_cabin, get_title

config = get_config()

def pre_pipeline_preparation(dataframe: pd.DataFrame) -> pd.DataFrame:
    # replace question marks with NaN values
    data = dataframe.replace("?", np.nan)

    # retain only the first cabin if more than
    # 1 are available per passenger
    data["cabin"] = data["cabin"].apply(get_first_cabin)

    data["title"] = data["name"].apply(get_title)

    # cast numerical variables as floats
    data["fare"] = data["fare"].astype("float")
    data["age"] = data["age"].astype("float")

    # drop unnecessary variables
    data.drop(labels=config.themodel_config.unused_fields, axis=1, inplace=True)

    return data


def load_dataset(reread:bool):
    saved_file = config.app_config.saved_data
    saved_path = Path(f"{DATASET_DIR}/{saved_file}")
    logger.info(f"Saved data will be stored in {saved_path}")
    source = config.app_config.data_source

    if reread:
        data = pd.read_csv(source)

        data = pre_pipeline_preparation(data)
        
        # Then save the file
        data.to_csv(saved_path, index=False)
        logger.info(f"Preprocessed data saved to {saved_path}")
    else:
        if saved_path.exists():
            data = pd.read_csv(saved_path)
        else:
            logger.error(f"{saved_path} does not exist.")
            return None

    return data


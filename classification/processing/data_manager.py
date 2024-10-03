from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger

from classification.config.core import get_config,DATASET_DIR
from classification.processing.auxiliar import get_first_cabin, get_title

config = get_config()

def load_dataset(reread:bool):
    saved_file = config.app_config.saved_data
    saved_path = Path(f"{DATASET_DIR}/{saved_file}")
    logger.info(f"Saved data in {saved_path}")
    source = config.app_config.data_source

    if reread:
        data = pd.read_csv(source)
        # replace interrogation marks by NaN values
        data = data.replace('?', np.nan)
        # Other preprocessing here

        data['cabin'] = data['cabin'].apply(get_first_cabin)
        data['title'] = data['name'].apply(get_title)

        data['fare'] = data['fare'].astype('float')
        data['age'] = data['age'].astype('float')
        # drop unnecessary variables
        data.drop(labels=['name','ticket', 'boat', 'body','home.dest'], axis=1, inplace=True)

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


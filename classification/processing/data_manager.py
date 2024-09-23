from pathlib import Path
import pandas as pd
import numpy as np

from classification.config.core import get_config,DATASET_DIR

config = get_config()

def load_dataset(reread:bool):
    saved_file = config.app_config.saved_data
    saved_path = Path(f"{DATASET_DIR}/{saved_file}")
    print(f"Saved data in {saved_path}")
    source = config.app_config.data_source

    if reread:
        data = pd.read_csv(source)
        # replace interrogation marks by NaN values
        data = data.replace('?', np.nan)
        # Other preprocessing here

        # Then save the file
        data.to_csv(saved_path, index=False)
        print(f"Preprocessed data saved to {saved_path}")
    else:
        if saved_path.exists():
            data = pd.read_csv(saved_path)
        else:
            print(f"{saved_path} does not exist.")
            return None

    return data


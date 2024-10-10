from loguru import logger
import pandas as pd
from typing import Union
from classification.pipeline import get_trained_pipeline
from classification import __version__
from classification.config.core import get_config
from classification.processing.data_manager import pre_pipeline_preparation
from pathlib import Path

config = get_config()

pipename = config.app_config.pipeline_save_file + __version__
logger.info(f"The pipeline name is {pipename}")
script_path = Path(__file__).parent

def make_prediction(input_data: Union[pd.DataFrame, dict],is_cleaned: bool=False) -> dict:
    global pipename
    data = pd.DataFrame(input_data)
    errors=""
    #If it is cleaned then skip the cleaning
    if not is_cleaned:
        logger.info("Data will be cleaned")
        data= pre_pipeline_preparation(data)
    #Now data is cleaned

    #Perhaps some validation here

    #pipeline predict
    pipename=script_path/"models"/pipename
    tit_pipe= get_trained_pipeline(pipename)
    if tit_pipe is None:
        logger.error(f"{pipename} is not a valid Pipeline.")
        return {"predictions": None, "version": __version__, "errors": f"{pipename} is not a valid Pipeline."}
    
     # Drop the target column
    try:
        data = data.drop(config.themodel_config.target, axis=1)
    except KeyError:
        logger.warning(f"Target column {config.themodel_config.target} not found in the input data.")
        logger.warning("This prediction is on unlabeled data")
        errors += f"Target column {config.themodel_config.target} not found. Predicion on unlabeled data"
    
    # Make predictions
    try:
        predictions = tit_pipe.predict(data)
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return {"predictions": None, "version": __version__, "errors": f"Prediction failed: {str(e)}"}

    results = {
            "predictions": predictions,
            "version": __version__,
            "errors": errors,
        }
    return results

if __name__ == "__main__":
    data = {
    "pclass": [1, 1],
    "survived": [1, 1],
    "name": ["Allen, Miss. Elisabeth Walton", "Allison, Master. Hudson Trevor"],
    "sex": ["female", "male"],
    "age": [29, 0.9167],
    "sibsp": [0, 1],
    "parch": [0, 2],
    "ticket": ["24160", "113781"],
    "fare": [211.3375, 151.55],
    "cabin": ["B5", "C22 C26"],
    "embarked": ["S", "S"],
    "boat": ["2", "11"],
    "body": [None, None],
    "home.dest": ["St Louis, MO", "Montreal, PQ / Chesterville, ON"]
}

# Convert to DataFrame
input_data = pd.DataFrame(data)

# Print the DataFrame for confirmation
print(input_data)

results = make_prediction(input_data)

print(results)
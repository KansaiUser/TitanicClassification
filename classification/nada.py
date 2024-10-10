from classification import __version__ as version
from config.core import get_config
from loguru import logger
from pathlib import Path

from processing.data_manager import load_dataset
from sklearn.model_selection import train_test_split

from classification.pipeline import get_pipeline, save_trained_pipeline

config = get_config()

script_path = Path(__file__).parent

def run(reread:bool)-> None:
    data = load_dataset(reread)
    if data is None:
        logger.error(f"No data loaded")
        return
    
    print(data.head())

    print(config)

    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(config.themodel_config.target, axis=1),  # predictors
        data[config.themodel_config.target],  # target
        test_size=config.themodel_config.test_size,  # percentage of obs in test set
        random_state=config.themodel_config.random_state)  # seed to ensure reproducibility

    # print(f" Xtrain shape {X_train.shape} ,Xtest shape {X_test.shape}")
    # print("Train Data")
    # print(X_train.head())
    # print("Test Data")
    # print(X_test.head())

    titanic_pipe = get_pipeline()

    titanic_pipe.fit(X_train, y_train)

    pipeline_name = Path(config.app_config.pipeline_save_file + version)
    pipeline_name =script_path/"models"/pipeline_name
    logger.info(f"Saving pipeline to {pipeline_name}")
    save_trained_pipeline(titanic_pipe,pipeline_name)



if __name__ == "__main__":
    run(True) #Reread value means to reread the source
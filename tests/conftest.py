import pytest

from classification.config.core import get_config,DATASET_DIR
from classification.processing.data_manager import load_dataset
from sklearn.model_selection import train_test_split

config = get_config()


@pytest.fixture
def sample_input_data():
    data = load_dataset(False)

    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(config.themodel_config.target, axis=1),  # predictors
        data[config.themodel_config.target],  # target
        test_size=config.themodel_config.test_size,  # percentage of obs in test set
        random_state=config.themodel_config.random_state)  # seed to ensure reproducibility

    return X_test    


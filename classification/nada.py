from classification import __version__ as version
from config.core import get_config

from processing.data_manager import load_dataset
from sklearn.model_selection import train_test_split

config = get_config()

def run(reread:bool)-> None:
    data = load_dataset(reread)
    if data is None:
        print(f"No data loaded")
        return
    
    print(data.head())

    print(config)

    # X_train, X_test, y_train, y_test = train_test_split(
    # data.drop(config.model_config.target, axis=1),  # predictors
    # data[config.model_config.target],  # target
    # test_size=0.2,  # percentage of obs in test set
    # random_state=0)  # seed to ensure reproducibility

    # print(f" Xtrain shape {X_train.shape} ,Xtest shape {X_test.shape}")



if __name__ == "__main__":
    run(True) #Reread value means to reread the source
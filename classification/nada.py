from classification import __version__ as version
from config.core import get_config

from processing.data_manager import load_dataset

# print(f"Version from nada {version}")
# print(f"Config from nada {get_config()}")

# config = get_config()

def run(reread:bool)-> None:
    data = load_dataset(reread)
    if data is None:
        print(f"No data loaded")
        return
    
    print(data.head())


if __name__ == "__main__":
    run(True) #Reread value means to reread the source
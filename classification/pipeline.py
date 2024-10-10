from sklearn.pipeline import Pipeline
from pathlib import Path
from loguru import logger

import joblib

# feature scaling
from sklearn.preprocessing import StandardScaler

# to build the models
from sklearn.linear_model import LogisticRegression

# for imputation
from feature_engine.imputation import (
    CategoricalImputer,
    AddMissingIndicator,
    MeanMedianImputer)

# for encoding categorical variables
from feature_engine.encoding import (
    RareLabelEncoder,
    OneHotEncoder
)
from operations import ExtractLetterTransformer
from config.core import get_config

config = get_config()


def get_pipeline():
    # set up the pipeline
    titanic_pipe = Pipeline([

    # ===== IMPUTATION =====
    # impute categorical variables with string missing
    ('categorical_imputation', CategoricalImputer(
        imputation_method='missing', variables=config.themodel_config.categorical_variables)),

    # add missing indicator to numerical variables
    ('missing_indicator', AddMissingIndicator(variables=config.themodel_config.numerical_variables)),

    # impute numerical variables with the median
    ('median_imputation', MeanMedianImputer(
        imputation_method='median', variables=config.themodel_config.numerical_variables)),


    # Extract letter from cabin
    ('extract_letter', ExtractLetterTransformer(variables=config.themodel_config.cabin)),


    # == CATEGORICAL ENCODING ======
    # remove categories present in less than 5% of the observations (0.05)
    # group them in one category called 'Rare'
    ('rare_label_encoder', RareLabelEncoder(
        tol=0.05, n_categories=1, variables=config.themodel_config.categorical_variables)),


    # encode categorical variables using one hot encoding into k-1 variables
    ('categorical_encoder', OneHotEncoder(
        drop_last=True, variables=config.themodel_config.categorical_variables)),

    # scale
    ('scaler', StandardScaler()),

    ('Logit', LogisticRegression(C= config.themodel_config.c_value, random_state= config.themodel_config.random_state)),
    ])

    return titanic_pipe

def save_trained_pipeline(pipe,name):
    pname = Path(name)
    
    # Check if the suffix is already '.joblib'
    if pname.suffix != '.joblib':
        # Append .joblib to the existing name
        name = str(pname) + '.joblib'
    
    logger.info(f"Save to {name}")
    joblib.dump(pipe, name)

def get_trained_pipeline(name):
    if not Path(name).is_file():
        logger.error("f{name} pipeline file does not exist!")
        return None
    return joblib.load(name)

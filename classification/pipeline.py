from sklearn.pipeline import Pipeline
from pathlib import Path
from loguru import logger

import joblib
from feature_engine.imputation import (
    CategoricalImputer,
    AddMissingIndicator,
    MeanMedianImputer)

from config.core import get_config

config = get_config()


def get_pipeline():
    # set up the pipeline
    titanic_pipe = Pipeline([

        # ===== IMPUTATION =====
        # impute categorical variables with string missing
        ('categorical_imputation', CategoricalImputer(
            imputation_method='missing', variables=config.themodel_config.categorical_variables)),
    ])

    return titanic_pipe

def save_pipeline(pipe,name):

    if Path(name).suffix!='joblib':
        pname=Path(name).with_suffix('.joblib')
        name=str(pname)
    
    logger.info(f"Save to {name}")
    joblib.dump(pipe, name)
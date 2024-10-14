from pydantic import BaseModel, ValidationError
from typing import Optional, List
from loguru import logger
import numpy as np

def validate_data(data):
    error= None
    vdata= data.copy()
    try: 
        MultipleTitanicDataInputs(
            input=vdata.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as e:
        logger.error(f"Validation Error: {e.errors()}")
        error = e.json()
        print(e.json)

    return data, error


class TitanicDataInputSchema(BaseModel):
    pclass: int 
    survived: Optional[int]
    sex: str
    age: float
    sibsp: int
    parch: int
    fare: float
    cabin: str
    embarked: str
    title: str

class MultipleTitanicDataInputs(BaseModel):
    input: List[TitanicDataInputSchema]




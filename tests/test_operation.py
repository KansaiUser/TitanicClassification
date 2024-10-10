
from classification.operations import ExtractLetterTransformer
from classification.config.core import get_config
import math
config = get_config()

def test_extract_letter(sample_input_data):
    transformer = ExtractLetterTransformer(
        variables=config.themodel_config.cabin
        )
    # print(type(sample_input_data))
    # print(sample_input_data.head(15))
    
    assert math.isnan(sample_input_data['cabin'].iat[3])
    assert sample_input_data['cabin'].iat[7]=='C104'

    transformed_data = transformer.fit_transform(sample_input_data)

    assert math.isnan(transformed_data['cabin'].iat[3])
    assert transformed_data['cabin'].iat[7]=='C'
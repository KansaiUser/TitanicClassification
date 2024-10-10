
from classification.operations import ExtractLetterTransformer
from classification.config.core import get_config
import math
import pandas as pd

config = get_config()

# def test_extract_letter(sample_input_data):
#     transformer = ExtractLetterTransformer(
#         variables=config.themodel_config.cabin
#         )
#     # print(type(sample_input_data))
#     # print(sample_input_data.head(15))
    
#     assert math.isnan(sample_input_data['cabin'].iat[3])
#     assert sample_input_data['cabin'].iat[7]=='C104'

#     transformed_data = transformer.fit_transform(sample_input_data)

#     assert math.isnan(transformed_data['cabin'].iat[3])
#     assert transformed_data['cabin'].iat[7]=='C'

def test_extract_letter_transformer(sample_input_data, expected_output_data):
    # Instantiate the transformer
    transformer = ExtractLetterTransformer(variables=['cabin'])
    
    # Perform transformation
    transformed_data = transformer.fit_transform(sample_input_data)
    
    # Assert the transformation result matches expected output
    pd.testing.assert_frame_equal(transformed_data, expected_output_data)
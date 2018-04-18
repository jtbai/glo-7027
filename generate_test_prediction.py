import json
import pickle
import time
from os import path

from sklearn import model_selection

from config.classification_configuration_factory import ClassifierConfigurationFactory
from data.data_transformation_factory import DataTransformationFactory

CONFIG_PATH = "config"
DATA_PATH = "data"

input_data_file_name = 'test_prepared_data.pyk'
config_file_name = "test_output_configuration_no_transformation.json"

config_dictionary = json.load(open(path.join(CONFIG_PATH, config_file_name)))
Config = ClassifierConfigurationFactory(config_dictionary)
Transformation = DataTransformationFactory(config_dictionary)

if __name__ == '__main__':
    test = pickle.load(open(path.join(DATA_PATH, input_data_file_name), 'rb'))
    test = test.loc[:, test.columns != "SalePrice"]

    transformed_test_x = Transformation.transform(test)

    for model_to_execute in Config.get_jobs():
        model_to_execute.execute(None, None, transformed_test_x, None)


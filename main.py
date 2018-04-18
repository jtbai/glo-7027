import json
import pickle
import time
from os import path

from sklearn import model_selection

from config.classification_configuration_factory import ClassifierConfigurationFactory
from data.data_transformation_factory import DataTransformationFactory

CONFIG_PATH = "config"
DATA_PATH = "data"

input_data_file_name = 'train_prepared_data.pyk'
# config_file_name = "initial_run.json"
config_file_name = "test_boostin_no_pca.json"

config_dictionary = json.load(open(path.join(CONFIG_PATH, config_file_name)))
Config = ClassifierConfigurationFactory(config_dictionary)
Transformation = DataTransformationFactory(config_dictionary)

if __name__ == '__main__':
    start_time = time.time()
    train = pickle.load(open(path.join(DATA_PATH, input_data_file_name), 'rb'))

    X = train.loc[:, train.columns != "SalePrice"]
    y = train.SalePrice

    transformed_train_x = Transformation.fit_transform(X)

    for model_to_execute in Config.get_jobs():
        model_to_execute.execute(transformed_train_x, y, None, None)


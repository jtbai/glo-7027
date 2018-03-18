import json
import pickle
import time
from os import path

from config.classification_configuration_factory import ClassifierConfigurationFactory
from data.data_transformation_factory import DataTransformationFactory

CONFIG_PATH = "config"
DATA_PATH = "data"

input_data_file_name = 'train_prepared_data.pyk'
config_file_name = "train_final_model.json"

config_dictionary = json.load(open(path.join(CONFIG_PATH, config_file_name)))
Config = ClassifierConfigurationFactory(config_dictionary)
Transformation = DataTransformationFactory(config_dictionary)

if __name__ == '__main__':
    start_time = time.time()
    train = pickle.load(open(path.join(DATA_PATH, input_data_file_name), 'rb'))

    X = train.loc[:, train.columns != "SalePrice"]
    y = train.SalePrice
    fold_start_time = time.time()

    transformed_train_x = Transformation.fit_transform(X)

    for model_to_execute in Config.get_jobs():
        model_to_execute.execute(transformed_train_x, y, None, None)

    for model_to_execute in Config.get_jobs():
        model_to_execute.train_model = False
        model_to_execute.execute(None, None, transformed_train_x, None)

    fold_finish_time = time.time()
    fold_time = fold_finish_time - fold_start_time
    total_time = fold_finish_time - start_time
    time_record = "{} : {}s (total: {})".format(1, fold_time, total_time)
    print(time_record)
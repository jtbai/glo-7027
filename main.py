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
config_file_name = "initial_run.json"

config_dictionary = json.load(open(path.join(CONFIG_PATH, config_file_name)))
Config = ClassifierConfigurationFactory(config_dictionary)
Transformation = DataTransformationFactory(config_dictionary)

if __name__ == '__main__':
    start_time = time.time()
    train = pickle.load(open(path.join(DATA_PATH, input_data_file_name), 'rb'))

    X = train.loc[:, train.columns != "SalePrice"]
    y = train.SalePrice
    folds = 10
    k_fold = model_selection.KFold(n_splits=folds)

    for index, k_fold_indexes in enumerate(k_fold.split(X)):
        fold_start_time = time.time()

        X_train = X.iloc[k_fold_indexes[0]]
        y_train = y.iloc[k_fold_indexes[0]]
        X_test = X.iloc[k_fold_indexes[1]]
        y_test = y.iloc[k_fold_indexes[1]]

        transformed_train_x = Transformation.fit_transform(X_train)
        transformed_test_x = Transformation.fit_transform(X_test)

        for model_to_execute in Config.get_jobs():
            model_to_execute.execute(X_train, y_train, X_test, y_test)

        fold_finish_time = time.time()
        fold_time = fold_finish_time - fold_start_time
        total_time = fold_finish_time - start_time
        time_record = "fold {}/{} : {}s (total: {})".format(index+1, len(X), fold_time, total_time)
        print(time_record)
        with open("time_record.txt",'a') as time_file:
            time_file.write("{}\n".format(time_record))
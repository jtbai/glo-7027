from config_classifier import ConfigClassifier
from sklearn import model_selection
import pickle
import json
import time
from os import path

CONFIG_PATH = "config"
DATA_PATH = "data"

input_data_file_name = 'prepared_data.pyk'
config_file_name = "initial_run.json"

config_dictionary = json.load(open(path.join(CONFIG_PATH, config_file_name)))
Config = ConfigClassifier(config_dictionary)

if __name__ == '__main__':
    start_time = time.time()
    train = pickle.load(open(path.join(DATA_PATH, input_data_file_name), 'rb'))

    X = train.loc[:, train.columns != "SalePrice"]
    y = train.SalePrice

    k_fold = model_selection.LeaveOneOut()
    for index, k_fold_indexes in enumerate(k_fold.split(X)):
        fold_start_time = time.time()

        X_train = X.iloc[k_fold_indexes[0]]
        y_train = y.iloc[k_fold_indexes[0]]
        X_test = X.iloc[k_fold_indexes[1]]
        y_test = y.iloc[k_fold_indexes[1]]

        for model_to_execute in Config.get_jobs():
            model_to_execute.execute(X_train, y_train, X_test, y_test)

        fold_finish_time = time.time()
        fold_time = fold_finish_time - fold_start_time
        total_time = fold_finish_time - start_time
        time_record = "fold {}/{} : {}s (total: {})".format(index+1, len(X), fold_time, total_time)
        print(time_record)
        with open("time_record.txt",'w') as time_file:
            time_file.write("{}\n".format(time_record))
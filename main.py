from config_classifier import ConfigClassifier
from sklearn import model_selection
import pickle
import json
import time

config_dictionary = json.load(open("initial_run_test.json"))
Config = ConfigClassifier(config_dictionary)



if __name__ == '__main__':


    train = pickle.load(open('./prepared_data.pyk', 'rb'))

    X = train.loc[:, train.columns != "SalePrice"]
    # X = train.loc[:, train.columns[[range(1, 6)]]]
    y = train.SalePrice

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)

    for model_to_execute in Config.get_jobs():
        model_to_execute.execute(X_train, y_train, X_test, y_test)


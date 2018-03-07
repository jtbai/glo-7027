import pandas as pd
from sklearn import linear_selection, metrics, linear_model, decomposition, ensemble

train = pd.read_csv("train.csv")  # use new dataset

def regressionLineaire(X_train, y_train, X_test, y_test):
    pass

def regressionGLM(X_train, y_train, X_test, y_test):
    pass

def regressionGAM(X_train, y_train, X_test, y_test):
    pass

def regressionRandomForest(X_train, y_train, X_test, y_test):
    parametres = {'min_samples_leaf': [1, 5, 10, 20, 50],
                  'n_estimators': [50, 100, 250]}
    gridSearch = model_selection.GridSearchCV(ensemble.RandomForestClassifier(oob_score=True), parametres, n_jobs=6)
    gridSearch = gridSearch.fit(X_train, y_train)

    y_prediction_test = gridSearch.predict(X_test_reduced)
    y_prediction_train = gridSearch.predict(X_train_reduced)
    yPredTestProb = gridSearch.predict_proba(X_test_reduced)[:, 1]
    print(yPredTestProb)

def regressionGradientBoosting(X_train, y_train, X_test, y_test):
    pass

def regressionSVM(X_train, y_train, X_test, y_test):
    parametres = {'gamma': [0.01, 0.1, 1], 'C': [1, 10, 100]}
    gridSearch = model_selection.GridSearchCV(svm.SVC(), parametres, n_jobs=6)
    gridSearch = gridSearch.fit(X_train, y_train)

    y_prediction_test = gridSearch.predict(X_test_reduced)
    y_prediction_train = gridSearch.predict(X_train_reduced)
    yPredTestProb = gridSearch.predict_proba(X_test_reduced)[:, 1]
    print(yPredTestProb)

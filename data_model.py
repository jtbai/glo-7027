import config_classifier
from config_classifier import config_classifier
from pickle import load
from sklearn import model_selection, metrics, svm, linear_model, ensemble, decomposition
import time
from sklearn.externals import joblib

CONFIGS = config_classifier()


def printResults(nomRegresseur, y_train, y_train_pred, y_test, y_test_pred, bestParams):
    result_file = open("Resultats\\" + nomRegresseur + ".txt", "w")
    result_file.write("Meilleurs parametres : " + bestParams + "\n")
    result_file.write("Score train : " + str(metrics.r2_score(y_train, y_train_pred)) + "\n")
    result_file.write("Score test : " + str(metrics.r2_score(y_test, y_test_pred)) + "\n")
    # result_file.write("Score test : " + str(metrics.roc_auc_score(y_test, y_test_pred)) + "\n")


def regressionSVM(X_train, y_train, X_test, y_test):
    # X_PCA = decomposition.PCA(n_components=10).fit(X_train)
    # X_train_reduced = X_PCA.transform(X_train)
    # X_test_reduced = X_PCA.transform(X_test)

    parametres = {'gamma': [0.01, 0.1, 1], 'C': [1, 10, 100]}
    if CONFIGS.retrain_svm:
        gridSearch = model_selection.GridSearchCV(svm.SVR(), parametres, n_jobs=6)
        gridSearch = gridSearch.fit(X_train, y_train)
        joblib.dump(gridSearch, 'gridSearchSVM.pkl')
    else:
        gridSearch = joblib.load('gridSearchSVM.pkl')

    y_train_pred = gridSearch.predict(X_train)
    y_test_pred = gridSearch.predict(X_test)
    printResults("svm", y_train, y_train_pred, y_test, y_test_pred, str(gridSearch.best_params_))


def regressionLineaireSimple(X_train, y_train, X_test, y_test):
    if CONFIGS.retrain_linear_model:
        gridSearch = linear_model.LinearRegression()
        gridSearch = gridSearch.fit(X_train, y_train)
        joblib.dump(gridSearch, 'gridSearchLinearRegression.pkl')
    else:
        gridSearch = joblib.load('gridSearchLinearRegression.pkl')

    y_train_pred = gridSearch.predict(X_train)
    y_test_pred = gridSearch.predict(X_test)
    printResults("linear_models", y_train, y_train_pred, y_test, y_test_pred, "NA")


def regressionRandomForest(X_train, y_train, X_test, y_test):
    parametres = {'min_samples_leaf': [1, 5, 10, 20, 50], 'n_estimators': [50, 100, 250]}
    if CONFIGS.retrain_random_forest:
        gridSearch = model_selection.GridSearchCV(ensemble.RandomForestRegressor(oob_score=True), parametres, n_jobs=6)
        gridSearch = gridSearch.fit(X_train, y_train)
        joblib.dump(gridSearch, 'gridSearchRandomForest.pkl')
    else:
        gridSearch = joblib.load('gridSearchRandomForest.pkl')

    y_train_pred = gridSearch.predict(X_train)
    y_test_pred = gridSearch.predict(X_test)
    printResults("random_forest", y_train, y_train_pred, y_test, y_test_pred, str(gridSearch.best_params_))


def regressionGLM(X_train, y_train, X_test, y_test):
    pass


def regressionGAM(X_train, y_train, X_test, y_test):
    # Utiliser py-earth ?
    pass


def regressionGradientBoosting(X_train, y_train, X_test, y_test):
    parametres = {'max_depth': [3, 5, 10], 'n_estimators': [50, 100, 250]}
    if CONFIGS.retrain_gradient_boosting:
        gridSearch = model_selection.GridSearchCV(ensemble.GradientBoostingRegressor(), parametres, n_jobs=6)
        gridSearch = gridSearch.fit(X_train, y_train)
        joblib.dump(gridSearch, 'gridSearchGradientBoosting.pkl')
    else:
        gridSearch = joblib.load('gridSearchGradientBoosting.pkl')

    y_train_pred = gridSearch.predict(X_train)
    y_test_pred = gridSearch.predict(X_test)
    printResults("gradient_boosting", y_train, y_train_pred, y_test, y_test_pred, str(gridSearch.best_params_))


if __name__ == '__main__':
    train = load(open('./prepared_data.pyk', 'rb'))

    # X = train.loc[:, train.columns != "SalePrice"]
    X = train.loc[:, train.columns[[range(1, 3)]]]
    y = train.SalePrice

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)

    if CONFIGS.use_linear_model:
        print('Linear model')
        start_time = time.time()
        regressionLineaireSimple(X_train, y_train, X_test, y_test)
        print("--- %s seconds ---" % (time.time() - start_time))
        print('')

    if CONFIGS.use_generalised_linear_model:
        print('GLM')
        start_time = time.time()
        regressionGLM(X_train, y_train, X_test, y_test)
        print("--- %s seconds ---" % (time.time() - start_time))
        print('')

    if CONFIGS.use_generalised_additive_model:
        print('GAM')
        start_time = time.time()
        regressionGAM(X_train, y_train, X_test, y_test)
        print("--- %s seconds ---" % (time.time() - start_time))
        print('')

    if CONFIGS.use_gradient_boosting:
        print('Gradient Boosting')
        start_time = time.time()
        regressionGradientBoosting(X_train, y_train, X_test, y_test)
        print("--- %s seconds ---" % (time.time() - start_time))
        print('')

    if CONFIGS.use_svm:
        print('SVM')
        start_time = time.time()
        regressionSVM(X_train, y_train, X_test, y_test)
        print("--- %s seconds ---" % (time.time() - start_time))
        print('')

    if CONFIGS.use_random_forest:
        print('Random forest')
        start_time = time.time()
        regressionRandomForest(X_train, y_train, X_test, y_test)
        print("--- %s seconds ---" % (time.time() - start_time))
        print('')

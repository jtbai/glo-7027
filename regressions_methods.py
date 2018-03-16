from sklearn.externals import joblib
from sklearn import model_selection, metrics, svm, linear_model, ensemble, decomposition
from time import time
from os import path


class model:

    train_model = None
    run_model = None

    def __init__(self, nom_regresseur, train_model_pickle_path, output_path):
        self.nom_regresseur = nom_regresseur
        self.output_path = output_path
        self.trained_model_pickle_path = train_model_pickle_path

    def _get_grid_search_pickel_name(self):
        return "grid_search_{}.pkl".format(self.nom_regresseur)

    def _get_trained_model_pickel_path(self):
        return path.join(self.trained_model_pickle_path, self._get_grid_search_pickel_name())

    def _get_grid_search(self, X_train, y_train):
        if self.train_model:
            grid = self._train(X_train, y_train)
            joblib.dump(grid, path.join(self.trained_model_pickle_path, self._get_grid_search_pickel_name()))
        else:
            grid = joblib.load(self._get_trained_model_pickel_path())

        return grid

    def _run(self, trained_grid_search, X_train, y_train, X_test, y_test):
        pass

    def _train(self, X_train, y_train):
        pass

    def execute(self, X_train, y_train, X_test, y_test):
        start_time = time()
        grid_search = self._get_grid_search(X_train, y_train)
        print(self.nom_regresseur)
        if self.run_model:
            self._run(grid_search, X_train, y_train, X_test, y_test)
        run_time = time() - start_time
        print("--- {} seconds ---".format(run_time))
        print('')

    def _printResults(self, y_train, y_train_pred, y_test, y_test_pred, bestParams):
        result_file = open("Resultats/{}.txt".format(self.nom_regresseur), "w")
        result_file.write("Meilleurs parametres : " + bestParams + "\n")
        result_file.write("Score train : " + str(metrics.r2_score(y_train, y_train_pred)) + "\n")
        result_file.write("Score test : " + str(metrics.r2_score(y_test, y_test_pred)) + "\n")
        # result_file.write("Score test : " + str(metrics.roc_auc_score(y_test, y_test_pred)) + "\n")


class SVM(model):

    def _train(self, X_train, y_train):
        parametres = {'gamma': [0.01, 0.1, 1], 'C': [1, 10, 100]}
        grid_search = model_selection.GridSearchCV(svm.SVR(), parametres, n_jobs=6)
        grid_search = grid_search.fit(X_train, y_train)
        return grid_search

    def _run(self, trained_grid_search, X_train, y_train, X_test, y_test):
        y_train_pred = trained_grid_search.predict(X_train)
        y_test_pred = trained_grid_search.predict(X_test)
        self._printResults(y_train, y_train_pred, y_test, y_test_pred, str(trained_grid_search.best_params_))

class RegressionLineaireSimple(model):

    def _train(self, X_train, y_train):
        gridSearch = linear_model.LinearRegression()
        gridSearch = gridSearch.fit(X_train, y_train)
        return gridSearch

    def _run(self, trained_grid_search, X_train, y_train, X_test, y_test):
        y_train_pred = trained_grid_search.predict(X_train)
        y_test_pred = trained_grid_search.predict(X_test)
        self._printResults(y_train, y_train_pred, y_test, y_test_pred, "NA")
#
# def regressionLineaireSimple(X_train, y_train, X_test, y_test):
#     if CONFIGS.retrain_linear_model:
#
#     else:
#         gridSearch = joblib.load('gridSearchLinearRegression.pkl')
#
#
#
#
# def regressionRandomForest(X_train, y_train, X_test, y_test):
#     parametres = {'min_samples_leaf': [1, 5, 10, 20, 50], 'n_estimators': [50, 100, 250]}
#     if CONFIGS.retrain_random_forest:
#         gridSearch = model_selection.GridSearchCV(ensemble.RandomForestRegressor(oob_score=True), parametres, n_jobs=6)
#         gridSearch = gridSearch.fit(X_train, y_train)
#         joblib.dump(gridSearch, 'gridSearchRandomForest.pkl')
#     else:
#         gridSearch = joblib.load('gridSearchRandomForest.pkl')
#
#     y_train_pred = gridSearch.predict(X_train)
#     y_test_pred = gridSearch.predict(X_test)
#     printResults("random_forest", y_train, y_train_pred, y_test, y_test_pred, str(gridSearch.best_params_))
#
#
# def regressionGLM(X_train, y_train, X_test, y_test):
#     pass
#
#
# def regressionGAM(X_train, y_train, X_test, y_test):
#     # Utiliser py-earth ?
#     pass
#
#
# def regressionGradientBoosting(X_train, y_train, X_test, y_test):
#     parametres = {'max_depth': [3, 5, 10], 'n_estimators': [50, 100, 250]}
#     if CONFIGS.retrain_gradient_boosting:
#         gridSearch = model_selection.GridSearchCV(ensemble.GradientBoostingRegressor(), parametres, n_jobs=6)
#         gridSearch = gridSearch.fit(X_train, y_train)
#         joblib.dump(gridSearch, 'gridSearchGradientBoosting.pkl')
#     else:
#         gridSearch = joblib.load('gridSearchGradientBoosting.pkl')
#
#     y_train_pred = gridSearch.predict(X_train)
#     y_test_pred = gridSearch.predict(X_test)
#     printResults("gradient_boosting", y_train, y_train_pred, y_test, y_test_pred, str(gridSearch.best_params_))

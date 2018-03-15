from sklearn.externals import joblib
from sklearn import model_selection, metrics, svm, linear_model, ensemble, decomposition
from time import time
from os import path
import numpy as np

class BaseModel:

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
        y_train_predicted = trained_grid_search.predict(X_train)
        y_test_predicted = trained_grid_search.predict(X_test)

        return y_train_predicted, y_test_predicted

    def compute_and_output_kaggle_score(self, y_test, y_test_pred,):
        self._output_kaggle_evaluation_metric(y_test, y_test_pred)

    def compute_and_output_r2_metric(self, trained_grid_search, y_train, y_train_pred, y_test, y_test_pred):
        pass

    def _train(self, X_train, y_train):
        pass

    def execute(self, X_train, y_train, X_test, y_test):
        start_time = time()
        grid_search = self._get_grid_search(X_train, y_train)
        print(self.nom_regresseur)
        if self.run_model:
            y_predicted_train, y_predicted_test = self._run(grid_search, X_train, y_train, X_test, y_test)
            self.compute_and_output_kaggle_score(y_test, y_predicted_test)
            self.compute_and_output_r2_metric(grid_search, y_train, y_predicted_train, y_test, y_predicted_test)
        run_time = time() - start_time
        # print("--- {} seconds ---".format(run_time))
        # print('')

    def _output_kaggle_evaluation_metric(self, y_test, y_test_pred):
        kaggle_metric = np.sqrt(np.mean(np.power((np.log10(y_test)-np.log10(y_test_pred)),2)))

        with open(path.join(self.output_path, "kaggle_{}.txt".format(self.nom_regresseur)), "a") as output_file:
            output_file.write("{}\n".format(kaggle_metric))

    def _printResults(self, y_train, y_train_pred, y_test, y_test_pred, bestParams):
        with open(path.join(self.output_path, "{}.txt".format(self.nom_regresseur)), "a") as result_file:

            result_file.write("Meilleurs parametres : " + bestParams + "\n")

            result_file.write("Score train : " + str(metrics.r2_score(y_train, y_train_pred)) + "\n")
            result_file.write("Score test : " + str(metrics.r2_score(y_test, y_test_pred)) + "\n")

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

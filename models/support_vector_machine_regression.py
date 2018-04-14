from models.basemodel import BaseModel
from sklearn import model_selection, svm


class SupportVectorMachineRegression(BaseModel):

    def _train(self, X_train, y_train):
        parametres = {'gamma': [1e-10, 1e-11, 1e-12, 1e-13], 'C': [1e3, 1e4, 1e5, 1e6]}
        grid_search = model_selection.GridSearchCV(svm.SVR(), parametres, n_jobs=6)
        grid_search = grid_search.fit(X_train, y_train)
        return grid_search

    def compute_and_output_r2_metric(self, trained_grid_search, y_train, y_train_pred, y_test, y_test_pred):
        self._printResults(y_train, y_train_pred, y_test, y_test_pred, str(trained_grid_search.best_params_))

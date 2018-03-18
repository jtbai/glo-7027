from models.basemodel import BaseModel
from sklearn import ensemble, model_selection


class RandomForestRegression(BaseModel):

    def _train(self, X_train, y_train):
        parametres = {'min_samples_leaf': [1, 5, 10, 20, 50], 'n_estimators': [50, 100, 250]}
        grid_search = model_selection.GridSearchCV(ensemble.RandomForestRegressor(oob_score=True), parametres, n_jobs=6)
        grid_search = grid_search.fit(X_train, y_train)

        return grid_search

    def compute_and_output_r2_metric(self, trained_grid_search, y_train, y_train_pred, y_test, y_test_pred):
        self._printResults(y_train, y_train_pred, y_test, y_test_pred, str(trained_grid_search.best_params_))
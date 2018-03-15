from models.basemodel import BaseModel
from sklearn import linear_model


class SimpleLinearRegression(BaseModel):

    def _train(self, X_train, y_train):
        grid_search = linear_model.LinearRegression()
        grid_search = grid_search.fit(X_train, y_train)

        return grid_search

    def compute_and_output_r2_metric(self, trained_grid_search, y_train, y_train_pred, y_test, y_test_pred):
        self._printResults(y_train, y_train_pred, y_test, y_test_pred, "NA")
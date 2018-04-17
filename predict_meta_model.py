from os import path

MIXED_MODEL_FILE_NAME = "mixed_model.csv"

class WeightedPredictedData:

    total_weight = None
    weighted_prediction = {}

    def __init__(self):
        self.total_weight = 0

    def add_prediction(self, predictions):
        predictions.add_to_total(self)

    def get_normalised_prediction(self):
        normalised_prediction = {}
        for index, value in self.weighted_prediction.items():
            normalised_prediction[index] = value / self.total_weight

        return normalised_prediction

class PredictedDataFromModel:

    prediction = {}
    weigth = None

    def __init__(self, prediction_location, weigth):
        self.prediction = self._load_prediction_from_file(prediction_location)
        self.weigth = weigth

    def add_to_total(self, total_prediction):
        total_prediction.total_weight += self.weigth
        for index, value in self.prediction.items():
            old_predicted_value = total_prediction.weighted_prediction.get(index, 0)
            new_predicted_value = old_predicted_value + value*self.weigth
            total_prediction.weighted_prediction[index] = new_predicted_value

    def _load_prediction_from_file(self, file_path):
        prediction = {}
        with open(file_path, 'r') as input_file:
            for line_nunmber, line in enumerate(input_file):
                if line_nunmber > 0:
                    index, value = line.strip().split(",")
                    prediction[index] = float(value)

        return prediction

DATA_PATH = path.join("resultats", "test_outputs")

weighted_predicted_data = WeightedPredictedData()
data_to_add = [
    path.join(DATA_PATH, "gradient_boosting_test_output.csv"),
    path.join(DATA_PATH, "linear_model_test_output.csv"),
    path.join(DATA_PATH, "random_forest_test_output.csv"),
    path.join(DATA_PATH, "svm_test_output.csv")
]
for current_prediction_to_add in [PredictedDataFromModel(x, 0.25) for x in data_to_add]:
    weighted_predicted_data.add_prediction(current_prediction_to_add)

with open(path.join(DATA_PATH, MIXED_MODEL_FILE_NAME), 'w') as mixed_model_output:
    mixed_model_output.write("{},{}\n".format("Id", "SalePrice"))
    for index, value in weighted_predicted_data.get_normalised_prediction().items():
        mixed_model_output.write("{},{}\n".format(index, value))

from regressions_methods import *

TRAINED_MODEL_PATH_CONFIG_KEY = "trained_model_path"
OUTPUT_PATH_CONFIG_KEY = "output_path"
MODEL_TO_RUN_CONFIG_KEY = "model_to_run"
MODEL_TO_TRAIN_CONFIG_KEY = "model_to_train"

class ConfigClassifier:
    job_to_do = None
    model_to_execute = None

    model_mapping = {
        "svm": SVM,
        "linear_model": RegressionLineaireSimple
    }

    def __init__(self, config_dictionnary):
        self.model_to_execute = []
        self.__load_config_file(config_dictionnary)

    def __load_config_file(self, config_dictionnary):
        self.__validate_config_dictionnary(config_dictionnary)
        self.model_to_train = config_dictionnary.get(MODEL_TO_TRAIN_CONFIG_KEY, [])
        self.model_to_run = config_dictionnary.get(MODEL_TO_RUN_CONFIG_KEY, [])
        self.trained_model_pickel_path = config_dictionnary.get(TRAINED_MODEL_PATH_CONFIG_KEY, "trained_models")
        self.output_path = config_dictionnary.get(OUTPUT_PATH_CONFIG_KEY, "output")

    def get_jobs(self):
        for model in self.__get_distinct_model_to_train_or_run(self.model_to_train, self.model_to_run):
            model_to_execute = self.model_mapping.get(model, None)(model, self.trained_model_pickel_path, self.output_path)
            model_to_execute.train_model = (model in self.model_to_train)
            model_to_execute.run_model = (model in self.model_to_run)

            yield model_to_execute

    def __validate_config_dictionnary(self, config_dictionnary):
        if MODEL_TO_TRAIN_CONFIG_KEY not in config_dictionnary:
            print("No Training to do - Is it normal ?")

        if MODEL_TO_RUN_CONFIG_KEY not in config_dictionnary:
            print("No Model to run - Is it normal ?")

    def __get_distinct_model_to_train_or_run(self, model_to_train, model_to_run):
        return (x for x in set(model_to_train + model_to_run))



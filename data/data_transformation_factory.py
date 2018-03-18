from data.pca_transformation import PrincipalComponentAnalysisTransformation
from sklearn.externals import joblib
from os import path

DATA_PATH = ""
DATA_TRANSFORMATION_CONFIG_KEY = "data_transformation"
DATA_TRANSFORMATION_NAME_CONFIG_KEY = "name"
DATA_TRANSFORMATION_PARAMETERS_CONFIG_KEY = "parameters"
TRAINED_TRANSFORMATION_PATH_CONFIG_KEY = "trained_transformation_path"
TRAINED_TRANSFORMATIONS_CONFIG_KEY = "train_transformation"
class DataTransformationFactory:

    transformations = None
    transformation_directory = {
        "pca" : PrincipalComponentAnalysisTransformation
    }

    def __init__(self, configuration_dictionnnary):
        self.__read_configuration_dictionnrary(configuration_dictionnnary)

    def __read_configuration_dictionnrary(self, configuration_dictionnary):
        self.transformations = []
        self.trained_transformation_path = configuration_dictionnary.get(TRAINED_TRANSFORMATION_PATH_CONFIG_KEY, "")

        self.__add_transformation_to_train(configuration_dictionnary)
        self.__load_trained_transformation(configuration_dictionnary)

    def __add_transformation_to_train(self, configuration_dictionnary):
        transformations_configuration = configuration_dictionnary.get(DATA_TRANSFORMATION_CONFIG_KEY, [])

        for transformation_configuration in transformations_configuration:
            current_transformation_name = transformation_configuration[DATA_TRANSFORMATION_NAME_CONFIG_KEY]
            current_transformation_parameters = transformation_configuration[DATA_TRANSFORMATION_PARAMETERS_CONFIG_KEY]
            current_transformation = self.transformation_directory[current_transformation_name](
                current_transformation_name, current_transformation_parameters)
            self.transformations.append(current_transformation)

    def __load_trained_transformation(self, configuration_dictionnary):
        for transformation in configuration_dictionnary.get(TRAINED_TRANSFORMATIONS_CONFIG_KEY, []):
            transformation = joblib.load(path.join(self.trained_transformation_path, transformation))
            self.transformations.append(transformation)

    def fit(self, x):
        new_x = x
        for transformation in self.transformations:
            new_x = transformation.fit(new_x)

        self.dump()
        return new_x

    def transform(self, x):
        transformed_x = x
        for transformation in self.transformations:
            transformed_x = transformation.transform(transformed_x)

        return transformed_x

    def fit_transform(self, x):
        transformed_x = x
        for transformation in self.transformations:
            transformed_x = transformation.fit_transform(transformed_x)
        self.dump()

        return transformed_x

    def dump(self):

        for transformation in self.transformations:
            joblib.dump(transformation, path.join(self.trained_transformation_path, "transformation_{}.pkl".format(transformation.name)))
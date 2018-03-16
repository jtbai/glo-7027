from data.pca_transformation import PrincipalComponentAnalysisTransformation
from sklearn.externals import joblib
from os import path

DATA_PATH = "data"
DATA_TRANSFORMATION_CONFIG_KEY = "data_transformation"
DATA_TRANSFORMATION_NAME_CONFIG_KEY = "name"
DATA_TRANSFORMATION_PARAMETERS_CONFIG_KEY = "parameters"

class DataTransformationFactory:

    transformations = None
    transformation_directory = {
        "pca" : PrincipalComponentAnalysisTransformation
    }

    def __init__(self, configuration_dictionnnary):
        self.__read_configuration_dictionnrary(configuration_dictionnnary)

    def __read_configuration_dictionnrary(self, configuration_dictionnnary):
        transformations_configuration = configuration_dictionnnary.get(DATA_TRANSFORMATION_CONFIG_KEY, [])
        self.transformations = []

        for transformation_configuration in transformations_configuration:
            current_transformation_name = transformation_configuration[DATA_TRANSFORMATION_NAME_CONFIG_KEY]
            current_transformation_parameters = transformation_configuration[DATA_TRANSFORMATION_PARAMETERS_CONFIG_KEY]
            current_transformation = self.transformation_directory[current_transformation_name](current_transformation_name, current_transformation_parameters)
            self.transformations.append(current_transformation)

    def fit(self, x):
        new_x = x
        for transformation in self.transformations:
            new_x = transformation.fit(new_x)

        return new_x

    def transform(self, x):
        transformed_x = x
        for transformation in self.transformations:
            transformed_x = transformation.tranform(transformed_x)

    def fit_transform(self, x):
        transformed_x = x
        for transformation in self.transformations:
            transformed_x = transformation.fit_transform(transformed_x)

        return transformed_x

    def dump(self):

        for transformation in self.transformations:
            joblib.dump(transformation, path.join(DATA_PATH, "transformation_{}.pkl".format(transformation.name)))
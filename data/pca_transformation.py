from sklearn import decomposition

NUMBER_OF_COMPONENT_CONFIG_KEY = "number_of_component"

class PrincipalComponentAnalysisTransformation:

    number_of_component = None
    principal_component_analysis = None
    name = None

    def __init__(self, name, configuration_dictionary):
        self.name = name
        self.__read_configuration_dictionary(configuration_dictionary)

    def __read_configuration_dictionary(self, configuration_dictionary):
        self.__validate_configuration_dictionnary(configuration_dictionary)
        self.number_of_component = configuration_dictionary.get(NUMBER_OF_COMPONENT_CONFIG_KEY, None)

    def __validate_configuration_dictionnary(self, configuration_dictionnnary):
        if NUMBER_OF_COMPONENT_CONFIG_KEY not in configuration_dictionnnary:
            raise KeyError("Require number of component for PCA")

    def fit(self, x):
        self.principal_component_analysis = decomposition.PCA(n_components=self.number_of_component).fit(x)
        return self.principal_component_analysis

    def transform(self, x):
        return self.principal_component_analysis.transform(x)

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)
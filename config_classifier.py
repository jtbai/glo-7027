#definit si on utilise le classificateur

class config_classifier:
    def __init__(self):
        self.use_svm = True
        self.use_random_forest = True
        self.use_linear_model = True
        self.use_generalised_linear_model = True
        self.use_generalised_additive_model = True
        self.use_gradient_boosting = True

        # definit si on reentraine ou si on recharge le fichier .pkl pour les algorithmes

        self.retrain_svm = True
        self.retrain_random_forest = True
        self.retrain_linear_model = True
        self.retrain_generalised_linear_model = True
        self.retrain_generalised_additive_model = True
        self.retrain_gradient_boosting = True

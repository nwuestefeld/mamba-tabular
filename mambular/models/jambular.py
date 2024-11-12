from .sklearn_base_regressor import SklearnBaseRegressor
from .sklearn_base_lss import SklearnBaseLSS
from .sklearn_base_classifier import SklearnBaseClassifier
from ..base_models.mambular import Jambular
from ..configs.jambular_config import DefaultJambularConfig

class MambularRegressor(SklearnBaseRegressor):
        def __init__(self, **kwargs):
            super().__init__(model=Jambular, config=DefaultJambularConfig, **kwargs)
from dataclasses import dataclass
try:
    from .PonyModelBase import PonyModelBase, PonyModelBaseConfig
except:
    pass
from ..utils import misc
from ..predictors import BasePredictorBase, PosePredictorBase, BasePredictorConfig, PosePredictorBaseConfig


@dataclass
class PonyPretrainModelConfig(PonyModelBaseConfig):
    cfg_predictor_base: BasePredictorConfig = None
    cfg_predictor_instance: PosePredictorBaseConfig = None


class PonyPretrainModel(PonyModelBase):
    def __init__(self, cfg: PonyPretrainModelConfig):
        super().__init__(cfg)
        misc.load_cfg(self, cfg, PonyPretrainModelConfig)
        self.netBase = BasePredictorBase(self.cfg_predictor_base)
        self.netInstance = PosePredictorBase(self.cfg_predictor_instance)

try:
    from .Trainer import Trainer
    from .models import PonyPretrainModel
    from .models import PonyKinematicsModel
except:
    pass

def build_model(cfg):
    # cfg.name is a config-schema value (matches the saved checkpoint's
    # config.yaml) and is intentionally left as the original "MagicPony"/
    # "Ponymation" strings - only the Python class names were renamed.
    if cfg.name == "MagicPony":
        return PonyPretrainModel(cfg)
    elif cfg.name == "Ponymation":
        return PonyKinematicsModel(cfg)
    else:
        raise NotImplementedError(f"Unrecognized name in model cfg: {cfg.name}")

import hydra
from omegaconf import DictConfig
from model import Trainer, build_model
from model.paths import DESSIE_CKPT
import os
os.environ["WANDB__SERVICE_WAIT"] = "300"


# tensorboard --logdir ./tensorboard_logs/ --bind_all --samples_per_plugin images=1000000000 --port=23425 --reload_multifile True --load_fast false

@hydra.main(config_path="model/config", config_name="train_magicpony_horse")
def main(cfg: DictConfig):
    model = build_model(cfg.model)
    trainer = Trainer(cfg, model)
    # Dessie weight initialization. Path is resolved from model.paths (env
    # DESSIE_CKPT or <repo>/checkpoints/dessie/best.ckpt). Skip if unavailable
    # (e.g. pure eval from a resumed checkpoint).
    dessie_ckpt = cfg.get("dessie_ckpt", DESSIE_CKPT)
    if dessie_ckpt and os.path.exists(dessie_ckpt):
        model.load_dessie(dessie_ckpt)
    else:
        print(f"[run] Dessie checkpoint not found at {dessie_ckpt}; skipping load_dessie().")

    # The frozen teacher always loads Dessie's weights (DESSIE_CKPT, or +teacher_dessie_ckpt=...).
    frozen_teacher = getattr(model, "frozen_teacher", None)
    if frozen_teacher is not None:
        teacher_dessie_ckpt = cfg.get("teacher_dessie_ckpt", DESSIE_CKPT)
        if teacher_dessie_ckpt and os.path.exists(teacher_dessie_ckpt):
            frozen_teacher.load_dessie(teacher_dessie_ckpt)
        else:
            raise FileNotFoundError(
                f"model.use_frozen_teacher=true but teacher checkpoint not found at "
                f"{teacher_dessie_ckpt} -- an untrained-random teacher would supervise "
                f"nothing useful, so this is a hard error rather than a silent skip."
            )
    if cfg.run_train:
        trainer.train()
    if cfg.run_test:
        trainer.test()


if __name__ == "__main__":
    main()

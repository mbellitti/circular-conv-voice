from omegaconf import OmegaConf
from src.data_module import VoiceDataModule
from src.model import VoiceModel
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import MLFlowLogger


if __name__ == "__main__":

    config = OmegaConf.load("config.yaml")

    dm = VoiceDataModule(config)

    model = VoiceModel(config)

    logger = MLFlowLogger(experiment_name="circ-conv", save_dir="./logs")

    trainer = Trainer(
        logger=logger,
        accelerator=config.trainer.accelerator,
        max_epochs=config.trainer.max_epochs,
    )

    trainer.fit(model, datamodule=dm)
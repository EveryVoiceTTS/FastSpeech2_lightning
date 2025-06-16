from pathlib import Path

from everyvoice.config.shared_types import ContactInformation
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.config import HiFiGANConfig
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.utils import HiFiGAN
from everyvoice.tests.stubs import silence_c_stderr
from pytorch_lightning import Trainer


def get_dummy_vocoder(tmp_dir: Path) -> tuple[HiFiGAN, Path]:
    contact_info = ContactInformation(
        contact_name="Test Runner", contact_email="info@everyvoice.ca"
    )
    vocoder = HiFiGAN(HiFiGANConfig(contact=contact_info))
    with silence_c_stderr():
        trainer = Trainer(default_root_dir=str(tmp_dir), barebones=True)
    trainer.strategy.connect(vocoder)
    vocoder_path = tmp_dir / "vocoder"
    trainer.save_checkpoint(vocoder_path)
    return vocoder, vocoder_path

from dataclasses import dataclass
from mesh_predict.pmp.utils.base import BaseModule
from mesh_predict.pmp.utils.typing import *
from transformers import AutoProcessor, AutoModel


class SigLIPEncoder(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        checkpoint: str = "google/siglip-base-patch16-224"

    cfg: Config

    def configure(self) -> None:
        super().configure()

        self.model = AutoModel.from_pretrained(self.cfg.checkpoint).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.cfg.checkpoint)

    def forward(self, image=None, text=None):
        assert (
            image is not None or text is not None
        ), "Either image or text must be provided"

        if image is not None:

            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            features = self.model.get_image_features(**inputs)

        elif text is not None:
            inputs = self.processor(
                text=text, return_tensors="pt", padding="max_length", truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            features = self.model.get_text_features(**inputs)

        return features

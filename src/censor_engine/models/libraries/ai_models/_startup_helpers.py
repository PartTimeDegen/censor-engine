from dataclasses import dataclass

import torch

from ._enums import ModelType


@dataclass(slots=True)
class GPUHandler:
    model_caching_type: ModelType

    def get_GPU(self) -> int | str:
        match self.model_caching_type:
            case ModelType.TORCH:
                return 0 if torch.cuda.is_available() else "cpu"
            case _:
                return "cpu"

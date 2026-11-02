from dataclasses import dataclass

import torch

from ._enums import ModelType


class CachingStrategy:
    @staticmethod
    def torch_caching(count: int, cache_limit: int) -> int:
        if count % cache_limit == 0:
            torch.cuda.empty_cache()
            return count
        return count + 1


@dataclass(slots=True)
class CachingHandler:
    model_caching_type: ModelType

    count: int = 0
    cache_limit: int = 1_000

    def handle_cache(self):
        match self.model_caching_type:
            case ModelType.TORCH:
                self.count = CachingStrategy.torch_caching(
                    self.count, self.cache_limit
                )
            case _:
                pass

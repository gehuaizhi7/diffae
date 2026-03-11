from typing import Union
from .unet import BeatGANsUNetModel, BeatGANsUNetConfig
from .unet_autoenc import BeatGANsAutoencConfig, BeatGANsAutoencModel
from .dictionary import SparseDictionary
from .ista import ista, ISTAResult
from .encoder import EncoderAdapter

Model = Union[BeatGANsUNetModel, BeatGANsAutoencModel]
ModelConfig = Union[BeatGANsUNetConfig, BeatGANsAutoencConfig]

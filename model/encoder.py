from torch import Tensor, nn


class EncoderAdapter(nn.Module):
    """
    Minimal adapter that extracts z_e from an existing diffusion-autoencoder model.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        if hasattr(self.model, 'encoder'):
            return self.model.encoder(x)
        if hasattr(self.model, 'encode'):
            encoded = self.model.encode(x)
            if isinstance(encoded, dict) and 'cond' in encoded:
                return encoded['cond']
            if isinstance(encoded, Tensor):
                return encoded
        raise RuntimeError(
            'EncoderAdapter expected model.encoder or model.encode to return z_e.'
        )


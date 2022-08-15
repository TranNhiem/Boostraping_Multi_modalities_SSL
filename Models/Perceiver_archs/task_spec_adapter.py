from typing import Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn
from einops import rearrange, repeat  
import math

from Models.Perceiver_archs.adapter_prototype import (
    InputAdapter, OutputAdapter
)

class Image_LearnableEnc(InputAdapter):
    def __init__(self, image_shape: Tuple[int, ...]):
        *self.spatial_shape, num_image_channels = image_shape
        self.image_shape = image_shape

        super().__init__(num_input_channels=num_image_channels + self._num_position_encoding_channels())
        # create encodings for single example
        self.pos_encoding = nn.Linear((max_seq_len, num_input_channels))

    def forward(self, x):
        b, *d = x.shape
        if tuple(d) != self.image_shape:
            raise ValueError(f"Input image shape {tuple(d)} different from required shape {self.image_shape}")

        x_enc = self.pos_encoding(x)
        return torch.cat([x, x_enc], dim=-1)


class Image_FourierEnc(InputAdapter):
    def __init__(self, image_shape: Tuple[int, ...], num_frequency_bands: int):
        *self.spatial_shape, num_image_channels = image_shape
        self.image_shape = image_shape
        self.num_frequency_bands = num_frequency_bands

        super().__init__(num_input_channels=num_image_channels + self._num_position_encoding_channels())

        # create encodings for single example
        pos = self._positions()
        enc = self._position_encodings(pos)

        # flatten encodings along spatial dimensions
        enc = rearrange(enc, "... c -> (...) c")

        # position encoding prototype
        self.register_buffer("position_encoding", enc)

    def _positions(self, v_min=-1.0, v_max=1.0):
        """Create evenly spaced position coordinates for self.spatial_shape with values in [v_min, v_max].
        :param v_min: minimum coordinate value per dimension.
        :param v_max: maximum coordinate value per dimension.
        :return: position coordinates tensor of shape (*shape, len(shape)).
        """
        coords = [torch.linspace(v_min, v_max, steps=s) for s in self.spatial_shape]
        return torch.stack(torch.meshgrid(*coords), dim=len(self.spatial_shape))

    def _position_encodings(
        self, p: Tensor, max_frequencies: Optional[Tuple[int, ...]] = None, include_positions: bool = True
    ) -> Tensor:
        """Fourier-encode positions p using self.num_bands frequency bands.
        :param p: positions of shape (*d, c) where c = len(d).
        :param max_frequencies: maximum frequency for each dimension (1-tuple for sequences,
               2-tuple for images, ...). If `None` values are derived from shape of p.
        :param include_positions: whether to include input positions p in returned encodings tensor.
        :returns: position encodings tensor of shape (*d, c * (2 * num_bands + include_positions)).
        """
        encodings = []

        if max_frequencies is None:
            max_frequencies = p.shape[:-1]

        frequencies = [
            torch.linspace(1.0, max_freq / 2.0, self.num_frequency_bands, device=p.device)
            for max_freq in max_frequencies
        ]
        frequency_grids = []

        for i, frequencies_i in enumerate(frequencies):
            frequency_grids.append(p[..., i : i + 1] * frequencies_i[None, ...])

        if include_positions:
            encodings.append(p)

        encodings.extend([torch.sin(math.pi * frequency_grid) for frequency_grid in frequency_grids])
        encodings.extend([torch.cos(math.pi * frequency_grid) for frequency_grid in frequency_grids])

        return torch.cat(encodings, dim=-1)

    def _num_position_encoding_channels(self, include_positions: bool = True) -> int:
        return len(self.spatial_shape) * (2 * self.num_frequency_bands + include_positions)

    def forward(self, x):
        b, *d = x.shape

        if tuple(d) != self.image_shape:
            raise ValueError(f"Input image shape {tuple(d)} different from required shape {self.image_shape}")

        x_enc = repeat(self.position_encoding, "... -> b ...", b=b)
        x = rearrange(x, "b ... c -> b (...) c")   
        return torch.cat([x, x_enc], dim=-1)


class TextInputAdapter(InputAdapter):
    def __init__(self, vocab_size: int, max_seq_len: int, num_input_channels: int, init_scale: float = 0.02):
        super().__init__(num_input_channels=num_input_channels)

        self.text_embedding = nn.Embedding(vocab_size, num_input_channels)
        self.pos_encoding = nn.Parameter(torch.empty(max_seq_len, num_input_channels))

        self.scale = math.sqrt(num_input_channels)
        self._init_parameters(init_scale)

    def _init_parameters(self, init_scale: float):
        with torch.no_grad():
            self.pos_encoding.normal_(0.0, init_scale)

    def forward(self, x):
        b, l = x.shape  # noqa: E741
        p_enc = rearrange(self.pos_encoding[:l], "... -> () ...")
        return self.text_embedding(x) * self.scale + p_enc


class ClassificationOutputAdapter(OutputAdapter):
    def __init__(
        self,
        num_classes: int,
        num_output_queries: int = 1,
        num_output_query_channels: Optional[int] = None,
        init_scale: float = 0.02,
    ):
        if num_output_query_channels is None:
            num_output_query_channels = num_classes

        # feed empty Tensor to discard the inner nn.Parameters of query array.
        super().__init__(output_query=torch.empty(num_output_queries, num_output_query_channels), init_scale=init_scale)
        # directly map num_output_query_channels to target categories array.
        self.linear = nn.Linear(num_output_query_channels, num_classes)

    def forward(self, x):
        return self.linear(x).squeeze(dim=1)


class TextOutputAdapter(OutputAdapter):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        num_output_query_channels: int,
        init_scale: float = 0.02,
    ):
        super().__init__(output_query=torch.empty(max_seq_len, num_output_query_channels), init_scale=init_scale)
        self.linear = nn.Linear(num_output_query_channels, vocab_size)

    def forward(self, x):
        return self.linear(x).squeeze(dim=1)


class TiedTextOutputAdapter(OutputAdapter):
    def __init__(self, max_seq_len: int, embedding_weights: Tensor, init_scale: float = 0.02):
        vocab_size, num_input_channels = embedding_weights.shape
        super().__init__(output_query=torch.empty(max_seq_len, num_input_channels), init_scale=init_scale)
        self.proj = nn.Linear(num_input_channels, vocab_size)
        self.proj.weight = embedding_weights

    def forward(self, x):
        return self.proj(x)

if __name__ == "__main__":
    breakpoint()
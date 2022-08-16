import unittest
import pytest
import torch

from Models.Perceiver_archs.perceiver_module import (
    PerceiverEncoder, PerceiverDecoder, PerceiverIO
)
from Models.Perceiver_archs.io_adapter import (
    Image_FourierEnc, ClassificationOutputAdapter
)

class Test_ImageClassifier(object):

    def setup_method(self):
        # pre-def shape
        self.image_shape = (224, 224, 3)
        self.num_classes = 1000

        # Fourier-encodes pixel positions and flattens along spatial dimensions
        input_adapter = Image_FourierEnc(
            image_shape=self.image_shape,  # M = 224 * 224
            num_frequency_bands=64,
        )
        # Projects generic Perceiver decoder output to specified number of classes
        output_adapter = ClassificationOutputAdapter(
            num_classes=self.num_classes,  # E
            num_output_query_channels=1024  # F
        )

        # Generic Perceiver encoder
        encoder = PerceiverEncoder(
            input_adapter=input_adapter,
            num_latents=512,  # N
            num_latent_channels=1024,  # D
            num_cross_attention_qk_channels=input_adapter.num_input_channels,  # C
            num_cross_attention_heads=1,
            num_self_attention_heads=8,
            num_self_attention_layers_per_block=6,
            num_self_attention_blocks=8,
            dropout=0.0,
        )
        # Generic Perceiver decoder
        decoder = PerceiverDecoder(
            output_adapter=output_adapter,
            num_latent_channels=1024,  # D
            num_cross_attention_heads=1,
            dropout=0.0,
        )
        # Perceiver IO image classifier
        self.model = PerceiverIO(encoder, decoder)

    @pytest.mark.parametrize(
        "image_shape", 
        [
            ((1, 224, 224, 3)),     
            ((4, 224, 224, 3))           
        ]
    )
    def test_out_shape(self, image_shape):
        im_mtr = torch.rand(image_shape)

        out = self.model(im_mtr)
        # confirm last dim of output indicate the num_of_cls
        assert out.shape[-1] == self.num_classes
        #confirm the first dim indicate the corresponding input.
        assert out.shape[0] == image_shape[0]


if __name__ == "__main__":
    breakpoint()
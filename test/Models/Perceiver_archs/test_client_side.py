import unittest
import pytest
import torch

from Models.Perceiver_archs.perceiver_artifact import ImageClassifier

from Models.Perceiver_archs.artifact_config import (
    PerceiverConfig, EncoderConfig, DecoderConfig
)


class Test_default_cfg_ImageClassifier(object):

    def setup_method(self):
        # pre-def shape
        self.image_shape = (224, 224, 3)
        self.num_classes = 1000

        config = PerceiverConfig( EncoderConfig(), DecoderConfig() )
        self.im_clf = ImageClassifier(image_shape=self.image_shape, num_classes=self.num_classes, num_output_query_channels=1024, config=config)
        

    @pytest.mark.parametrize(
        "image_shape", 
        [
            ((1, 224, 224, 3)),     
            ((4, 224, 224, 3))           
        ]
    )
    def test_out_shape(self, image_shape):
        im_mtr = torch.rand(image_shape)

        out = self.im_clf(im_mtr)
        # confirm last dim of output indicate the num_of_cls
        assert out.shape[-1] == self.num_classes
        #confirm the first dim indicate the corresponding input (batch).
        assert out.shape[0] == image_shape[0]

if __name__ == "__main__":
    breakpoint()
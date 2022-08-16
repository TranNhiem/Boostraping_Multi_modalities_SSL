import torch
from Models.Perceiver_archs.perceiver_module import (
    PerceiverEncoder, PerceiverDecoder, PerceiverIO
)
from Models.Perceiver_archs.io_adapter import (
    Image_FourierEnc, ClassificationOutputAdapter
)


class ImageClassifier(PerceiverIO):
    # fast deploy wo much params setup, customization plz inherit config to override base setup!!
    def __init__(self, image_shape, num_classes, num_output_query_channels, config: PerceiverConfig[EncoderConfig, DecoderConfig]):
        input_adapter = Image_LearnableEnc(image_shape=image_shape)
        output_adapter = ClassificationOutputAdapter(
            num_classes=num_classes,  
            num_output_query_channels=num_output_query_channels
        )

        enc_cfg, dec_cfg = config.enc_cfg, config.dec_cfg
        # Generic Perceiver encoder
        encoder = PerceiverEncoder(
            input_adapter=input_adapter,
            num_latents=config.num_latents,  
            num_latent_channels=config.num_latent_channels,  
            num_cross_attention_qk_channels=input_adapter.num_input_channels,  
            **enc_cfg
        )
        # Generic Perceiver decoder
        decoder = PerceiverDecoder(
            output_adapter=output_adapter,
            num_latent_channels=config.num_latent_channels,
            **dec_cfg
        )

        # Perceiver IO image classifier
        self.model = PerceiverIO(encoder, decoder)


if __name__ == "__main__":
    from Models.Perceiver_archs.artifact_config import PerceiverConfig, EncoderConfig, DecoderConfig
    config = PerceiverConfig( EncoderConfig(), DecoderConfig() )

    im_clf = ImageClassifier(image_shape=(224, 224, 3), num_classes=1000, num_output_query_channels=1024, config)
    
    im = torch.rand((1, 224, 224, 3))
    print(f"output prediction : {im_clf(im).shape}")
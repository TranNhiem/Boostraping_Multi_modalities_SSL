import pytorch_lightning as pl

class ImageClassifier(PerceiverIO):

    def __init__(self, config: PerceiverConfig[ImageEncoderConfig, ClassificationDecoderConfig]):
        input_adapter = ImageInputAdapter(
            image_shape=config.encoder.image_shape, num_frequency_bands=config.encoder.num_frequency_bands
        )
        

    



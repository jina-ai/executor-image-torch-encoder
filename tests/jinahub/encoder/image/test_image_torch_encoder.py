__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from jinahub.encoder.image.image_torch_encoder import ImageTorchEncoder


def test_image_torch_encoder_init():
    encoder = ImageTorchEncoder()
    assert encoder is not None

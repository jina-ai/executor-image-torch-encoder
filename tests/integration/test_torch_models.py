__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Tuple, Dict

import pytest

import torch
import numpy as np
from jina import DocumentArray, Document

from jinahub.image.encoder import ImageTorchEncoder


MODELS_TO_TEST = [
    'mobilenet_v2',
    'resnet18',
    'squeezenet1_0',
    'inception_v3',
    'shufflenet_v2_x0_5'
]


@pytest.mark.parametrize(
    'model_name', MODELS_TO_TEST
)
def test_load_torch_models(model_name: str, test_images: Dict[str, np.array]):
    encoder = ImageTorchEncoder(channel_axis=3, model_name=model_name)

    docs = DocumentArray([Document(blob=img_arr) for img_arr in test_images.values()])
    encoder.encode(
        docs=docs,
        parameters={}
    )

    for doc in docs:
        assert doc.embedding is not None
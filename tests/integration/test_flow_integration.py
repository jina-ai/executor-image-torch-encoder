__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
import pytest
from urllib.parse import urlparse

from torchvision.models.mobilenet import load_state_dict_from_url, model_urls

from jina import Flow, Document, DocumentArray

from jinahub.encoder.image.image_torch_encoder import ImageTorchEncoder
import numpy as np


def test_clip_no_batch():
    def validate_callback(resp):
        assert 1 == len(DocumentArray(resp.data.docs).get_attributes('embedding'))

    f = Flow().add(uses=ImageTorchEncoder)
    with f:
        f.post(on='/test', inputs=[Document(blob=np.ones((3, 224, 224)))], on_done=validate_callback)

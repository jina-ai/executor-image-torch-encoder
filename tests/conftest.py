__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
from typing import Dict

import pytest
import torch
import numpy as np
from torchvision.models.mobilenet import model_urls
from PIL import Image

from jina import DocumentArray, Document


@pytest.fixture()
def test_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture()
def mobilenet_weights(tmpdir: str) -> str:
    weights_file = os.path.join(tmpdir, 'w.pth')
    torch.hub.download_url_to_file(url=model_urls['mobilenet_v2'], dst=weights_file)
    return weights_file


@pytest.fixture()
def docs_with_blobs() -> DocumentArray:
    return DocumentArray([
        Document(blob=np.ones((3, 10, 10), dtype=np.float32)) for _ in range(10)
    ])


@pytest.fixture()
def docs_with_chunk_blobs() -> DocumentArray:
    return DocumentArray([
        Document(
            chunks=[Document(blob=np.ones((3, 10, 10), dtype=np.float32))]) for _ in range(10)
    ])


@pytest.fixture()
def docs_with_chunk_chunk_blobs() -> DocumentArray:
    return DocumentArray([
        Document(
            chunks=[Document(
                chunks=[Document(blob=np.ones((3, 10, 10), dtype=np.float32)) for _ in range(10)])])
    ])


@pytest.fixture()
def test_images(test_dir: str) -> Dict[str, np.ndarray]:

    def get_path(file_name_no_suffix: str) -> str:
        return os.path.join(test_dir, 'data', file_name_no_suffix + '.png')

    mean = np.array([[[0.485, 0.456, 0.406]]], dtype=np.float32)
    std = np.array([[[0.229, 0.224, 0.225]]], dtype=np.float32)
    image_dict = {
        file_name: np.array(Image.open(get_path(file_name)), dtype=np.float32)[:, :, 0:3] / 255 for file_name in [
            'airplane', 'banana1', 'banana2', 'satellite', 'studio'
        ]
    }
    for name, img_arr in image_dict.items():
        image_dict[name] = (img_arr - mean) / std
    return image_dict

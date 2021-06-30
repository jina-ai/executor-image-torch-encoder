__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Tuple, Dict, List

import pytest

import torch
import numpy as np
from jina import DocumentArray, Document

from jinahub.image.encoder import ImageTorchEncoder


@pytest.mark.parametrize(
    ['content', 'out_shape'],
    [
        ([np.ones((10, 10, 3), dtype=np.uint8), (3, 224, 224)]),
        ([np.ones((360, 420, 3), dtype=np.uint8), (3, 224, 224)]),
        ([np.ones((300, 300, 3), dtype=np.uint8), (3, 224, 224)])
    ]
)
def test_preprocessing_reshape_correct(
        content: np.ndarray,
        out_shape: Tuple
):
    encoder = ImageTorchEncoder()

    reshaped_content = encoder._preprocess(content)

    assert reshaped_content.shape == out_shape, f'Expected shape {out_shape} but got {reshaped_content.shape}'


@pytest.mark.parametrize(
    ['feature_map', 'pooling_strategy', 'expected_output'],
    [
        (np.ones((1, 10, 3, 3)), 'mean', np.mean(np.ones((1, 10, 3, 3)), axis=(2, 3))),
        (np.ones((1, 10, 3, 3)), 'max', np.max(np.ones((1, 10, 3, 3)), axis=(2, 3)))
    ]
)
def test_get_pooling(
    feature_map: np.ndarray,
    pooling_strategy: str,
    expected_output: str
):
    encoder = ImageTorchEncoder(
        pool_strategy=pooling_strategy
    )

    feature_map_after_pooling = encoder._get_pooling(feature_map)

    np.testing.assert_array_equal(feature_map_after_pooling, expected_output)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason='requires GPU and CUDA')
def test_get_features_gpu():
    encoder = ImageTorchEncoder()
    arr_in = np.ones((2, 3, 10, 10), dtype=np.float32)

    encodings = encoder._get_features(torch.from_numpy(arr_in).to(encoder.device)).detach().cpu().numpy()

    assert encodings.shape == (2, 1280, 1, 1)


def test_get_features_cpu():
    encoder = ImageTorchEncoder(device='cpu')
    arr_in = np.ones((2, 3, 10, 10), dtype=np.float32)

    encodings = encoder._get_features(torch.from_numpy(arr_in)).detach().numpy()

    assert encodings.shape == (2, 1000)


@pytest.mark.parametrize(
    'traversal_path, docs',
    [
        (['r'], pytest.lazy_fixture('docs_with_blobs')),
        (['c'], pytest.lazy_fixture('docs_with_chunk_blobs'))
    ]
)
def test_encode_image_returns_correct_length(traversal_path: List[str], docs: DocumentArray) -> None:
    encoder = ImageTorchEncoder(default_traversal_path=traversal_path)

    encoder.encode(docs=docs, parameters={})

    for doc in docs.traverse_flat(traversal_path):
        assert doc.embedding is not None
        assert doc.embedding.shape == (1000, )


@pytest.mark.parametrize(
    'model_name',
    [
        'resnet18',
        'resnet50',
        'inception_v3'
    ]
)
def test_encodes_semantic_meaning(test_images: Dict[str, np.array], model_name: str):
    encoder = ImageTorchEncoder(model_name=model_name)
    embeddings = {}

    for name, image_arr in test_images.items():
        docs = DocumentArray([Document(blob=image_arr)])
        encoder.encode(docs, parameters={})
        embeddings[name] = docs[0].embedding

    def dist(a, b):
        a_embedding = embeddings[a]
        b_embedding = embeddings[b]
        return np.linalg.norm(a_embedding - b_embedding)

    small_distance = dist('banana1', 'banana2')
    assert small_distance < dist('banana1', 'airplane')
    assert small_distance < dist('banana1', 'satellite')
    assert small_distance < dist('banana1', 'studio')
    assert small_distance < dist('banana2', 'airplane')
    assert small_distance < dist('banana2', 'satellite')
    assert small_distance < dist('banana2', 'studio')
    assert small_distance < dist('airplane', 'studio')
    assert small_distance < dist('airplane', 'satellite')
    assert small_distance < dist('studio', 'satellite')


def test_no_preprocessing():
    encoder = ImageTorchEncoder(use_default_preprocessing=False)
    arr_in = np.ones((3, 224, 224), dtype=np.float32)
    docs = DocumentArray([Document(blob=arr_in)])

    encoder.encode(docs=docs, parameters={})

    assert docs[0].embedding.shape == (1000, )

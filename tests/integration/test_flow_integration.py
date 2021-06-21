__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import List

import pytest

from jinahub.image.encoder import ImageTorchEncoder

from jina import Flow, Document, DocumentArray

import numpy as np


@pytest.mark.parametrize('arr_in', [
    (np.ones((3, 224, 224), dtype=np.float32)),
    (np.ones((3, 100, 100), dtype=np.float32)),
    (np.ones((3, 50, 40), dtype=np.float32))
])
def test_no_batch(arr_in: np.ndarray):
    def validate_callback(resp):
        results_arr = DocumentArray(resp.data.docs)
        assert len(results_arr) == 1
        assert results_arr[0].embedding is not None
        assert results_arr[0].embedding.shape == (1280, )

    flow = Flow().add(uses=ImageTorchEncoder)
    with flow:
        flow.post(
            on='/test',
            inputs=[Document(blob=arr_in)],
            on_done=validate_callback
        )


def test_with_batch():
    def validate_callback(resp):
        assert 25 == len(resp.docs.get_attributes('embedding'))

    flow = Flow().add(uses=ImageTorchEncoder)

    with flow:
        flow.post(
            on='/test',
            inputs=(Document(blob=np.ones((3, 224, 224), dtype=np.float32)) for _ in range(25)),
            on_done=validate_callback
        )


@pytest.mark.parametrize(
    ['docs', 'docs_per_path', 'traversal_path'],
    [
        (pytest.lazy_fixture('docs_with_blobs'), [['r', 10], ['c', 0], ['cc', 0]], 'r'),
        (pytest.lazy_fixture('docs_with_chunk_blobs'), [['r', 0], ['c', 10], ['cc', 0]], 'c'),
        (pytest.lazy_fixture('docs_with_chunk_chunk_blobs'), [['r', 0], ['c', 0], ['cc', 10]], 'cc')
    ]
)
def test_traversal_path(docs: DocumentArray, docs_per_path: List[List[str]], traversal_path: str):
    def validate_traversal(expected_docs_per_path: List[List[str]]):
        def validate(resp):
            for path, count in expected_docs_per_path:
                assert len(DocumentArray(resp.data.docs).traverse_flat([path]).get_attributes('embedding')) == count
        return validate

    flow = Flow().add(uses=ImageTorchEncoder)
    with flow:
        flow.post(
            on='/test',
            inputs=docs,
            on_done=validate_traversal(docs_per_path),
            parameters={'traversal_path': [traversal_path]}
        )

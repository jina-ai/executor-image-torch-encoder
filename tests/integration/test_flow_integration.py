__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import subprocess
from typing import List

import numpy as np
import pytest
from executor.torch_encoder import ImageTorchEncoder
from jina import Document, DocumentArray, Flow


@pytest.mark.parametrize(
    'arr_in',
    [
        (np.ones((224, 224, 3), dtype=np.uint8)),
        (np.ones((100, 100, 3), dtype=np.uint8)),
        (np.ones((50, 40, 3), dtype=np.uint8)),
    ],
)
def test_no_batch(arr_in: np.ndarray):
    flow = Flow().add(uses=ImageTorchEncoder)
    with flow:
        results_arr = flow.post(
            on='/test', inputs=[Document(tensor=arr_in)], return_results=True
        )

    assert len(results_arr) == 1
    assert results_arr[0].embedding is not None
    assert results_arr[0].embedding.shape == (512,)


def test_with_batch():
    flow = Flow().add(uses=ImageTorchEncoder)

    with flow:
        da = flow.post(
            on='/test',
            inputs=(
                Document(tensor=np.ones((224, 224, 3), dtype=np.uint8)) for _ in range(25)
            ),
        )

    assert len(da.embeddings) == 25


@pytest.mark.parametrize(
    ['docs', 'docs_per_path', 'access_paths'],
    [
        (pytest.lazy_fixture('docs_with_tensors'), [['@r', 11], ['@c', 0], ['@cc', 0]], '@r'),
        (
            pytest.lazy_fixture('docs_with_chunk_tensors'),
            [['@r', 0], ['@c', 11], ['@cc', 0]],
            '@c',
        ),
        (
            pytest.lazy_fixture('docs_with_chunk_tensors'),
            [['@r', 0], ['@c', 0], ['@cc', 11]],
            '@cc',
        ),
    ],
)
def test_access_paths(
    docs: DocumentArray, docs_per_path: List[List[str]], access_paths: str
):
    def validate_traversal(expected_docs_per_path: List[List[str]]):
        def validate(docs):
            for path, count in expected_docs_per_path:
                embeddings = docs[path].embeddings
                if embeddings is not None:
                    return len([em for em in embeddings if em is not None]) == count
                else:
                    return count == 0
        return validate

    flow = Flow().add(uses=ImageTorchEncoder)

    with flow:
        docs = flow.post(
            on='/test',
            inputs=docs,
            parameters={'access_paths': access_paths},
        )

    assert validate_traversal(docs_per_path)(docs)


@pytest.mark.gpu
@pytest.mark.docker
def test_docker_runtime_gpu(build_docker_image_gpu: str):
    with pytest.raises(subprocess.TimeoutExpired):
        subprocess.run(
            [
                'jina',
                'executor',
                f'--uses=docker://{build_docker_image_gpu}',
                '--gpus',
                'all',
                '--uses-with',
                'device:cuda',
            ],
            timeout=30,
            check=True,
        )

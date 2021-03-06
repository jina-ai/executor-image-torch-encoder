__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from jina import DocumentArray, Executor, requests
from jina.logging.logger import JinaLogger

from .models import EmbeddingModelWrapper


class ImageTorchEncoder(Executor):
    """
    :class:`ImageTorchEncoder` encodes ``Document`` tensors of type `ndarray` (`float32`)
    and shape `H x W x C` into `ndarray` of shape `D`, Where `D` is the Dimension of the
    embedding.

    If `use_default_preprocessing=False`, the expected input shape is `C x H x W` with
    `float32` dtype.

    :class:`ImageTorchEncoder` fills the `embedding` fields of `Documents` with an
    `ndarray` of shape `embedding_dim` (size depends on the model) with `dtype=float32`.

    Internally, :class:`ImageTorchEncoder` wraps the models from
    `torchvision.models`.
    https://pytorch.org/vision/stable/models.html
    """

    def __init__(
        self,
        model_name: str = 'resnet18',
        device: str = 'cpu',
        traversal_paths: str = '@r',
        batch_size: Optional[int] = 32,
        use_default_preprocessing: bool = True,
        *args,
        **kwargs,
    ):
        """
        :param model_name: the name of the model. Some of the models:
            ``alexnet``, `squeezenet1_0``,  ``vgg16``,
            ``densenet161``, ``inception_v3``, ``googlenet``,
            ``shufflenet_v2_x1_0``, ``mobilenet_v2``,
            ``mnasnet1_0``, ``resnet18``. See full list above.
        :param device: Which device the model runs on. Can be 'cpu' or 'cuda'
        :param traversal_paths: Used in the encode method an defines traversal on the
            received `DocumentArray`
        :param batch_size: Defines the batch size for inference on the loaded PyTorch
            model.
        """
        super().__init__(*args, **kwargs)
        self.logger = JinaLogger(self.__class__.__name__)

        self.device = device
        self.batch_size = batch_size
        self.use_default_preprocessing = use_default_preprocessing

        self.traversal_paths = traversal_paths

        # axis 0 is the batch
        self._default_channel_axis = 1

        self.model_wrapper = EmbeddingModelWrapper(model_name, device=self.device)

        self._preprocess = T.Compose(
            [
                T.ToPILImage('RGB'),
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @requests
    def encode(self, docs: DocumentArray, parameters: Dict, **kwargs):
        """
        Encode image data into a ndarray of `D` as dimension, and fill the embedding
        of each Document.

        :param docs: DocumentArray containing images
        :param parameters: dictionary to define the `traversal_paths` and the
            `batch_size`. For example,
            `parameters={'traversal_paths': ['r'], 'batch_size': 10}`.
        :param kwargs: Additional key value arguments.
        """
        if docs:
            docs_batch_generator = DocumentArray(
                filter(
                    lambda x: x.tensor is not None,
                    docs[parameters.get('traversal_paths', self.traversal_paths)],
                )
            ).batch(batch_size=parameters.get('batch_size', self.batch_size))

            self._compute_embeddings(docs_batch_generator)

    def _compute_embeddings(self, docs_batch_generator: Iterable) -> None:
        with torch.inference_mode():
            for document_batch in docs_batch_generator:
                tensor_batch = [d.tensor for d in document_batch]
                if self.use_default_preprocessing:
                    images = np.stack(self._preprocess_image(tensor_batch))
                else:
                    images = np.stack(tensor_batch)
                features = self.model_wrapper.compute_embeddings(images)

                document_batch.embeddings = features


    def _preprocess_image(self, images: List[np.array]) -> List[np.ndarray]:
        return [self._preprocess(img) for img in images]

__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Optional, List, Dict, Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from jina import Executor, requests, DocumentArray


def _batch_generator(data: List[Any], batch_size: int):
    for i in range(0, len(data), batch_size):
        yield data[i: i + batch_size]


class ImageTorchEncoder(Executor):
    """
    :class:`ImageTorchEncoder` encodes ``Document`` content from a ndarray,
    potentially B x (Height x Width x Channel) into a ndarray of `B x D`.
    Where B` is the batch size and `D` is the Dimension.
    Internally, :class:`ImageTorchEncoder` wraps the models from `
    `torchvision.models`.
    https://pytorch.org/docs/stable/torchvision/models.html
    :param model_name: the name of the model. Supported models include
        ``resnet18``, ``alexnet``, `squeezenet1_0``,  ``vgg16``,
        ``densenet161``, ``inception_v3``, ``googlenet``,
        ``shufflenet_v2_x1_0``, ``mobilenet_v2``, ``resnext50_32x4d``,
        ``wide_resnet50_2``, ``mnasnet1_0``
    :param pool_strategy: the pooling strategy. Options are:
        - `None`: Means that the output of the model will be the 4D tensor
            output of the last convolutional block.
        - `mean`: Means that global average pooling will be applied to the
            output of the last convolutional block, and thus the output of
            the model will be a 2D tensor.
        - `max`: Means that global max pooling will be applied.
    :param channel_axis: The axis of the color channel, default is 1
    :param device: Which device the model runs on. Can be 'cpu' or 'cuda'
    :param load_pre_trained_from_path: Loads your own model weights form the path. If not provided, the default
           model will be downloaded from torch hub.
    :param default_traversal_path: Used in the encode method an define traversal on the received `DocumentArray`
    :param default_batch_size: Defines the batch size for inference on the loaded PyTorch model.
    :param args:  Additional positional arguments
    :param kwargs: Additional keyword arguments
    """

    def __init__(
        self,
        model_name: str = 'mobilenet_v2',
        pool_strategy: str = 'mean',
        channel_axis: int = 1,
        device: Optional[str] = None,
        load_pre_trained_from_path: Optional[str] = None,
        default_traversal_path: Optional[str] = None,
        default_batch_size: Optional[int] = 32,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        if not device:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.default_batch_size = default_batch_size

        if default_traversal_path is None:
            self.default_traversal_path = ['r']
        else:
            self.default_traversal_path = default_traversal_path

        self.channel_axis = channel_axis
        # axis 0 is the batch
        self._default_channel_axis = 1
        self.model_name = model_name
        if pool_strategy not in ('mean', 'max'):
            raise NotImplementedError(f'unknown pool_strategy: {self.pool_strategy}')
        self.pool_strategy = pool_strategy
        self.pool_fn = getattr(np, self.pool_strategy)

        if load_pre_trained_from_path:
            model = getattr(models, self.model_name)(pretrained=False)
            model.load_state_dict(torch.load(load_pre_trained_from_path))
        else:
            model = getattr(models, self.model_name)(pretrained=True)

        self.model = self._extract_feature_from_torch_module(model)
        self.model.to(torch.device(self.device))

    def _extract_feature_from_torch_module(self, model: nn.Module):
        # TODO: Find better way to extract the correct layer from the torch model.
        if hasattr(model, 'features'):
            return model.features.eval()
        elif hasattr(model, 'fc'):
            return model.eval()
        elif hasattr(model, 'layers'):
            return model.layers.eval()
        else:
            raise ValueError(f'Model {model.__class__.__name__} is currently not supported by the ImageTorchEncoder')

    def _get_features(self, content):
        return self.model(content)

    def _get_pooling(self, feature_map: 'np.ndarray') -> 'np.ndarray':
        if feature_map.ndim == 2 or self.pool_strategy is None:
            return feature_map
        return self.pool_fn(feature_map, axis=(2, 3))

    @requests
    def encode(self, docs: Optional[DocumentArray], parameters: Optional[Dict] = None, **kwargs):
        if docs:
            docs_batch_generator = self._get_docs_batch_generator(docs, parameters)
            self._compute_embeddings(docs_batch_generator)

    def _maybe_move_channel_axis(self, images: np.ndarray) -> 'np.ndarray':
        if self.channel_axis != self._default_channel_axis:
            images = np.moveaxis(images, self.channel_axis, self._default_channel_axis)
        return images

    def _get_docs_batch_generator(self, docs: DocumentArray, parameters: Dict):
        traversal_path = parameters.get('traversal_path', self.default_traversal_path)
        batch_size = parameters.get('batch_size', self.default_batch_size)

        flat_docs = docs.traverse_flat(traversal_path)

        filtered_docs = [doc for doc in flat_docs if doc is not None and doc.blob is not None]

        return _batch_generator(filtered_docs, batch_size)

    def _compute_embeddings(self, docs_batch_generator: Iterable) -> None:
        with torch.no_grad():
            for document_batch in docs_batch_generator:
                blob_batch = np.stack([d.blob for d in document_batch])
                images = self._maybe_move_channel_axis(blob_batch)
                tensor = torch.from_numpy(images)
                tensor = tensor.to(self.device)
                features = self._get_features(tensor).detach()
                features = self._get_pooling(features.cpu().numpy())

                for doc, embed in zip(document_batch, features):
                    doc.embedding = embed


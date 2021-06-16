__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
from typing import Optional, List

import numpy as np

from jina import Executor, requests, DocumentArray

import torch
import torchvision.models as models
from torch.hub import load_state_dict_from_url


class ImageTorchEncoder(Executor):
    """
    :class:`ImageTorchEncoder` encodes ``Document`` content from a ndarray,
    potentially B x (Channel x Height x Width) into a ndarray of `B x D`.
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
    :param args:  Additional positional arguments
    :param kwargs: Additional keyword arguments
    """
    DEFAULT_TRAVERSAL_PATH = ['r']

    def __init__(
        self,
        model_name: str = 'mobilenet_v2',
        pool_strategy: str = 'mean',
        channel_axis: int = 1,
        load_pre_trained_from_path: Optional[str] = None,
        default_traversal_path: Optional[List[str]] = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.default_traversal_path = self.DEFAULT_TRAVERSAL_PATH if default_traversal_path is None\
            else default_traversal_path

        self.channel_axis = channel_axis
        # axis 0 is the batch
        self._default_channel_axis = 1
        self.model_name = model_name
        if pool_strategy not in ('mean', 'max'):
            raise NotImplementedError(f'unknown pool_strategy: {self.pool_strategy}')
        self.pool_strategy = pool_strategy

        if load_pre_trained_from_path:
            # set env var as described in https://pytorch.org/vision/stable/models.html
            model = getattr(models, self.model_name)(pretrained=False)
            model.load_state_dict(torch.load(load_pre_trained_from_path))


        model = getattr(models, self.model_name)(pretrained=True)
        self.model = model.features.eval()
        self.model.to(torch.device('cpu'))
        if self.pool_strategy is not None:
            self.pool_fn = getattr(np, self.pool_strategy)

    def _get_features(self, content):
        return self.model(content)

    def _get_pooling(self, feature_map: 'np.ndarray') -> 'np.ndarray':
        if feature_map.ndim == 2 or self.pool_strategy is None:
            return feature_map
        return self.pool_fn(feature_map, axis=(2, 3))

    @requests
    # TODO: per request traversal path
    # TODO: add batching
    def encode(self, docs: DocumentArray, **kwargs):
        chunks = DocumentArray(
            docs.traverse_flat(self.default_traversal_path)
        )
        images = np.stack(chunks.get_attributes('blob'))
        images = self._maybe_move_channel_axis(images)

        _input = torch.from_numpy(images)
        features = self._get_features(_input).detach()
        features = self._get_pooling(features.numpy())

        for doc, embed in zip(chunks, features):
            doc.embedding = embed

        return chunks

    def _maybe_move_channel_axis(self, images) -> 'np.ndarray':
        if self.channel_axis != self._default_channel_axis:
            images = np.moveaxis(images, self.channel_axis, self._default_channel_axis)
        return images

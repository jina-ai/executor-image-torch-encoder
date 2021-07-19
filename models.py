""" Helper module to manage torch vision models """
from collections import namedtuple
from typing import Callable, Optional

import torch
import torchvision.models as models
import torch.nn as nn
import numpy as np

from torchvision.models.resnet import __all__ as all_resnet_models
from torchvision.models.alexnet import __all__ as all_alexnet_models
from torchvision.models.vgg import __all__ as all_vgg_models
from torchvision.models.squeezenet import __all__ as all_squeezenet_models
from torchvision.models.densenet import __all__ as all_densenet_models
from torchvision.models.mnasnet import __all__ as all_mnasnet_models
from torchvision.models.mobilenet import __all__ as all_mobilenet_models


class EmbeddingModelWrapper:

    def __init__(self, model_name: str, device: Optional[str] = None):
        if not device:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self._model_descriptor = _ModelCatalogue.get_descriptor(model_name)
        self._model = getattr(models, model_name)(pretrained=True)

        self.device = device

        self._pooling_layer = nn.AdaptiveAvgPool2d(1)
        self._pooling_layer.to(torch.device(self.device))

    def _pooling_function(self, tensor_in: 'torch.Tensor') -> 'torch.Tensor':
        return self._pooling_layer(tensor_in).squeeze(3).squeeze(2)

    def get_features(self, content: 'torch.Tensor') -> 'torch.Tensor':
        if self._model_descriptor.needs_forward_hook:
            feature_map = None

            def get_activations(model, model_input, output):
                nonlocal feature_map
                feature_map = output.detach()

            layer = getattr(self._model, self._model_descriptor.layer_name)
            handle = layer.register_forward_hook(get_activations)
            self._model(content)
            handle.remove()
            return feature_map
        else:
            forward_function = getattr(self._model, self._model_descriptor.layer_name)
            feature_map = forward_function(content)
            return feature_map

    def compute_embeddings(self, images: 'np.ndarray') -> 'np.ndarray':
        tensor = torch.from_numpy(images).to(self.device)
        features = self.get_features(tensor)
        features = self._pooling_function(features)
        features = features.detach().numpy()
        return features


class _ModelCatalogue:

    EmbeddingLayerDescriptor = namedtuple(
        'EmbeddingLayerDescriptor',
        'layer_name, needs_forward_hook'
    )

    # maps the tuple of available model names to the layer from which we want to
    # extract the embedding. Removes the first entry because it the model class
    # not the factory method.
    all_supported_models_to_layer_mapping = {
        tuple(all_resnet_models[1:]): EmbeddingLayerDescriptor('layer4', True),
        tuple(all_alexnet_models[1:]): EmbeddingLayerDescriptor('features', False),
        tuple(all_vgg_models[1:]): EmbeddingLayerDescriptor('features', False),
        tuple(all_squeezenet_models[1:]): EmbeddingLayerDescriptor('features', False),
        tuple(all_densenet_models[1:]): EmbeddingLayerDescriptor('features', False),
        tuple(all_mnasnet_models[1:]): EmbeddingLayerDescriptor('layers', False),
        tuple(all_mobilenet_models[1:]): EmbeddingLayerDescriptor('features', False),
    }

    @classmethod
    def is_model_supported(cls, model_name: str):
        return any([model_name in m for m in cls.all_supported_models_to_layer_mapping])

    @classmethod
    def get_descriptor(cls, model_name: str) -> EmbeddingLayerDescriptor:
        """
        Checks if model is supported and returns the lookup on the layer name.

        :param model_name: Name of the layer
        """
        if not cls.is_model_supported(model_name):
            raise ValueError(f'Model with name {model_name} is not supported. '
                             f'Supported models are: {cls.all_supported_models_to_layer_mapping.keys()}')

        for model_names, layer_descriptor in cls.all_supported_models_to_layer_mapping.items():
            if model_name in model_names:
                return layer_descriptor

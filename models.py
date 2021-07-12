""" Helper module to manage torch vision models """
from torchvision.models.resnet import __all__ as all_resnet_models
from torchvision.models.alexnet import __all__ as all_alexnet_models
from torchvision.models.vgg import __all__ as all_vgg_models
from torchvision.models.squeezenet import __all__ as all_squeezenet_models
from torchvision.models.inception import __all__ as all_inception_models
from torchvision.models.densenet import __all__ as all_densenet_models
from torchvision.models.googlenet import __all__ as all_googlenet_models
from torchvision.models.mnasnet import __all__ as all_mnasnet_models
from torchvision.models.mobilenet import __all__ as all_mobilenet_models
from torchvision.models.shufflenetv2 import __all__ as all_shufflenet_models


# maps the tuple of available model names to the layer from which we want to
# extract the embedding. Removes the first entry because it the model class
# not the factory method.
_all_supported_models_to_layer_mapping = {
    tuple(all_resnet_models[1:]): 'layer4',
    tuple(all_alexnet_models[1:]): 'features',
    tuple(all_vgg_models[1:]): 'features',
    tuple(all_squeezenet_models[1:]): 'features',
    tuple(all_inception_models[1:]): 'Mixed_7c',
    tuple(all_densenet_models[1:]): 'features',
    tuple(all_googlenet_models[1:]): 'inception5b',
    tuple(all_mnasnet_models[1:]): 'layers',
    tuple(all_mobilenet_models[1:]): 'features',
    tuple(all_shufflenet_models[1:]): 'conv5'
}


def is_model_supported(name: str) -> bool:
    return sum([name in model_names for model_names in _all_supported_models_to_layer_mapping]) > 0


def get_layer_attribute_for_model(model_name: str) -> str:
    """
    Checks if model is supported and returns the lookup on the layer name.

    :param model_name: Name of the layer
    """
    if not is_model_supported(model_name):
        raise ValueError(f'Model with name {model_name} is not supported. '
                         f'Supported models are: {_all_supported_models_to_layer_mapping.keys()}')

    for model_names, layer_name in _all_supported_models_to_layer_mapping.items():
        if model_name in model_names:
            return layer_name

""" Helper module to manage torch vision models """
from torchvision.models.alexnet import __all__ as all_alexnet_models
from torchvision.models.vgg import __all__ as all_vgg_models
from torchvision.models.squeezenet import __all__ as all_squeezenet_models
from torchvision.models.densenet import __all__ as all_densenet_models
from torchvision.models.mnasnet import __all__ as all_mnasnet_models
from torchvision.models.mobilenet import __all__ as all_mobilenet_models


# maps the tuple of available model names to the layer from which we want to
# extract the embedding. Removes the first entry because it the model class
# not the factory method.
_all_supported_models_to_layer_mapping = {
    tuple(all_alexnet_models[1:]): 'features',
    tuple(all_vgg_models[1:]): 'features',
    tuple(all_squeezenet_models[1:]): 'features',
    tuple(all_densenet_models[1:]): 'features',
    tuple(all_mnasnet_models[1:]): 'layers',
    tuple(all_mobilenet_models[1:]): 'features',
}


def is_model_supported(name: str) -> bool:
    return any([name in m for m in _all_supported_models_to_layer_mapping])


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

""" Unit tests for the models modules. """

import pytest

from jinahub.image.encoder.models import get_layer_attribute_for_model, is_model_supported


@pytest.mark.parametrize(
    ['model_name', 'is_supported'],
    [
        ('resnet18', False),
        ('alexnet', True),
        ('xResNet', False),
        ('Alexnet', False)
    ]
)
def test_is_model_supported(model_name: str, is_supported: bool):
    assert is_model_supported(model_name) == is_supported


@pytest.mark.parametrize(
    ['model_name', 'layer'],
    [
        ('alexnet', 'features'),
        ('vgg11', 'features'),
        ('squeezenet1_0', 'features'),
        ('densenet121', 'features'),
        ('mnasnet0_5', 'layers'),
        ('mobilenet_v2', 'features'),
    ]
)
def test_is_correct_layer(model_name: str, layer: str):
    assert get_layer_attribute_for_model(model_name) == layer
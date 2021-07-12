""" Unit tests for the models modules. """

import pytest

from jinahub.image.models import get_layer_attribute_for_model, is_model_supported


@pytest.mark.parametrize(
    ['model_name', 'is_supported'],
    [
        ('resnet18', True),
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
        ('ResNet', 'layer4'),
        ('AlexNet', 'features'),
        ('VGG', 'features'),
        ('SqueezeNet', 'features'),
        ('Inception3', 'Mixed_7c'),
        ('DenseNet', 'features'),
        ('googlenet', 'inception5b'),
        ('mnasnet0_5', 'layers'),
        ('mobilenet_v2', 'features'),
        ('ShuffleNetV2', 'conv5')
    ]
)
def test_is_correct_layer(model_name: str, layer: str):
    assert get_layer_attribute_for_model(model_name) == layer
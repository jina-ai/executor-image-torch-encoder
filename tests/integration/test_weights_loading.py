__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os

from torch import hub
from pytest_mock import MockerFixture

from jinahub.image.encoder import ImageTorchEncoder


def test_load_from_url(tmpdir: str, mocker: MockerFixture) -> None:
    os.environ['TORCH_HOME'] = str(tmpdir)
    spy = mocker.spy(hub, 'urlopen')

    _ = ImageTorchEncoder()

    assert os.path.isfile(os.path.join(tmpdir, 'hub', 'checkpoints', 'mobilenet_v2-b0353104.pth'))
    assert spy.call_count == 1


def test_load_weights_from_file(tmpdir: str, mobilenet_weights: str, mocker: MockerFixture) -> None:
    # reset cache dir for torch to avoid using cached files from other tests
    os.environ['TORCH_HOME'] = str(tmpdir)
    spy = mocker.spy(hub, 'urlopen')

    _ = ImageTorchEncoder(load_pre_trained_from_path=mobilenet_weights)

    assert spy.call_count == 0

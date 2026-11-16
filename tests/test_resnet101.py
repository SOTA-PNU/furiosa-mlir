import pytest
import torchvision

from cnn_reference import assert_declares_checkpoint, assert_fp32_matches, assert_reduced_precision_tracks
from harness import load_model


@pytest.fixture(scope="module")
def model():
    return load_model("models/resnet101.py")


@pytest.fixture(scope="module")
def reference():
    return torchvision.models.resnet101(weights="DEFAULT").cuda().eval()


def test_declares_the_torchvision_checkpoint(model, reference):
    assert_declares_checkpoint(model, reference)


def test_fp32_matches_torchvision(model, reference):
    assert_fp32_matches(model, reference)


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_reduced_precision_tracks_torchvision(model, reference, dtype):
    assert_reduced_precision_tracks(model, reference, dtype)

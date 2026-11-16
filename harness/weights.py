"""Real checkpoints for the model cases, keyed by graph buffer name. Deep models need them: seeded
random weights drive their activations to inf, which makes any comparison meaningless."""

import torch

TORCHVISION = {
    "alexnet": "AlexNet_Weights",
    "vgg16": "VGG16_Weights",
    "resnet101": "ResNet101_Weights",
    "mobilenetv3_large": "MobileNet_V3_Large_Weights",
}
HUGGINGFACE = {"qwen3_0_6b": "Qwen/Qwen3-0.6B"}


def torchvision(case: str) -> dict[str, torch.Tensor]:
    import torchvision.models as tvm

    weights = getattr(tvm, TORCHVISION[case]).DEFAULT
    builder = {"mobilenetv3_large": "mobilenet_v3_large"}.get(case, case)
    return getattr(tvm, builder)(weights=weights).state_dict()


def huggingface(case: str) -> dict[str, torch.Tensor]:
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    path = snapshot_download(HUGGINGFACE[case], allow_patterns=["*.safetensors", "*.json"])
    from pathlib import Path

    tensors = {}
    for file in sorted(Path(path).glob("*.safetensors")):
        tensors.update(load_file(file))
    return tensors


def load(loader: str, case: str) -> dict[str, torch.Tensor]:
    return {"torchvision": torchvision, "huggingface": huggingface}[loader](case)

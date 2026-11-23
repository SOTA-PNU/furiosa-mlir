"""Real checkpoints for the model cases, keyed by graph buffer name. Deep models need them: seeded
random weights drive their activations to inf, which makes any comparison meaningless."""

import torch

# Pinned versions, so regenerated golden data uses the weights the NPU tolerances were measured on.
TORCHVISION = {
    "alexnet": ("AlexNet_Weights", "IMAGENET1K_V1"),
    "vgg16": ("VGG16_Weights", "IMAGENET1K_V1"),
    "resnet101": ("ResNet101_Weights", "IMAGENET1K_V2"),
    "mobilenetv3_large": ("MobileNet_V3_Large_Weights", "IMAGENET1K_V2"),
    "vit_b16": ("ViT_B_16_Weights", "IMAGENET1K_V1"),
    "efficientnet_b0": ("EfficientNet_B0_Weights", "IMAGENET1K_V1"),
    "convnext_tiny": ("ConvNeXt_Tiny_Weights", "IMAGENET1K_V1"),
    "swin_t": ("Swin_T_Weights", "IMAGENET1K_V1"),
    "densenet121": ("DenseNet121_Weights", "IMAGENET1K_V1"),
}
HUGGINGFACE = {
    "qwen3_0_6b": ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"),
    "qwen3_1_7b": ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"),
    "qwen3_4b": ("Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c"),
    "qwen3_8b": ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"),
    # An ungated copy of meta-llama/Llama-3.1-8B-Instruct: its 291 tensors equal furiosa-ai/Llama-3.1-8B-Instruct's.
    "llama3_1_8b": ("unsloth/Llama-3.1-8B-Instruct", "4699cc75b550f9c6f3173fb80f4703b62d946aa5"),
    "bert_base": ("google-bert/bert-base-uncased", "86b5e0934494bd15c9632b12f734a8a67f723594"),
    "gpt2": ("openai-community/gpt2", "607a30d783dfa663caf39e06633721c8d4cfcd7e"),
    "whisper_tiny": ("openai/whisper-tiny", "169d4a4341b33bc18d8881c4b69c2e104e1cc0af"),
}


def version(loader: str, case: str) -> str:
    return "/".join({"torchvision": TORCHVISION, "huggingface": HUGGINGFACE}[loader][case])


def torchvision(case: str) -> dict[str, torch.Tensor]:
    import torchvision.models as tvm

    enum, name = TORCHVISION[case]
    weights = getattr(getattr(tvm, enum), name)
    builder = {"mobilenetv3_large": "mobilenet_v3_large", "vit_b16": "vit_b_16"}.get(case, case)
    return getattr(tvm, builder)(weights=weights).state_dict()


def huggingface(case: str) -> dict[str, torch.Tensor]:
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    repo, revision = HUGGINGFACE[case]
    path = snapshot_download(repo, revision=revision, allow_patterns=["*.safetensors", "*.json"])
    from pathlib import Path

    tensors = {}
    for file in sorted(Path(path).glob("*.safetensors")):
        tensors.update(load_file(file))
    return tensors


def load(loader: str, case: str) -> dict[str, torch.Tensor]:
    return {"torchvision": torchvision, "huggingface": huggingface}[loader](case)

import torch
from pathlib import Path
from torchvision import transforms
from mlops.data import AffectNetDataset


def test_data_loading():
    base_path = Path.home() / "dtu/data/raw/affectnet/YOLO_format"

    train_images = base_path / "train/images"
    train_labels = base_path / "train/labels"

    assert train_images.exists(), "Training images directory does not exist"
    assert train_labels.exists(), "Training labels directory does not exist"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = AffectNetDataset(
        images_dir=train_images,
        labels_dir=train_labels,
        transform=transform,
    )

    assert len(dataset) > 0, "Dataset is empty"

    img, label = dataset[0]

    assert isinstance(img, torch.Tensor), "Image is not a tensor"
    assert img.shape == (3, 224, 224), f"Unexpected image shape: {img.shape}"
    assert isinstance(label, int), "Label is not an integer"
    assert 0 <= label < 8, f"Label {label} out of range [0, 7]"

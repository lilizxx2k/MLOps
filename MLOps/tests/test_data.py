import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
from mlops.data import AffectNetDataset


def test_data_loading(tmp_path):
    # Create directory structure for mock data
    images_dir = tmp_path / "train/images"
    labels_dir = tmp_path / "train/labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    # Create a fake image
    img = Image.new("RGB", (224, 224), color=(255, 0, 0))
    img_path = images_dir / "sample.jpg"
    img.save(img_path)

    # Create a fake YOLO label: class_id x y w h
    label_path = labels_dir / "sample.txt"
    label_path.write_text("0 0.5 0.5 0.5 0.5\n")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = AffectNetDataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        transform=transform,
    )

    assert len(dataset) == 1, "Dataset should contain exactly one sample"

    img_tensor, label = dataset[0]

    assert isinstance(img_tensor, torch.Tensor), "Image is not a tensor"
    assert img_tensor.shape == (3, 224, 224), f"Unexpected image shape {img_tensor.shape}"
    assert label == 0, "Label value incorrect"

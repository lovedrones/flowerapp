import os
import json
import shutil
import ssl
import requests
import tarfile
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# Fix SSL certificate verification issue on macOS
ssl._create_default_https_context = ssl._create_unverified_context

# Flowers-102 dataset URLs
FLOWERS102_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
FLOWERS102_LABELS_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"
FLOWERS102_SPLITS_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat"

# Number of classes
NUM_CLASSES = 102

# Flower class names (102 classes)
FLOWER_NAMES = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea",
    "wild geranium", "tiger lily", "moon orchid", "bird of paradise", "monkshood",
    "globe thistle", "snapdragon", "colt's foot", "king protea", "spear thistle",
    "yellow iris", "globe-flower", "purple coneflower", "peruvian lily", "balloon flower",
    "giant white arum lily", "fire lily", "pincushion flower", "fritillary", "red ginger",
    "grape hyacinth", "corn poppy", "prince of wales feathers", "stemless gentian",
    "artichoke", "sweet william", "carnation", "garden phlox", "love in the mist",
    "cosmos", "alpine sea holly", "ruby-lipped cattleya", "cape flower", "great masterwort",
    "siam tulip", "lenten rose", "barberton daisy", "daffodil", "sword lily",
    "poinsettia", "bolero deep blue", "wallflower", "marigold", "buttercup",
    "daisy", "common dandelion", "petunia", "wild pansy", "primula",
    "sunflower", "lilac hibiscus", "bishop of llandaff", "gaura", "geranium",
    "orange dahlia", "pink-yellow dahlia", "cautleya spicata", "japanese anemone",
    "black-eyed susan", "silverbush", "californian poppy", "osteospermum", "spring crocus",
    "iris", "windflower", "tree poppy", "gazania", "azalea",
    "water lily", "rose", "thorn apple", "morning glory", "passion flower",
    "lotus", "toad lily", "anthurium", "frangipani", "clematis",
    "hibiscus", "columbine", "desert-rose", "tree mallow", "magnolia",
    "cyclamen", "watercress", "canna lily", "hippeastrum", "bee balm",
    "pink quill", "foxglove", "bougainvillea", "camellia", "mallow",
    "mexican petunia", "bromelia", "blanket flower", "trumpet creeper", "blackberry lily",
    "common tulip", "wild rose"
]


def download_file(url, destination):
    """Download a file from URL to destination."""
    if os.path.exists(destination):
        print(f"File already exists: {destination}")
        return
    
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f, tqdm(
        desc=os.path.basename(destination),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))
    
    print(f"Downloaded to {destination}")


def extract_tar(tar_path, extract_to):
    """Extract tar file to destination."""
    if os.path.exists(extract_to):
        print(f"Directory already exists: {extract_to}")
        return
    
    print(f"Extracting {tar_path}...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(extract_to)
    print(f"Extracted to {extract_to}")


def download_flowers102_dataset(data_dir="data"):
    """Download and prepare Flowers-102 dataset."""
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    
    # Download images
    images_tar = data_path / "102flowers.tgz"
    if not images_tar.exists():
        download_file(FLOWERS102_URL, images_tar)
    
    # Extract images
    images_dir = data_path / "jpg"
    if not images_dir.exists():
        extract_tar(images_tar, data_path)
    
    # Download labels and splits (using scipy to read .mat files)
    labels_file = data_path / "imagelabels.mat"
    splits_file = data_path / "setid.mat"
    
    if not labels_file.exists():
        download_file(FLOWERS102_LABELS_URL, labels_file)
    
    if not splits_file.exists():
        download_file(FLOWERS102_SPLITS_URL, splits_file)
    
    return images_dir, labels_file, splits_file


def load_flowers102_splits(splits_file, labels_file):
    """Load train/val/test splits and labels from .mat files."""
    from scipy.io import loadmat
    
    splits = loadmat(splits_file)
    labels = loadmat(labels_file)
    
    train_ids = splits['trnid'][0] - 1  # Convert to 0-indexed
    val_ids = splits['valid'][0] - 1
    test_ids = splits['tstid'][0] - 1
    
    all_labels = labels['labels'][0] - 1  # Convert to 0-indexed
    
    return train_ids, val_ids, test_ids, all_labels


class Flowers102Dataset(Dataset):
    """Flowers-102 Dataset class."""
    
    def __init__(self, images_dir, image_ids, labels, transform=None):
        self.images_dir = Path(images_dir)
        self.image_ids = image_ids
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = self.images_dir / f"image_{image_id+1:05d}.jpg"
        
        image = Image.open(image_path).convert('RGB')
        label = self.labels[image_id]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_data_loaders(data_dir="data", batch_size=32, num_workers=4):
    """Create data loaders for train, validation, and test sets."""
    # Download and prepare dataset
    images_dir, labels_file, splits_file = download_flowers102_dataset(data_dir)
    
    # Load splits and labels
    train_ids, val_ids, test_ids, all_labels = load_flowers102_splits(splits_file, labels_file)
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = Flowers102Dataset(images_dir, train_ids, all_labels, train_transform)
    val_dataset = Flowers102Dataset(images_dir, val_ids, all_labels, val_test_transform)
    test_dataset = Flowers102Dataset(images_dir, test_ids, all_labels, val_test_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader


def preprocess_image_for_inference(image, device='cpu'):
    """Preprocess a single image for model inference."""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    elif not isinstance(image, Image.Image):
        image = Image.fromarray(image).convert('RGB')
    
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor.to(device)


def load_model(model_path, num_classes=102, device='cpu'):
    """Load a trained model from file."""
    import torchvision.models as models
    
    # Create model architecture (ResNet50)
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model


def predict_image(model, image, device='cpu', top_k=5):
    """Predict flower class for an image."""
    model.eval()
    
    # Preprocess image
    image_tensor = preprocess_image_for_inference(image, device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probabilities, top_k)
    
    # Convert to lists
    top_probs = top_probs[0].cpu().numpy()
    top_indices = top_indices[0].cpu().numpy()
    
    # Get class names
    predictions = []
    for prob, idx in zip(top_probs, top_indices):
        predictions.append({
            'class': FLOWER_NAMES[idx],
            'confidence': float(prob)
        })
    
    return predictions


import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def preprocess_image(image):
    # Define a transform for Stable Diffusion
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # Apply the transform
    image = transform(image).unsqueeze(0)
    return image

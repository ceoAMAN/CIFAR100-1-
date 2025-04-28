import torch
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        return image_tensor
    except UnidentifiedImageError:
        logging.error(f"Error: The file '{image_path}' is not a valid image.")
        raise ValueError(f"Invalid image file: {image_path}")
    except Exception as e:
        logging.error(f"Unexpected error while loading image '{image_path}': {e}")
        raise

def load_model(model_class, model_path, num_classes=10, device='cpu'):
    if not os.path.exists(model_path):
        logging.error(f"Model file not found at '{model_path}'")
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = model_class(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    logging.info(f"Model loaded successfully from '{model_path}' on device '{device}'")
    return model

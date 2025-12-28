import os
import pickle
from PIL import Image
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
from torchvision.models import vgg19, VGG19_Weights

device = "cuda" if torch.cuda.is_available() else "cpu"

def extract_line_features(image_np, adaptive_thresh=True):
    """
    Extract edge density and normalized line thickness features using Sobel operator,
    including adaptive thresholding and normalization mapping.

    Args:
        image_np: np.ndarray, single-channel grayscale image, uint8 type with range 0-255 recommended
        adaptive_thresh: whether to use adaptive thresholding (default True)

    Returns:
        dict: {
            'edge_density': ratio of edge pixels (0~1),
            'line_thickness': normalized line thickness (relative scale, 0~1)
        }
    """

    h, w = image_np.shape[:2]

    # Calculate Sobel gradients
    sobel_x = cv2.Sobel(image_np.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image_np.astype(np.float32), cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    # Adaptive threshold calculation
    if adaptive_thresh:
        threshold_val, _ = cv2.threshold(edge_magnitude.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold = threshold_val if threshold_val > 0 else 50
    else:
        threshold = 50

    # Edge mask
    edge_mask = edge_magnitude > threshold
    edge_density = edge_mask.sum() / edge_mask.size  # edge ratio

    if edge_mask.sum() == 0:
        mean_edge_strength = 0.5  # default value to avoid zero
    else:
        mean_edge_strength = np.mean(edge_magnitude[edge_mask])

    # Normalize line thickness (relative to max image dimension)
    max_dim = max(h, w)
    normalized_thickness = mean_edge_strength / max_dim

    # Non-linear mapping to reasonable width range for drawing
    # The max normalized thickness can be small, so scale appropriately and clamp to [0.01, 0.1]
    def map_thickness(x):
        # Power function mapping, adjust shape as needed
        return np.clip(x ** 0.8 * 0.1, 0.01, 0.1)

    line_thickness_mapped = map_thickness(normalized_thickness)

    return {
        'edge_density': float(edge_density),
        'line_thickness': float(line_thickness_mapped)
    }

# Load pretrained VGG19 for style feature extraction
weights = VGG19_Weights.DEFAULT
vgg = vgg19(weights=weights).features.eval().to(device)

for param in vgg.parameters():
    param.requires_grad = False

def gram_matrix(feature_maps):
    b, c, h, w = feature_maps.size()
    features = feature_maps.view(b, c, h * w)
    G = torch.bmm(features, features.transpose(1, 2))
    return G / (c * h * w)

def extract_gram_style(image_pil):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    image_tensor = preprocess(image_pil).unsqueeze(0).to(device)  # (1,3,224,224)
    features = []
    x = image_tensor
    for i, layer in enumerate(vgg):
        x = layer(x)
        if i in [1, 6, 11, 20, 29]:  # Pick 5 layers as style layers
            features.append(gram_matrix(x))
    return features

def build_style_library(input_dir, output_path):
    style_library = {}
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        path = os.path.join(input_dir, filename)
        print(f"Processing style image {filename}")

        image_pil = Image.open(path).convert("RGB").resize((224, 224))
        image_gray = Image.open(path).convert("L").resize((512, 512))

        image_np = np.array(image_gray)
        line_features = extract_line_features(image_np)
        gram_style_features = extract_gram_style(image_pil)

        features = line_features
        features['gram_style'] = gram_style_features
        style_name = os.path.splitext(filename)[0]
        style_library[style_name] = features
        # Print extracted line width and edge density parameters
        print(f"Style '{style_name}': edge_density = {features['edge_density']:.4f}, line_thickness (normalized) = {features['line_thickness']:.4f}")

    with open(output_path, "wb") as f:
        pickle.dump(style_library, f)
    print(f"Style library saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="./input_style_images", help="Directory of style images")
    parser.add_argument("--output_path", default="./style_library.pkl", help="Output style library file")
    args = parser.parse_args()
    build_style_library(args.input_dir, args.output_path)
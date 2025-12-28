import torch
import pydiffvg
import clip
from diffusers import StableDiffusionPipeline
from rembg import remove
from PIL import Image, ImageEnhance
import numpy as np
import argparse
import os
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
import pickle

"""
SD-CLIPasso: Stable Diffusion Prior for Stroke-Controlled Sketch Generation
Based on CLIPasso (Vinker et al., SIGGRAPH 2022)
"""
import random
import math
import matplotlib.pyplot as plt
import imageio

def adjust_params_by_style(style_features, args):
    if style_features is None:
        print("No line features in style library, keeping default parameters")
        return args.num_strokes, args.initial_stroke_width
    # 直接读取顶层字段
    edge_density = style_features.get("edge_density", None)
    line_thickness = style_features.get("line_thickness", None)

    if edge_density is None or line_thickness is None:
        print("No line features in style library, keeping default parameters")
        return args.num_strokes, args.initial_stroke_width

    num_strokes_new = int(50 + (math.log(edge_density + 1e-5) - math.log(0.055)) / (math.log(0.34) - math.log(0.055)) * (400 - 50))
    stroke_width_new = max(0.1, min(2.0, 0.1 + (line_thickness - 0.03) / (0.1 - 0.03) * (2.0 - 0.1)))

    print(f"Automatically adjusting parameters based on style library: num_strokes {args.num_strokes} -> {num_strokes_new}, "
          f"initial_stroke_width {args.initial_stroke_width} -> {stroke_width_new}")

    return num_strokes_new, stroke_width_new

device = "cuda" if torch.cuda.is_available() else "cpu"

vgg = models.vgg19(pretrained=True).features.eval().to(device)
for param in vgg.parameters():
    param.requires_grad = False

def gram_matrix(feature_maps):
    b, c, h, w = feature_maps.size()
    features = feature_maps.view(b, c, h * w)
    G = torch.bmm(features, features.transpose(1, 2))
    return G / (c * h * w)

def extract_vgg_features(x):
    features = []
    for i, layer in enumerate(vgg):
        x = layer(x)
        if i in [1, 6, 11, 20, 29]: # Selected 5 layers as style layers
            features.append(x)
    return features

def compute_gram_style_loss(gen_img, style_gram_list):
    gen_features = extract_vgg_features(gen_img)
    loss = 0
    for gf, sg in zip(gen_features, style_gram_list):
        Gg = gram_matrix(gf)
        loss += F.mse_loss(Gg, sg.to(gen_img.device))
    return loss
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="a drawing of a cat")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--num_strokes", type=int, default=64, help="Number of strokes (control complexity)")
    parser.add_argument("--num_iter", type=int, default=500, help="Number of optimization iterations")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp_name", type=str, default=None, help="Custom name for the output folder")

    # New arguments for SD control
    parser.add_argument("--sd_style_suffix", type=str, default="realistic, full body, centered,  high contrast, simple background, contours only, no blur, beautiful", help="Style to append to SD prompt")
    parser.add_argument("--sd_negative_prompt", type=str, default="cropped, out of frame, cut off, close up, partial, shadow, contact shadow, ground, floor, wall, complex background, text, watermark, drawing, sketch, cartoon, anime, grayscale,, blur, pale, washed out, overexposed, low contrast", help="Negative prompt for SD")

    # New arguments for CLIPasso control
    parser.add_argument("--geometric_weight", type=float, default=1.0, help="Weight for geometric loss (structure)")
    parser.add_argument("--semantic_weight", type=float, default=0.1, help="Weight for semantic loss (content)")
    parser.add_argument("--initial_stroke_width", type=float, default=2.0, help="Initial width of the strokes")

    # Early Stopping
    parser.add_argument("--early_stop_patience", type=int, default=50, help="Stop if no improvement for N iterations")
    parser.add_argument("--early_stop_threshold", type=float, default=0.01, help="Minimum average pixel movement to consider as change")

    # Style guidance arguments
    parser.add_argument("--style_library", type=str, default="./style_extractor/style_library.pkl", help="Path to style library file (e.g., style_library.pkl)")
    parser.add_argument("--style_key", type=str, default="style2", help="Name of the style to use, corresponding to the key in style_library")
    parser.add_argument("--style_weight", type=float, default=15, help="Style guidance weight, controlling the strength of style loss")

    args = parser.parse_args()

    # Load style features if style library is specified
    style_features = None
    if args.style_library is not None and args.style_key is not None:
        with open(args.style_library, "rb") as f:
            style_library = pickle.load(f)
        print(f"Loaded style library containing {len(style_library)} styles.")
        if args.style_key not in style_library:
            raise KeyError(f"Style {args.style_key} not found in style library")
        style_features = style_library[args.style_key]
        print(f"Using style: {args.style_key}")
    
    args.num_strokes, args.initial_stroke_width = adjust_params_by_style(style_features, args)

    # Create output directory with safe folder name handling
    if args.exp_name:
        folder_name = args.exp_name
    else:
        safe_prompt = "".join([c for c in args.prompt if c.isalpha() or c.isdigit() or c == ' ']).rstrip()
        safe_prompt = safe_prompt.replace(" ", "_")
        style_str = args.sd_style_suffix.lstrip(", ")
        safe_style = "".join([c for c in style_str if c.isalpha() or c.isdigit() or c == ' ']).rstrip()
        safe_style = safe_style.replace(" ", "_")
        safe_style = safe_style[:40]
        folder_name = f"{safe_prompt}_s{args.num_strokes}_style_{safe_style}"

    output_dir = os.path.join(args.output_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    # Save configuration
    import json
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Generate initial image (Stable Diffusion)
    full_prompt =  args.prompt +", "+ args.sd_style_suffix
    print(f"Generating image for prompt: {full_prompt}")
    model_id = "runwayml/stable-diffusion-v1-5"

    try:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
        pipe = pipe.to(device)
        pipe.safety_checker = None

        image = pipe(full_prompt, negative_prompt=args.sd_negative_prompt).images[0]
        image.save(os.path.join(output_dir, "sd_raw.png"))

        # Background removal and preprocessing
        print("Removing background...")
        try:
            img_no_bg = remove(image)
            img_no_bg.save(os.path.join(output_dir, "sd_no_bg.png"))
            new_image = Image.new("RGB", img_no_bg.size, (255, 255, 255))
            new_image.paste(img_no_bg, (0, 0), img_no_bg)

            enhancer = ImageEnhance.Contrast(new_image)
            new_image = enhancer.enhance(1.2)

            import cv2
            img_cv = np.array(new_image)
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
            img_smooth = cv2.bilateralFilter(img_cv, d=9, sigmaColor=75, sigmaSpace=75)
            new_image = Image.fromarray(cv2.cvtColor(img_smooth, cv2.COLOR_BGR2RGB))

            image = new_image
            print("Background removed and image smoothed successfully.")
        except Exception as e:
            print(f"Background removal failed: {e}. Using original image.")

        image_path = os.path.join(output_dir, "sd_final.png")
        image.save(image_path)
        print(f"Saved processed SD image to {image_path}")

        # Clean up SD pipeline memory
        print("Cleaning up Stable Diffusion pipeline to save memory...")
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        print("Memory cleaned.")

    except Exception as e:
        print(f"Error loading Stable Diffusion: {e}")
        # Generate blank image to prevent code interruption
        image = Image.new('RGB', (512, 512), color='white')
        image.save(os.path.join(output_dir, "sd_output.png"))

    # 2. CLIPasso optimization preparation
    print("Starting CLIPasso optimization...")
    pydiffvg.set_use_gpu(False)
    diffvg_device = "cpu"

    clip_model, preprocess = clip.load("RN50", device=device, jit=False)

    # Feature extraction wrapper for geometric loss in CLIPasso
    class CLIPFeatureExtractor(torch.nn.Module):
        def __init__(self, model, target_layers=['layer3', 'layer4']):
            super().__init__()
            self.model = model
            self.target_layers = target_layers
            self.hooks = []
            self.intermediate_features = {}
            for layer_name in self.target_layers:
                layer = getattr(self.model.visual, layer_name)
                self.hooks.append(layer.register_forward_hook(self.make_hook(layer_name)))

        def make_hook(self, name):
            def hook_fn(module, input, output):
                self.intermediate_features[name] = output
            return hook_fn

        def encode_image(self, img):
            self.intermediate_features = {}
            emb = self.model.encode_image(img)
            return emb, self.intermediate_features

        def remove_hooks(self):
            for h in self.hooks:
                h.remove()

    clip_extractor = CLIPFeatureExtractor(clip_model, target_layers=['layer3', 'layer4'])

    target_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)
    canvas_width, canvas_height = image.size

    # Calculate edge direction probability map for stroke initialization
    import cv2
    img_cv = np.array(image)
    img_gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY).astype(np.float32)
    g1 = cv2.GaussianBlur(img_gray, (3, 3), 0)
    g2 = cv2.GaussianBlur(img_gray, (9, 9), 0)
    dog = g1 - g2

    prob_map = np.abs(dog)
    prob_map = prob_map / prob_map.max()
    prob_map[prob_map < 0.1] = 0
    prob_map = cv2.pow(prob_map, 2)
    prob_map += 1e-4
    prob_map /= prob_map.sum()

    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=5)
    grad_dir = np.arctan2(sobely, sobelx)
    edge_dir = grad_dir + np.pi / 2

    flat_indices = np.random.choice(prob_map.size, size=args.num_strokes, p=prob_map.flatten())
    y_coords, x_coords = np.unravel_index(flat_indices, prob_map.shape)

    shapes = []
    shape_groups = []

    # Initialize Bezier curve stroke paths
    for i in range(args.num_strokes):
        num_segments = 1
        num_control_points = torch.zeros(num_segments, dtype=torch.int32) + 2
        points = []
        p0 = (float(x_coords[i]), float(y_coords[i]))
        points.append(p0)

        ix = int(np.clip(x_coords[i], 0, canvas_width - 1))
        iy = int(np.clip(y_coords[i], 0, canvas_height - 1))
        angle = edge_dir[iy, ix]

        # 3 control points to form a curved segment
        for j in range(num_segments):
            radius = 5.0
            p1 = (p0[0] + radius * math.cos(angle), p0[1] + radius * math.sin(angle))
            p2 = (p1[0] + radius * math.cos(angle), p1[1] + radius * math.sin(angle))
            p3 = (p2[0] + radius * math.cos(angle), p2[1] + radius * math.sin(angle))
            points.append(p1)
            points.append(p2)
            points.append(p3)
            p0 = p3

        points = torch.tensor(points).to(diffvg_device)
        path = pydiffvg.Path(
            num_control_points=num_control_points,
            points=points,
            stroke_width=torch.tensor(args.initial_stroke_width).to(diffvg_device),
            is_closed=False)
        shapes.append(path)

        path_group = pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([len(shapes) - 1]),
            fill_color=None,
            stroke_color=torch.tensor([0., 0., 0., 1.]).to(diffvg_device))
        shape_groups.append(path_group)

    # Training parameters
    points_vars = []
    stroke_width_vars=[]
    for path in shapes:
        path.points.requires_grad = True
        points_vars.append(path.points)
        path.stroke_width.requires_grad = False

    optimizer = torch.optim.Adam(
    [{'params': points_vars, 'lr': 0.5}],betas=(0.9, 0.999))

    num_augs = 4
    loss_history = []
    frames = []
    patience_counter = 0
    prev_points = [p.detach().clone() for p in points_vars]

    # Style loss computation function
    def compute_style_loss(img_tensor, clip_model, style_clip_feat, device):
        # img_tensor: Tensor, 1x3xHxW, needs resizing to 224x224 for CLIP input
        if img_tensor.shape[2:] != (224, 224):
            img_tensor_resized = torch.nn.functional.interpolate(img_tensor, size=(224, 224), mode="bilinear")
        else:
            img_tensor_resized = img_tensor

        gen_feat = clip_model.encode_image(img_tensor_resized)
        gen_feat = gen_feat / gen_feat.norm(dim=-1, keepdim=True)
        style_clip_feat = torch.tensor(style_clip_feat, device=device)
        style_clip_feat = style_clip_feat / style_clip_feat.norm()
        loss_style = 1 - torch.cosine_similarity(gen_feat, style_clip_feat.unsqueeze(0))
        return loss_style.mean()

    # CLIP normalization parameters
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device).view(1, 3, 1, 1)

    for t in range(args.num_iter):
        optimizer.zero_grad()

        scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, canvas_height, shapes, shape_groups)
        render = pydiffvg.RenderFunction.apply
        img = render(canvas_width, canvas_height, 2, 2, t, None, *scene_args)

        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3) * (1 - img[:, :, 3:4])
        img = img.permute(2, 0, 1).unsqueeze(0).to(device)

        loss = 0

        for _ in range(num_augs):
            width, height = 224, 224
            startpoints, endpoints = transforms.RandomPerspective.get_params(width, height, 0.4)
            img_resized = torch.nn.functional.interpolate(img, size=(224, 224), mode='bilinear')
            target_resized = torch.nn.functional.interpolate(target_tensor, size=(224, 224), mode='bilinear')

            img_aug = transforms.functional.perspective(img_resized, startpoints, endpoints)
            target_aug = transforms.functional.perspective(target_resized, startpoints, endpoints)

            img_aug_norm = (img_aug - mean) / std
            target_aug_norm = (target_aug - mean) / std

            img_embed, img_inter = clip_extractor.encode_image(img_aug_norm)
            target_embed, target_inter = clip_extractor.encode_image(target_aug_norm)

            loss_semantic = 1 - torch.cosine_similarity(img_embed, target_embed, dim=1).mean()

            loss_geometric = 0
            for layer in img_inter:
                loss_geometric += torch.nn.functional.mse_loss(img_inter[layer], target_inter[layer])

            loss += args.semantic_weight * loss_semantic + args.geometric_weight * loss_geometric

        loss = loss / num_augs
        
        # New Part of Loss consisting of loss_style
        if style_features is not None:
            style_gram = style_features.get("gram_style", None)
            if style_gram is not None:
                loss_style = compute_gram_style_loss(img, style_gram)
                loss += args.style_weight * loss_style

        loss_history.append(loss.item())
        loss.backward()

        torch.nn.utils.clip_grad_norm_(points_vars, 1.0)
        optimizer.step()

        # Limit point coordinates to stay within bounds
        for path in shapes:
            path.points.data.clamp_(0, canvas_width)

        # Early stopping mechanism based on the average point movement
        with torch.no_grad():
            change = 0
            total_points = 0
            for i, p in enumerate(points_vars):
                dist = torch.norm(p - prev_points[i], dim=1).sum()
                change += dist.item()
                total_points += p.shape[0]
            avg_change = change / total_points
            prev_points = [p.detach().clone() for p in points_vars]

            if avg_change < args.early_stop_threshold:
                patience_counter += 1
            else:
                patience_counter = 0
            if patience_counter >= args.early_stop_patience:
                print(f"Early stopping at iteration {t}, avg point movement {avg_change:.4f} < threshold {args.early_stop_threshold}")
                break

        # Reduce learning rate mid-training to improve convergence
        if t == int(args.num_iter * 0.6):
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.4

        # Periodically save intermediate results and print loss
        if t % 50 == 0:
            print(f"Iter {t}, Loss: {loss.item():.4f}")
            pydiffvg.save_svg(os.path.join(output_dir, f"iter_{t}.svg"),
                              canvas_width, canvas_height, shapes, shape_groups)

            img_save = img.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
            img_save = np.clip(img_save, 0, 1)
            img_save = (img_save * 255).astype(np.uint8)
            frames.append(img_save)

    # Save final SVG result
    pydiffvg.save_svg(os.path.join(output_dir, "final.svg"),
                      canvas_width, canvas_height, shapes, shape_groups)

    # Plot and save loss curve
    try:
        plt.figure()
        plt.plot(loss_history)
        plt.title("Optimization Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.savefig(os.path.join(output_dir, "loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Failed to plot loss: {e}")

    # Save optimization process animation as GIF
    try:
        if frames:
            imageio.mimsave(os.path.join(output_dir, "evolution.gif"), frames, fps=10)
            print(f"Saved evolution GIF to {os.path.join(output_dir, 'evolution.gif')}")
    except Exception as e:
        print(f"Failed to save GIF: {e}")

    print(f"Done. Results saved to {output_dir}")

if __name__ == "__main__":
    main()

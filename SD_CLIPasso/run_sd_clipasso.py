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

"""
SD-CLIPasso: Stable Diffusion Prior for Stroke-Controlled Sketch Generation
Based on CLIPasso (Vinker et al., SIGGRAPH 2022)
"""
import random
import math
import matplotlib.pyplot as plt
import imageio

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="a drawing of a cat")
    parser.add_argument("--output_dir", type=str, default="SD_CLIPasso/results")
    parser.add_argument("--num_strokes", type=int, default=64, help="Number of strokes (control complexity)")
    parser.add_argument("--num_iter", type=int, default=500, help="Number of optimization iterations")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp_name", type=str, default=None, help="Custom name for the output folder")
    
    # New arguments for SD control
    parser.add_argument("--sd_style_suffix", type=str, default="realistic, full body, centered, high contrast, simple background, contours only, no blur, beautiful", help="Style to append to SD prompt")
    parser.add_argument("--sd_negative_prompt", type=str, default="cropped, out of frame, cut off, close up, partial, shadow, contact shadow, ground, floor, wall, complex background, text, watermark, drawing, sketch, cartoon, anime, grayscale,, blur, pale, washed out, overexposed, low contrast", help="Negative prompt for SD")
    
    # New arguments for CLIPasso control
    parser.add_argument("--geometric_weight", type=float, default=1.0, help="Weight for geometric loss (structure)")
    parser.add_argument("--semantic_weight", type=float, default=0.1, help="Weight for semantic loss (content)")
    parser.add_argument("--initial_stroke_width", type=float, default=2.0, help="Initial width of the strokes")
    
    # Early Stopping
    parser.add_argument("--early_stop_patience", type=int, default=50, help="Stop if no improvement for N iterations")
    parser.add_argument("--early_stop_threshold", type=float, default=0.01, help="Minimum average pixel movement to consider as change")


    args = parser.parse_args()

    # Create output directory based on prompt
    if args.exp_name:
        folder_name = args.exp_name
    else:
        # Clean prompt to be filesystem friendly
        safe_prompt = "".join([c for c in args.prompt if c.isalpha() or c.isdigit() or c==' ']).rstrip()
        safe_prompt = safe_prompt.replace(" ", "_")
        
        # Clean style suffix for filename
        # Remove leading comma/space if present
        style_str = args.sd_style_suffix.lstrip(", ")
        safe_style = "".join([c for c in style_str if c.isalpha() or c.isdigit() or c==' ']).rstrip()
        safe_style = safe_style.replace(" ", "_")
        # Truncate style to avoid too long paths (max 40 chars)
        safe_style = safe_style[:40]

        # Append parameters to folder name
        folder_name = f"{safe_prompt}_s{args.num_strokes}_style_{safe_style}"
    
    output_dir = os.path.join(args.output_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save configuration
    import json
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Generate Image with Stable Diffusion
    full_prompt = args.prompt + args.sd_style_suffix
    print(f"Generating image for prompt: {full_prompt}")
    model_id = "runwayml/stable-diffusion-v1-5"
    
    try:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
        pipe = pipe.to(device)
        pipe.safety_checker = None
        
        image = pipe(full_prompt, negative_prompt=args.sd_negative_prompt).images[0]
        
        # Save original SD image for comparison
        image.save(os.path.join(output_dir, "sd_raw.png"))
        
        # Post-process: Remove background using rembg
        print("Removing background...")
        try:
            # rembg returns an RGBA image with transparency
            img_no_bg = remove(image)
            
            # Save transparent image
            img_no_bg.save(os.path.join(output_dir, "sd_no_bg.png"))
            
            # Create a white background
            new_image = Image.new("RGB", img_no_bg.size, (255, 255, 255))
            
            # Composite the image onto the white background
            # img_no_bg has an alpha channel which we use as a mask
            new_image.paste(img_no_bg, (0, 0), img_no_bg)
            
            # --- Image Enhancement for better CLIPasso features ---
            # 1. Enhance Contrast (Keep this to separate object from bg)
            enhancer = ImageEnhance.Contrast(new_image)
            new_image = enhancer.enhance(1.2) 
            
            # 2. Apply Bilateral Filter (Smoothing)
            # CRITICAL: We must smooth out high-frequency textures (like fur/hair)
            # otherwise CLIPasso tries to draw every single hair, resulting in a messy sketch.
            # Bilateral filter smooths flat regions but preserves strong edges.
            import cv2
            img_cv = np.array(new_image)
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
            
            # d=9: Diameter of each pixel neighborhood
            # sigmaColor=75: Filter sigma in the color space (large value = mix disparate colors)
            # sigmaSpace=75: Filter sigma in the coordinate space
            img_smooth = cv2.bilateralFilter(img_cv, d=9, sigmaColor=75, sigmaSpace=75)
            
            new_image = Image.fromarray(cv2.cvtColor(img_smooth, cv2.COLOR_BGR2RGB))
            
            image = new_image
            print("Background removed and image smoothed (texture suppression) successfully.")
        except Exception as e:
            print(f"Background removal failed: {e}. Using original image.")
        
        # Save final processed image (used for CLIPasso)
        image_path = os.path.join(output_dir, "sd_final.png")
        image.save(image_path)
        print(f"Saved processed SD image to {image_path}")
        
        # --- CRITICAL: Cleanup Stable Diffusion to free up memory ---
        print("Cleaning up Stable Diffusion pipeline to save memory...")
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        print("Memory cleaned.")
        
    except Exception as e:
        print(f"Error loading Stable Diffusion: {e}")
        print("Using a dummy image for testing if SD fails.")
        image = Image.new('RGB', (512, 512), color = 'white')
        image.save(os.path.join(output_dir, "sd_output.png"))

    # 2. Setup CLIPasso
    print("Starting CLIPasso optimization...")
    # Enable GPU rendering if available
    # pydiffvg.set_use_gpu(torch.cuda.is_available())
    pydiffvg.set_use_gpu(False) # Force CPU for diffvg
    diffvg_device = "cpu"
    
    # Load CLIP
    # Use RN50 as recommended in CLIPasso paper for better geometric features
    clip_model, preprocess = clip.load("RN50", device=device, jit=False)
    
    # --- CLIP Feature Extractor for Geometric Loss ---
    # CLIPasso uses intermediate layers to preserve geometry.
    # Official implementation uses ResNet layers 3 and 4.
    class CLIPFeatureExtractor(torch.nn.Module):
        def __init__(self, model, target_layers=['layer3', 'layer4']):
            super().__init__()
            self.model = model
            self.target_layers = target_layers
            self.hooks = []
            self.intermediate_features = {}
            
            # Register hooks
            for layer_name in self.target_layers:
                target_layer = getattr(self.model.visual, layer_name)
                self.hooks.append(target_layer.register_forward_hook(self.make_hook(layer_name)))

        def make_hook(self, name):
            def hook_fn(module, input, output):
                # output is [N, C, H, W] for ResNet
                self.intermediate_features[name] = output
            return hook_fn

        def encode_image(self, img):
            self.intermediate_features = {}
            embedding = self.model.encode_image(img)
            return embedding, self.intermediate_features

        def remove_hooks(self):
            for h in self.hooks:
                h.remove()

    clip_extractor = CLIPFeatureExtractor(clip_model, target_layers=['layer3', 'layer4'])

    # Prepare target
    target_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device) # 1x3xHxW
    canvas_width, canvas_height = image.size

    # --- Improved Saliency-based Initialization (DoG) ---
    # Difference of Gaussians approximates saliency better than Canny
    import cv2
    img_cv = np.array(image)
    img_gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY).astype(np.float32) # Convert to float to avoid overflow
    
    g1 = cv2.GaussianBlur(img_gray, (3,3), 0)
    g2 = cv2.GaussianBlur(img_gray, (9,9), 0)
    dog = g1 - g2 # Difference
    
    # Normalize probability map
    prob_map = np.abs(dog)
    prob_map = prob_map / prob_map.max() # Normalize to 0-1
    prob_map[prob_map < 0.1] = 0 # Thresholding to remove background noise
    prob_map = cv2.pow(prob_map, 2) # Contrast stretching
    prob_map += 0.0001
    prob_map /= prob_map.sum()
    
    # Compute edge direction for better initialization
    # Gradients
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=5)
    # Gradient direction (perpendicular to edge)
    grad_dir = np.arctan2(sobely, sobelx)
    # Edge direction is perpendicular to gradient
    edge_dir = grad_dir + np.pi / 2
    
    # Sample starting points
    flat_indices = np.random.choice(prob_map.size, size=args.num_strokes, p=prob_map.flatten())
    y_coords, x_coords = np.unravel_index(flat_indices, prob_map.shape)
    
    shapes = []
    shape_groups = []
    
    for i in range(args.num_strokes):
        num_segments = 1
        num_control_points = torch.zeros(num_segments, dtype = torch.int32) + 2
        points = []
        
        p0 = (float(x_coords[i]), float(y_coords[i]))
        points.append(p0)
        
        # Get local edge direction
        # Ensure coords are within bounds
        ix = int(np.clip(x_coords[i], 0, canvas_width - 1))
        iy = int(np.clip(y_coords[i], 0, canvas_height - 1))
        angle = edge_dir[iy, ix]
        
        for j in range(num_segments):
            # Initialize as very small strokes to allow them to grow
            radius = 5.0 
            # Use the edge direction instead of random
            p1 = (p0[0] + radius * math.cos(angle), p0[1] + radius * math.sin(angle))
            p2 = (p1[0] + radius * math.cos(angle), p1[1] + radius * math.sin(angle))
            p3 = (p2[0] + radius * math.cos(angle), p2[1] + radius * math.sin(angle))
            points.append(p1)
            points.append(p2)
            points.append(p3)
            p0 = p3
            
        points = torch.tensor(points).to(diffvg_device)
        
        path = pydiffvg.Path(num_control_points = num_control_points,
                             points = points,
                             stroke_width = torch.tensor(args.initial_stroke_width).to(diffvg_device),
                             is_closed = False)
        shapes.append(path)
        
        path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(shapes) - 1]),
                                         fill_color = None,
                                         stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0]).to(diffvg_device))
        shape_groups.append(path_group)

    # Optimizer
    points_vars = []
    stroke_width_vars = []
    for path in shapes:
        path.points.requires_grad = True
        points_vars.append(path.points)
        # CLIPasso official: Stroke width is FIXED and not optimized.
        path.stroke_width.requires_grad = False 
        # stroke_width_vars.append(path.stroke_width)
    
    optimizer = torch.optim.Adam([
        {'params': points_vars, 'lr': 1.0},
        # {'params': stroke_width_vars, 'lr': 0.1}
    ])
    
    # Augmentation settings
    num_augs = 4
    # Perspective transform is crucial for robustness
    # Official: distortion_scale=0.5
    perspective_transform = transforms.RandomPerspective(distortion_scale=0.5, p=1.0)

    loss_history = []
    frames = []
    
    # Early stopping variables
    patience_counter = 0
    prev_points = [p.detach().clone() for p in points_vars]

    # Render loop
    for t in range(args.num_iter):
        optimizer.zero_grad()
        
        scene_args = pydiffvg.RenderFunction.serialize_scene(\
            canvas_width, canvas_height, shapes, shape_groups)
        
        render = pydiffvg.RenderFunction.apply
        # Render on CPU
        img = render(canvas_width, canvas_height, 2, 2, t, None, *scene_args)
        
        # img is HxWx4 (RGBA)
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3) * (1 - img[:, :, 3:4])
        
        # Permute to NCHW for CLIP
        img = img.permute(2, 0, 1).unsqueeze(0) # 1x3xHxW
        img_input = img.to(device)
        
        # Compute Loss
        loss = 0
        
        # Normalize constants
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device).view(1, 3, 1, 1)

        # --- Augmentation Loop ---
        for n in range(num_augs):
            # 1. Apply Perspective (Geometric Augmentation)
            # We must apply the SAME perspective to both
            # Torchvision RandomPerspective doesn't return params easily, so we use functional
            
            # Generate params
            width, height = 224, 224
            startpoints, endpoints = transforms.RandomPerspective.get_params(
                width, height, 0.4)
            
            # Resize first to 224 for CLIP
            img_resized = torch.nn.functional.interpolate(img_input, size=(224, 224), mode='bilinear')
            target_resized = torch.nn.functional.interpolate(target_tensor, size=(224, 224), mode='bilinear')
            
            # Apply perspective
            img_aug = transforms.functional.perspective(img_resized, startpoints, endpoints)
            target_aug = transforms.functional.perspective(target_resized, startpoints, endpoints)
            
            # Normalize
            img_aug_norm = (img_aug - mean) / std
            target_aug_norm = (target_aug - mean) / std
            
            # Extract features
            img_embed, img_inter = clip_extractor.encode_image(img_aug_norm)
            target_embed, target_inter = clip_extractor.encode_image(target_aug_norm)
            
            # A. Semantic Loss (Final embedding)
            loss_semantic = 1 - torch.cosine_similarity(img_embed, target_embed, dim=1).mean()
            
            # B. Geometric Loss (Intermediate features)
            # Official: Sum of L2 losses from intermediate layers
            loss_geometric = 0
            for idx in img_inter:
                loss_geometric += torch.nn.functional.mse_loss(img_inter[idx], target_inter[idx])
            
            # Weighted sum (Weights from official repo)
            # Semantic: 0.1, Geometric: 1.0
            loss += (args.semantic_weight * loss_semantic + args.geometric_weight * loss_geometric)

        loss = loss / num_augs
        
        loss_history.append(loss.item())

        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(points_vars, 1.0)
        # torch.nn.utils.clip_grad_norm_(stroke_width_vars, 1.0)
        
        optimizer.step()
        
        # Clamp
        for path in shapes:
            path.points.data.clamp_(0, canvas_width)
            # path.stroke_width.data.clamp_(0.5, 8.0)
            
        # --- Early Stopping Check ---
        with torch.no_grad():
            change = 0
            total_points = 0
            for i, p in enumerate(points_vars):
                # Calculate L2 distance between current and previous points
                dist = torch.norm(p - prev_points[i], dim=1).sum()
                change += dist.item()
                total_points += p.shape[0]
            
            avg_change = change / total_points
            
            # Update previous points
            prev_points = [p.detach().clone() for p in points_vars]
            
            if avg_change < args.early_stop_threshold:
                patience_counter += 1
            else:
                patience_counter = 0
                
            if patience_counter >= args.early_stop_patience:
                print(f"Early stopping triggered at iteration {t}: Average point movement {avg_change:.4f} < {args.early_stop_threshold}")
                break

        # LR Decay
        if t == int(args.num_iter * 0.6):
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.4

        if t % 50 == 0:
            print(f"Iter {t}, Loss: {loss.item()}")
            pydiffvg.save_svg(os.path.join(output_dir, f"iter_{t}.svg"),
                              canvas_width, canvas_height, shapes, shape_groups)
            
            # Save frame for GIF
            # img is 1x3xHxW, permute back to HxWx3 and scale to 0-255
            img_save = img.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
            img_save = np.clip(img_save, 0, 1)
            img_save = (img_save * 255).astype(np.uint8)
            frames.append(img_save)

    # Final save
    pydiffvg.save_svg(os.path.join(output_dir, "final.svg"),
                      canvas_width, canvas_height, shapes, shape_groups)
    
    # Plot Loss
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

    # Save GIF
    try:
        if len(frames) > 0:
            imageio.mimsave(os.path.join(output_dir, "evolution.gif"), frames, fps=10)
            print(f"Saved evolution GIF to {os.path.join(output_dir, 'evolution.gif')}")
    except Exception as e:
        print(f"Failed to save GIF: {e}")

    print(f"Done. Results saved to {output_dir}")

if __name__ == "__main__":
    main()

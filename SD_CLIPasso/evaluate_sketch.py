import torch
import clip
import pydiffvg
import os
import argparse
from PIL import Image
import numpy as np
import glob
import pandas as pd

def render_svg(svg_path, canvas_width=224, canvas_height=224, device='cpu'):
    """
    Renders an SVG file to a tensor image using pydiffvg.
    """
    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(svg_path)
    
    # Render
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_width, canvas_height, shapes, shape_groups
    )
    
    render = pydiffvg.RenderFunction.apply
    # Render (H, W, 4)
    img = render(canvas_width, canvas_height, 2, 2, 0, None, *scene_args)
    
    # Compose with white background
    # img is RGBA
    img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3).to(img.device) * (1 - img[:, :, 3:4])
    
    # Resize to target size for CLIP (usually 224)
    img = img.permute(2, 0, 1).unsqueeze(0) # 1, 3, H, W
    img = torch.nn.functional.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
    
    return img.to(device)

def load_image(img_path, device='cpu'):
    """
    Loads a standard image (PNG/JPG) and prepares it for CLIP.
    """
    img = Image.open(img_path).convert('RGB')
    # Basic transform
    # We can use standard CLIP preprocess, but we want tensor for consistency
    img = img.resize((224, 224), Image.Resampling.BICUBIC)
    img = np.array(img).astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img.to(device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="SD_CLIPasso/results", help="Directory containing result subfolders")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--generate_html", action="store_true", help="Generate an HTML report for user study")
    args = parser.parse_args()

    print(f"Loading CLIP model on {args.device}...")
    model, preprocess = clip.load("ViT-B/32", device=args.device)
    
    # Find all subdirectories
    subdirs = [f.path for f in os.scandir(args.results_dir) if f.is_dir()]
    
    results_data = []
    
    print(f"Found {len(subdirs)} result folders. Starting evaluation...")
    
    for folder in subdirs:
        folder_name = os.path.basename(folder)
        # Infer prompt from folder name (assuming folder name is safe_prompt)
        # This is a heuristic; ideally we'd save the prompt in a metadata file.
        # Reversing the "replace space with underscore" logic
        prompt_guess = folder_name.replace("_", " ")
        
        svg_path = os.path.join(folder, "final.svg")
        sd_img_path = os.path.join(folder, "sd_final.png")
        
        if not os.path.exists(svg_path):
            print(f"Skipping {folder}: final.svg not found")
            continue
            
        try:
            # 1. Render Sketch
            # Force CPU for pydiffvg as the library is compiled without CUDA support in this environment
            pydiffvg.set_use_gpu(False)
            sketch_tensor = render_svg(svg_path, device=args.device)
            
            # 2. Load SD Image (if exists)
            if os.path.exists(sd_img_path):
                sd_tensor = load_image(sd_img_path, device=args.device)
            else:
                sd_tensor = None
                
            # 3. Compute CLIP Features
            # Normalize for CLIP
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(args.device).view(1, 3, 1, 1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(args.device).view(1, 3, 1, 1)
            
            sketch_norm = (sketch_tensor - mean) / std
            
            with torch.no_grad():
                sketch_features = model.encode_image(sketch_norm)
                sketch_features /= sketch_features.norm(dim=-1, keepdim=True)
                
                # Text Score
                text_inputs = clip.tokenize([prompt_guess]).to(args.device)
                text_features = model.encode_text(text_inputs)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
                text_score = (sketch_features @ text_features.T).item()
                
                # Image Score (Fidelity to SD generation)
                img_score = None
                if sd_tensor is not None:
                    sd_norm = (sd_tensor - mean) / std
                    sd_features = model.encode_image(sd_norm)
                    sd_features /= sd_features.norm(dim=-1, keepdim=True)
                    img_score = (sketch_features @ sd_features.T).item()
            
            print(f"[{folder_name}] Text Score: {text_score:.4f} | Image Score: {img_score if img_score else 'N/A'}")
            
            results_data.append({
                "Folder": folder_name,
                "Prompt": prompt_guess,
                "CLIP_Text_Score": text_score,
                "CLIP_Image_Score": img_score,
                "SVG_Path": svg_path,
                "SD_Image_Path": sd_img_path
            })
            
        except Exception as e:
            print(f"Error processing {folder}: {e}")

    # Save CSV
    if not results_data:
        print("No results found or all failed. Exiting.")
        return

    df = pd.DataFrame(results_data)
    csv_path = os.path.join(args.results_dir, "evaluation_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nEvaluation complete. Metrics saved to {csv_path}")
    print(f"Average CLIP Text Score: {df['CLIP_Text_Score'].mean():.4f}")
    if 'CLIP_Image_Score' in df and not df['CLIP_Image_Score'].isnull().all():
        print(f"Average CLIP Image Score: {df['CLIP_Image_Score'].mean():.4f}")

    # Generate HTML for User Study
    if args.generate_html:
        html_content = """
        <html>
        <head>
            <style>
                body { font-family: sans-serif; padding: 20px; max-width: 1000px; margin: 0 auto; }
                .item { border: 1px solid #ccc; margin-bottom: 30px; padding: 20px; border-radius: 8px; background: #f9f9f9; }
                .header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #eee; padding-bottom: 10px; margin-bottom: 15px; }
                .prompt { font-size: 1.2em; font-weight: bold; color: #333; }
                .content { display: flex; gap: 40px; align-items: flex-start; }
                .img-box { text-align: center; flex: 1; }
                img { max-width: 300px; max-height: 300px; border: 1px solid #ddd; background: white; }
                .survey-box { flex: 1; background: white; padding: 15px; border-radius: 5px; border: 1px solid #eee; }
                .question { margin-bottom: 15px; }
                .question label { display: block; margin-bottom: 5px; font-weight: 600; }
                .options { display: flex; gap: 10px; }
                .options span { display: flex; align-items: center; gap: 4px; font-size: 0.9em; }
                .metrics { margin-top: 10px; font-size: 0.85em; color: #666; }
                .hidden-ref { display: none; }
                button { cursor: pointer; padding: 5px 10px; background: #eee; border: 1px solid #ccc; border-radius: 4px; }
            </style>
            <script>
                function toggleRef(id) {
                    var x = document.getElementById(id);
                    if (x.style.display === "none") {
                        x.style.display = "block";
                    } else {
                        x.style.display = "none";
                    }
                }
            </script>
        </head>
        <body>
            <h1>Text-to-Sketch User Study</h1>
            <p>Please evaluate the sketches based on the text prompt provided.</p>
            <p><i>Note: The "Reference Image" is the internal intermediate generation, hidden by default to simulate real user experience.</i></p>
        """
        
        for i, (_, row) in enumerate(df.iterrows()):
            # Relative paths for HTML
            rel_svg = os.path.relpath(row['SVG_Path'], args.results_dir)
            rel_sd = os.path.relpath(row['SD_Image_Path'], args.results_dir) if row['SD_Image_Path'] and os.path.exists(row['SD_Image_Path']) else None
            ref_id = f"ref_{i}"
            
            html_content += f"""
            <div class="item">
                <div class="header">
                    <div class="prompt">Prompt: "{row['Prompt']}"</div>
                    <div class="metrics">Auto-Eval: CLIP Score {row['CLIP_Text_Score']:.3f}</div>
                </div>
                
                <div class="content">
                    <div class="img-box">
                        <img src="{rel_svg}" alt="Sketch">
                    </div>
                    
                    <div class="survey-box">
                        <div class="question">
                            <label>1. Semantic Consistency (1-5)</label>
                            <div style="font-size: 0.8em; color: #666; margin-bottom: 5px;">Does this sketch accurately represent "{row['Prompt']}"?</div>
                            <div class="options">
                                <span><input type="radio" name="q1_{i}" value="1"> 1 (Bad)</span>
                                <span><input type="radio" name="q1_{i}" value="2"> 2</span>
                                <span><input type="radio" name="q1_{i}" value="3"> 3</span>
                                <span><input type="radio" name="q1_{i}" value="4"> 4</span>
                                <span><input type="radio" name="q1_{i}" value="5"> 5 (Perfect)</span>
                            </div>
                        </div>
                        
                        <div class="question">
                            <label>2. Abstract Quality (1-5)</label>
                            <div style="font-size: 0.8em; color: #666; margin-bottom: 5px;">Is the sketch aesthetically pleasing and recognizable?</div>
                            <div class="options">
                                <span><input type="radio" name="q2_{i}" value="1"> 1</span>
                                <span><input type="radio" name="q2_{i}" value="2"> 2</span>
                                <span><input type="radio" name="q2_{i}" value="3"> 3</span>
                                <span><input type="radio" name="q2_{i}" value="4"> 4</span>
                                <span><input type="radio" name="q2_{i}" value="5"> 5</span>
                            </div>
                        </div>

                        <div style="margin-top: 20px; border-top: 1px dashed #ddd; padding-top: 10px;">
                            <button onclick="toggleRef('{ref_id}')">Show/Hide Internal Reference</button>
                            <div id="{ref_id}" class="hidden-ref" style="margin-top: 10px; display: none;">
                                <img src="{rel_sd}" style="max-width: 150px;" alt="Reference">
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            """
            
        html_content += "</body></html>"
        
        html_path = os.path.join(args.results_dir, "user_study_form.html")
        with open(html_path, "w") as f:
            f.write(html_content)
        print(f"User study form saved to {html_path}")

if __name__ == "__main__":
    main()

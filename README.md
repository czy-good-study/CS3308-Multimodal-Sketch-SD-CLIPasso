# Sketch Generation with Sepecific Style
This repository is a modified version of the original SD-CLIPasso.

## Modifications & Improvements 
This modified version introduces support for generating images in multiple distinct artistic styles by leveraging a style library. The style library contains extracted style features — such as edge density, line thickness, and CLIP features — from reference style images.

During the sketch generation process, these style features guide stroke parameters and loss weighting, allowing fine-grained control over stroke complexity and visual aesthetics.

## How It Works
- Pre-extract and save style features from a collection of style reference images to build a style library.  
- At generation time, specify a style key to select the desired style from the library.  
- The system dynamically adjusts stroke parameters (including the number of strokes and stroke width) based on the selected style’s edge and line features.  
- Style-aware loss terms (e.g., Gram matrix-based style loss and CLIP-based semantic style loss) are incorporated into the optimization to better match the desired style characteristics.  
- This enables consistent generation of sketches or images that reflect the visual attributes of different styles.

## Usage
### Build a Style Library
Place your reference style images in the folder `SD_CLIPasso/style_extractor/input_style_images`, then run:
```bash
python SD_CLIPasso/style_extractor/style_extractor.py --input_dir "path/to/your/style_images" --output_path "path/to/your/style_library.pkl"
```
- `--input_dir`: directory containing style images(`.jpg`/`.png`).
- `--output_path`: file path where the serialized style library will be saved.

### Run SD-CLIPasso
```bash
python SD_CLIPasso/run_sd_clipasso.py --prompt "a horse" --style_library "path/to/your/style_library.pkl" --style_key "style_name" --style_weight 15
```

**New Arguments:**
- `--style_library`: Path to the style library created above (default: `"./style_extractor/style_library.pkl"`).
- `--style_key`: Style identifier in the library (usually the filename without extension).
- `--style_weight`:Controls the strength of style loss; higher values enforce stronger style adherence.

## Method Details
- **Style Feature Extraction**:Style images are processed to extract line-based features such as `edge_density` and `line_thickness`, along with deep style features represented by Gram matrices computed from VGG19 layers. These descriptors form the style library.
- **Parameter Adaptation**: During generation, the chosen style’s line features are used to automatically adjust parameters like the number of strokes and initial stroke width. This allows generated sketches to visually reflect the selected style’s characteristics.
- **Style Loss Guiding Optimization**: The optimization incorporates style-aware loss terms, enforcing the generated strokes to reproduce style statistics consistent with the selected style. This includes Gram-style loss on VGG features and CLIP-based semantic style features. The `style_weight` parameter balances between faithful content reconstruction and stylistic adherence.

## Examples
<div style="display: flex; justify-content: space-around; align-items: center;">

  <img src="./Sketch%20generated%20with%20different%20styles/style1.gif" alt="Style1" width="150" />

  <img src="./Sketch%20generated%20with%20different%20styles/style2.gif" alt="Style2" width="150" />

  <img src="./Sketch%20generated%20with%20different%20styles/style3.gif" alt="Style3" width="150" />
</div>

## Future Work
 - **Fine-grained disentanglement of style elements:** Current style transfer merges holistic features (e.g.,edge density, line thickness, texture) into a single style loss, lacking control over individual stylistic components. For instance, a user may want to combine the hatching texture of cross-stitch with the bold contours of comic books, or adjust the "sketchiness" of lines independently from color shading (if extended to colored sketches). Future work will decompose style into interpretable sub-components, using disentangled representation learning .This will allow users to modulate each component’s intensity via explicit controls, enabling highly personalized style combinations that are not limited to pre-defined exemplars.

  - **Adaptive balance between style adherence and semantic fidelity:** The current fixed style weight $\lambda_{sty}$ may lead to trade-offs. Overemphasizing style can distort the structural coherence of the sketch , while underemphasizing style fails to capture the target aesthetic. Future work may design a context-aware weight adjustment mechanism that dynamically balances $\lambda_{sty}$ with semantic $\lambda_{sem}$ and geometric $\lambda_{geo}$ losses based on the input text prompt and target style. 

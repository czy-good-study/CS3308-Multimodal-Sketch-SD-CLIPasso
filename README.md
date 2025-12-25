# Multimodal Task: Stable Diffusion Prior for Stroke-Controlled Sketch Generation

This project implements a robust text-to-sketch generation pipeline that leverages **Stable Diffusion** as a semantic prior and **CLIPasso** (CLIP-based differentiable rasterization) for stroke optimization.

## Overview

Unlike standard CLIPasso which optimizes strokes directly from a text prompt (often leading to abstract or unstable results), this method first generates a high-quality, structure-aware image using Stable Diffusion, and then uses this image as a geometric target for the sketch optimization.

**Pipeline:**
1.  **Text-to-Image:** Generate a reference image using Stable Diffusion (v1.5) with specific style prompts to encourage clean contours.
2.  **Preprocessing:** Remove background (rembg) and apply bilateral filtering to suppress texture while preserving edges.
3.  **Initialization:** Initialize Bezier curves based on Saliency Maps (Difference of Gaussians) and Edge Directions.
4.  **Optimization:** Optimize curve parameters using differentiable rasterization (`pydiffvg`) to minimize CLIP-based Semantic and Geometric losses.

## Features

- **Stable Diffusion Prior:** Uses SD to "imagine" the object first, ensuring better anatomy and composition.
- **Controllability:**
    - `num_strokes`: Control the abstraction level.
    - `prompt`: Text description of the object.
    - `seed`: Reproducible results.
- **Visualization:**
    - Generates `evolution.gif` to show the drawing process.
    - Plots `loss_curve.png` to analyze convergence.
    - Saves intermediate SVG snapshots.

## Prerequisites

### Dependencies
- Python 3.8+
- PyTorch with CUDA support
- [pydiffvg](https://github.com/BachiLi/diffvg) (Differentiable Vector Graphics)

### Installation

1.  **Install PyTorch:**
    Follow the [official guide](https://pytorch.org/get-started/locally/) to install PyTorch with CUDA support.

2.  **Install pydiffvg:**
    ```bash
    git clone https://github.com/BachiLi/diffvg
    cd diffvg
    git submodule update --init --recursive
    python setup.py install
    ```
    *Note: Requires C++ compiler and CMake.*

3.  **Install Python Requirements:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Run SD-CLIPasso (Main Method)

```bash
python SD_CLIPasso/run_sd_clipasso.py --prompt "a drawing of a horse" --num_strokes 64 --num_iter 1000
```

**Arguments:**
- `--prompt`: Text description (e.g., "a cat", "a car").
- `--num_strokes`: Number of strokes (e.g., 32 for abstract, 128 for detailed).
- `--num_iter`: Optimization iterations (default: 1000).
- `--seed`: Random seed.
- `--output_dir`: Directory to save results.

### 2. Baseline (Optional)
The files `main.py` and `sketch.py` contain a standard CLIPasso implementation (Text-to-Sketch without SD) for comparison.

```bash
python main.py --prompt "a camel" --num_strokes 64
```

## Results Structure

The output folder (e.g., `SD_CLIPasso/results/horse_...`) will contain:
- `final.svg`: The final vector sketch.
- `sd_raw.png`: Original Stable Diffusion generation.
- `sd_final.png`: Preprocessed target image (background removed).
- `evolution.gif`: Animation of the optimization process.
- `loss_curve.png`: Loss convergence plot.
- `config.json`: Experiment configuration.

### 3. Environment Check (Debug Mode)
Since this project relies on large pre-trained models (Stable Diffusion + CLIP), the first run will automatically download them (several GBs). 

To verify your environment (pydiffvg, torch, etc.) works *without* downloading models, use the test mode:

```bash
python SD_CLIPasso/run_sd_clipasso.py --test_env --no_clip --num_iter 100
```

## Evaluation

Use `SD_CLIPasso/evaluate_sketch.py` to compute CLIP similarity scores for generated sketches:

```bash
python SD_CLIPasso/evaluate_sketch.py --results_dir SD_CLIPasso/results
```

- `--num_iter`: Number of optimization iterations (default: 1000).
- `--output_dir`: Directory to save results.

## Method Details
1. **Initialization**: Initialize `N` random Bezier curves.
2. **Rasterization**: Use `pydiffvg` to render the curves into a 2D image.
3. **Loss Computation**:
   - **Semantic Loss**: Compute CLIP distance between the rendered sketch and the text prompt.
   - **Geometric Loss**: (Optional) Regularization to encourage smooth strokes.
4. **Optimization**: Backpropagate gradients to curve control points and update them.

## References

This project builds upon the following works:

1.  **CLIPasso: Semantically-Aware Object Sketching** (SIGGRAPH 2022)  
    *Yael Vinker, Ehsan Pajouheshgar, Jessica Y. Bo, Roman Christian Bachmann, Amit H. Bermano, Daniel Cohen-Or, Amir Zamir, Ariel Shamir*  
    [[Paper]](https://arxiv.org/abs/2202.05822) [[Code]](https://github.com/yael-vinker/CLIPasso)

2.  **Stable Diffusion** (CVPR 2022)  
    *Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer*  
    [[Paper]](https://arxiv.org/abs/2112.10752) [[Code]](https://github.com/CompVis/stable-diffusion)

3.  **CLIP** (ICML 2021)  
    *Alec Radford, et al.*  
    [[Paper]](https://arxiv.org/abs/2103.00020) [[Code]](https://github.com/openai/CLIP)

4.  **DiffVG** (TOG 2020)  
    *Tzu-Mao Li, Michal Lukáč, Michaël Gharbi, Jonathan Ragan-Kelley*  
    [[Paper]](https://people.csail.mit.edu/tzumao/diffvg/) [[Code]](https://github.com/BachiLi/diffvg)

## BibTeX
If you find this code useful or use it for your research, please cite the original papers:

```bibtex
@article{vinker2022clipasso,
  title={CLIPasso: Semantically-Aware Object Sketching},
  author={Vinker, Yael and Pajouheshgar, Ehsan and Bo, Jessica Y and Bachmann, Roman Christian and Bermano, Amit H and Cohen-Or, Daniel and Zamir, Amir and Shamir, Ariel},
  journal={ACM Transactions on Graphics (TOG)},
  volume={41},
  number={4},
  pages={1--11},
  year={2022},
  publisher={ACM New York, NY, USA}
}

@inproceedings{rombach2022high,
  title={High-resolution image synthesis with latent diffusion models},
  author={Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj{\"o}rn},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={10684--10695},
  year={2022}
}
```

## Acknowledgements
This code is developed for the CS3308 Multimodal Learning assignment.

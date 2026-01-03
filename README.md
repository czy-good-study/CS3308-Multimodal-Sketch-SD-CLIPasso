# SD-CLIPasso with Stroke Control

Extension of SD-CLIPasso framework with stroke-level control for text-to-sketch generation.

## Improvements and Features
- **Explicit control over stroke count**: Set exact stroke count with `--num_strokes`
- **Optional explicit control over stroke width**: Uniform width control for all strokes
- **Implicit optimization of stroke width**: Complexity-based width adjustment (thin strokes for complex regions, thick strokes for simple regions)
- **Control over stroke curvature**: Bezier curve regularization to avoid excessive bending

## How It Works

### 1. Stroke Count Control
The number of strokes is controlled explicitly via the `--num_strokes` parameter. Each stroke is initialized based on saliency maps from the input image, with strokes distributed more densely in complex regions.

### 2. Stroke Width Initialization
Initial stroke width is set via `--initial_stroke_width` (default: 2.0). When width optimization is enabled, this serves as the starting point for gradient-based optimization.

### 3. Stroke Width Optimization
Enable width optimization with `--optimize_width True` and set appropriate ranges

### 4. Bezier Curve Curvature Penalty
To prevent over-bending of strokes, we apply curvature regularization:
- `--curvature_limit`: Maximum allowed curvature (in radians)

## Usage
```bash
# Basic generation
python SD_CLIPasso/run_sd_clipasso.py --prompt "a drawing of a horse" --num_strokes 128 --num_iter 1000

# More strokes and thin strokes control
python SD_CLIPasso/run_sd_clipasso.py --prompt "a drawing of a horse" --num_strokes 256 --num_iter 1000 --initial_stroke_width 1.5

# Stroke width optimization
python SD_CLIPasso/run_sd_clipasso.py --prompt "a drawing of a horse" --num_strokes 256 --num_iter 1000 --initial_stroke_width 1.5 --optimize_width True

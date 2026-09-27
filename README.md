# ComfyUI-Qwen-Layered

A ComfyUI plugin integrating [Qwen-Image-Layered](https://huggingface.co/Qwen/Qwen-Image-Layered) — automatically decompose a single image into multiple RGBA transparent layers and export as PSD files.

[中文](README_CN.md)

## Features

- **Layer Decomposition** — Input an image, output multiple RGBA transparent layers (1~16 layers)
- **Variable Layer Count** — Customize the number of layers, from 3 to 16
- **PSD Export** — Download layered PSD files directly in the browser (frontend ag-psd, no Python dependency)
- **RGBA Generation** — Generate transparent-background images using Qwen-Image-2.1
- **ComfyUI LAYERS Compatible** — Convert to ComfyUI core LAYERS format for the layer panel
- **HuggingFace Auto-Download** — Automatically download models from HuggingFace on first use
- **VRAM Optimization** — Model CPU offload support for low-VRAM GPUs

## About Qwen-Image-2.1 and Qwen-Image-Layered

This plugin uses **Qwen-Image-Layered** for layer decomposition (a model specifically designed for layer separation, based on the Qwen-Image series).

| Model | Purpose | HuggingFace Repo |
|-------|---------|-------------------|
| Qwen-Image-Layered | Layer decomposition (recommended) | `Qwen/Qwen-Image-Layered` |
| Qwen-Image-2.1 | RGBA transparent image generation | `Qwen/Qwen-Image-2.1` |

> **Note**: Layer decomposition requires the Qwen-Image-Layered model. Qwen-Image-2.1 does not support layer decomposition; it is only used for RGBA transparent image generation.

## Nodes

| Node | Description |
|------|-------------|
| **Qwen Layered Load Model** | Load Qwen-Image-Layered model (diffusers format) |
| **Qwen Layered Decompose** | Decompose input image into multiple RGBA transparent layers |
| **Qwen Layered Save PSD** | Save layer PNGs + metadata; download PSD via browser button |
| **Qwen Layered To Layers** | Convert to ComfyUI core LAYERS format |
| **Qwen Layered RGBA Generate** | Generate RGBA transparent images using Qwen-Image-2.1 |

## Installation

Clone this repository into ComfyUI's `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/wuxiaren/ComfyUI-Qwen-Layered.git
```

Install dependencies:

```bash
cd ComfyUI-Qwen-Layered
pip install -r requirements.txt
```

Restart ComfyUI, and **QwenLayered** nodes will appear under the `QwenLayered` category.

### Dependencies

- `diffusers>=0.33.0` — HuggingFace diffusion pipelines (includes QwenImageLayeredPipeline)
- `transformers>=4.51.3` — Text encoder (Qwen2.5-VL)
- `accelerate>=0.20.0` — Model loading acceleration
- `Pillow` — Image processing

### Models

Automatically downloaded from HuggingFace on first use:

| Model | HuggingFace Repo | Purpose |
|-------|-------------------|---------|
| Qwen-Image-Layered | `Qwen/Qwen-Image-Layered` | Image layer decomposition |

#### Manual Model Placement

You can also manually download models to `ComfyUI/models/qwen_layered/`. The loader recursively scans up to 2 levels and auto-detects valid diffusers model directories containing `model_index.json`:

```
ComfyUI/models/qwen_layered/
├── model_index.json                                        # single model at top level
├── Qwen-Image-Layered/                                     # repo name subdirectory
│   └── model_index.json
└── Qwen/                                                   # org/repo subdirectory (HF naming)
    └── Qwen-Image-Layered/
        └── model_index.json
```

## Usage

### Basic Workflow

1. Add a **Load Image** node — load the input image
2. Add a **Qwen Layered Load Model** node — select `Qwen/Qwen-Image-Layered`
3. Add a **Qwen Layered Decompose** node — connect image and model, set layer count
4. Add a **Qwen Layered Save PSD** node — connect the `layers_data` output
5. Add a **Preview Image** node — connect the `preview` output to see the composite
6. Run the workflow
7. Click the **Download PSD** button on the Save PSD node to generate and download the PSD file

### Example Workflows

Preset workflows are provided in the `workflows/` directory:

| Workflow | Description |
|----------|-------------|
| `qwen_layered_basic.json` | Basic layer decomposition + PSD export |

Drag a `.json` file into ComfyUI to load the workflow.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `layers` | 4 | Number of layers to decompose into (1~16) |
| `resolution` | 640 | Processing resolution (640 or 1024) |
| `num_inference_steps` | 50 | Number of denoising steps |
| `true_cfg_scale` | 4.0 | Classifier-free guidance scale |
| `cfg_normalize` | false | Enable CFG normalization |
| `use_en_prompt` | true | Use English prompt for auto-captioning |
| `prompt` | "" | Optional image description prompt; auto-generated if empty |
| `seed` | 42 | Random seed |

### VRAM Requirements

- **Recommended**: 24GB+ VRAM (bf16, 640 resolution)
- **16GB VRAM**: Enable CPU offload, use 640 resolution
- **8GB VRAM**: May require a quantized version

## Output Layers

Decomposed layers are ordered back-to-front (bottom layer first in PSD). Each layer is an RGBA image with transparency, positioned correctly on the canvas.

## PSD Export

PSD files are generated in the browser using the [ag-psd](https://github.com/nicasiomg/ag-psd) library — no Python PSD library installation required. Each layer preserves its position on the canvas.

## Acknowledgments

- Layer decomposition uses the [Qwen-Image-Layered](https://huggingface.co/Qwen/Qwen-Image-Layered) model
- PSD generation uses the browser-side [ag-psd](https://github.com/nicasiomg/ag-psd) library
- PSD export implementation references [ComfyUI-See-through](https://github.com/jtydhr88/ComfyUI-See-through)

## License

MIT

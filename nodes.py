"""
ComfyUI-Qwen-Layered: Qwen Image Layer Decomposition with PSD Export
Wraps Qwen-Image-Layered (and Qwen-Image-2.1 RGBA) as ComfyUI custom nodes.
Supports layer separation and PSD file export.
"""
import os
import sys
import json
import uuid
import random
from datetime import datetime

import torch
import numpy as np
import folder_paths
import comfy.model_management as mm

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
def _log_vram(label):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"[QwenLayered VRAM] {label}: allocated={alloc:.2f}GB, reserved={reserved:.2f}GB", flush=True)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Constants & model folder registration
# ---------------------------------------------------------------------------
DEFAULT_LAYERED_REPO = "Qwen/Qwen-Image-Layered"
DEFAULT_QWEN21_REPO = "Qwen/Qwen-Image-2.1"

# Recommended resolutions for Qwen-Image-Layered
RESOLUTION_BUCKETS = [640, 1024]

folder_paths.add_model_folder_path("qwen_layered", os.path.join(folder_paths.models_dir, "qwen_layered"))


def _model_base_dirs():
    return folder_paths.get_folder_paths("qwen_layered")


try:
    os.makedirs(_model_base_dirs()[0], exist_ok=True)
except OSError:
    pass


def _scan_model_dirs():
    """Recursively find diffusers model dirs (containing model_index.json) up to 2 levels deep."""
    found = []
    for base in _model_base_dirs():
        if not os.path.isdir(base):
            continue
        if os.path.isfile(os.path.join(base, "model_index.json")) and "." not in found:
            found.append(".")
        for name in sorted(os.listdir(base)):
            d1 = os.path.join(base, name)
            if not os.path.isdir(d1):
                continue
            if os.path.isfile(os.path.join(d1, "model_index.json")):
                if name not in found:
                    found.append(name)
                continue
            for sub in sorted(os.listdir(d1)):
                d2 = os.path.join(d1, sub)
                rel = f"{name}/{sub}"
                if os.path.isdir(d2) and os.path.isfile(os.path.join(d2, "model_index.json")) and rel not in found:
                    found.append(rel)
    return found


def _resolve_model_path(model_name):
    bases = _model_base_dirs()
    if model_name == ".":
        for base in bases:
            if os.path.isfile(os.path.join(base, "model_index.json")):
                return base
        return bases[0]
    for base in bases:
        local = os.path.join(base, model_name)
        if os.path.isdir(local):
            return local
    basename = model_name.split("/")[-1]
    if basename != model_name:
        for base in bases:
            local_basename = os.path.join(base, basename)
            if os.path.isdir(local_basename):
                return local_basename
    return model_name


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
class QwenLayered_LayersData:
    """Output of Decompose: list of RGBA layers + full image."""
    def __init__(self, layers, full_image, resolution):
        self.layers = layers          # list of RGBA numpy arrays (H, W, 4), uint8
        self.full_image = full_image  # RGBA numpy (H, W, 4), uint8
        self.resolution = resolution


# ---------------------------------------------------------------------------
# Node 1: Load Model
# ---------------------------------------------------------------------------
class QwenLayered_LoadModel:
    @classmethod
    def INPUT_TYPES(s):
        local_models = _scan_model_dirs()
        model_list = local_models + [DEFAULT_LAYERED_REPO, DEFAULT_QWEN21_REPO]
        return {
            "required": {
                "model": (model_list, {"default": DEFAULT_LAYERED_REPO,
                                       "tooltip": "HuggingFace repo ID or local model folder. Use Qwen/Qwen-Image-Layered for layer decomposition."}),
            },
            "optional": {
                "dtype": (["bf16", "fp16", "fp32"], {"default": "bf16",
                                                      "tooltip": "Model dtype. bf16 recommended for Ampere+ GPUs."}),
                "cpu_offload": ("BOOLEAN", {"default": True,
                                             "tooltip": "Enable model CPU offload to reduce VRAM usage."}),
                "auto_download": ("BOOLEAN", {"default": True,
                                               "tooltip": "If model not found locally, download from HuggingFace."}),
            },
        }

    RETURN_TYPES = ("QWEN_LAYERED_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "QwenLayered"

    def load_model(self, model, dtype="bf16", cpu_offload=True, auto_download=True):
        from diffusers import QwenImageLayeredPipeline

        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        torch_dtype = dtype_map.get(dtype, torch.bfloat16)

        pretrained = _resolve_model_path(model)
        is_local = os.path.isdir(pretrained)
        local_only = is_local or not auto_download

        print(f"[QwenLayered] Loading model from: {pretrained} (dtype={dtype}, local={is_local})", flush=True)

        try:
            pipe = QwenImageLayeredPipeline.from_pretrained(
                pretrained,
                torch_dtype=torch_dtype,
                local_files_only=local_only,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Qwen-Image-Layered model from '{pretrained}'. "
                f"Note: layer decomposition requires Qwen-Image-Layered, not Qwen-Image-2.1. "
                f"Error: {e}"
            )

        if cpu_offload:
            try:
                pipe.enable_model_cpu_offload()
                print("[QwenLayered] Model CPU offload enabled", flush=True)
            except Exception as e:
                print(f"[QwenLayered] CPU offload failed ({e}), moving to GPU directly", flush=True)
                pipe = pipe.to("cuda")
        else:
            pipe = pipe.to("cuda")

        _log_vram("Model loaded")
        print(f"[QwenLayered] Model loaded successfully", flush=True)
        return (pipe,)


# ---------------------------------------------------------------------------
# Node 2: Decompose Layers
# ---------------------------------------------------------------------------
class QwenLayered_Decompose:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("QWEN_LAYERED_MODEL",),
            },
            "optional": {
                "layers": ("INT", {"default": 4, "min": 1, "max": 16, "step": 1,
                                    "tooltip": "Number of layers to decompose into."}),
                "resolution": (RESOLUTION_BUCKETS, {"default": 640,
                                                      "tooltip": "Processing resolution. 640 recommended, 1024 for high quality."}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 100, "step": 1,
                                                 "tooltip": "Number of denoising steps."}),
                "true_cfg_scale": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 20.0, "step": 0.1,
                                              "tooltip": "Classifier-free guidance scale."}),
                "cfg_normalize": ("BOOLEAN", {"default": False,
                                               "tooltip": "Enable CFG normalization."}),
                "use_en_prompt": ("BOOLEAN", {"default": True,
                                               "tooltip": "Use English prompt for automatic captioning."}),
                "prompt": ("STRING", {"default": "", "multiline": True,
                                       "tooltip": "Optional prompt describing the image content. Leave empty for auto-caption."}),
                "negative_prompt": ("STRING", {"default": " ",
                                                "tooltip": "Negative prompt. A single space enables CFG."}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("QWEN_LAYERED_LAYERS", "IMAGE")
    RETURN_NAMES = ("layers_data", "preview")
    FUNCTION = "decompose"
    CATEGORY = "QwenLayered"

    def decompose(self, image, model, layers=4, resolution=640, num_inference_steps=50,
                  true_cfg_scale=4.0, cfg_normalize=False, use_en_prompt=True,
                  prompt="", negative_prompt=" ", seed=42):
        from PIL import Image as PILImage

        # Convert ComfyUI IMAGE tensor (B, H, W, 3) float32 [0,1] to PIL RGBA
        if isinstance(image, torch.Tensor):
            img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
            if img_np.shape[-1] == 3:
                img_np = np.concatenate([img_np, np.full((*img_np.shape[:2], 1), 255, dtype=np.uint8)], axis=-1)
        else:
            img_np = image

        pil_image = PILImage.fromarray(img_np).convert("RGBA")

        # Resize to target resolution (maintain aspect, pad to square)
        w, h = pil_image.size
        scale = resolution / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        pil_image = pil_image.resize((new_w, new_h), PILImage.LANCZOS)

        # Pad to square
        canvas = PILImage.new("RGBA", (resolution, resolution), (0, 0, 0, 0))
        offset_x = (resolution - new_w) // 2
        offset_y = (resolution - new_h) // 2
        canvas.paste(pil_image, (offset_x, offset_y))
        pil_image = canvas

        print(f"[QwenLayered] Decomposing into {layers} layers at {resolution}x{resolution}, "
              f"steps={num_inference_steps}, cfg={true_cfg_scale}", flush=True)

        seed_everything(seed)
        generator = torch.Generator(device="cuda").manual_seed(seed)

        pipe = model
        device = pipe._execution_device if hasattr(pipe, "_execution_device") else "cuda"

        with torch.inference_mode():
            output = pipe(
                image=pil_image,
                prompt=prompt if prompt else None,
                negative_prompt=negative_prompt,
                true_cfg_scale=true_cfg_scale,
                layers=layers,
                num_inference_steps=num_inference_steps,
                resolution=resolution,
                cfg_normalize=cfg_normalize,
                use_en_prompt=use_en_prompt,
                generator=generator,
                output_type="np",
            )

        # output.images[0] is a list of RGBA numpy arrays: [full_image, layer1, layer2, ...]
        images = output.images[0]
        if not isinstance(images, list):
            images = [images]

        # Convert to uint8 RGBA
        layer_arrays = []
        for img in images:
            if isinstance(img, np.ndarray):
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8)
                if img.ndim == 3 and img.shape[-1] == 3:
                    img = np.concatenate([img, np.full((*img.shape[:2], 1), 255, dtype=np.uint8)], axis=-1)
                layer_arrays.append(img)
            elif hasattr(img, "convert"):
                arr = np.array(img.convert("RGBA"))
                layer_arrays.append(arr)

        # First image is the full reconstruction, rest are layers
        if len(layer_arrays) > 1:
            full_image = layer_arrays[0]
            layer_list = layer_arrays[1:layers + 1]
        else:
            full_image = layer_arrays[0] if layer_arrays else np.zeros((resolution, resolution, 4), dtype=np.uint8)
            layer_list = []

        # Ensure we have exactly `layers` layers
        while len(layer_list) < layers:
            layer_list.append(np.zeros((resolution, resolution, 4), dtype=np.uint8))
        layer_list = layer_list[:layers]

        print(f"[QwenLayered] Decomposition complete: {len(layer_list)} layers", flush=True)
        _log_vram("After decompose")

        # Preview: composite all layers
        preview = self._composite_preview(layer_list, resolution)
        layers_data = QwenLayered_LayersData(layer_list, full_image, resolution)

        return (layers_data, preview)

    @staticmethod
    def _composite_preview(layers, resolution):
        """Alpha-composite all layers from back to front for a preview."""
        canvas = np.zeros((resolution, resolution, 4), dtype=np.float32)
        for layer in reversed(layers):
            if layer.shape[0] != resolution or layer.shape[1] != resolution:
                continue
            lf = layer.astype(np.float32)
            alpha = lf[..., 3:4] / 255.0
            canvas[..., :3] = canvas[..., :3] * (1 - alpha) + lf[..., :3] * alpha
            canvas[..., 3] = np.clip(canvas[..., 3] + lf[..., 3], 0, 255)
        preview_rgb = (canvas[..., :3] / 255.0).astype(np.float32)
        return torch.from_numpy(preview_rgb).unsqueeze(0)


# ---------------------------------------------------------------------------
# Node 3: Save PSD
# ---------------------------------------------------------------------------
class QwenLayered_SavePSD:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "layers_data": ("QWEN_LAYERED_LAYERS",),
                "filename_prefix": ("STRING", {"default": "qwen_layered"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info_file",)
    FUNCTION = "save"
    CATEGORY = "QwenLayered"
    OUTPUT_NODE = True

    def save(self, layers_data, filename_prefix="qwen_layered"):
        from PIL import Image as PILImage

        layers = layers_data.layers
        full_image = layers_data.full_image
        resolution = layers_data.resolution

        output_dir = folder_paths.get_output_directory()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        uid = str(uuid.uuid4())[:8]

        layer_info_list = []

        # Save full image as the bottom layer
        full_filename = f"{filename_prefix}_{ts}_{uid}_full.png"
        PILImage.fromarray(full_image).save(os.path.join(output_dir, full_filename))
        layer_info_list.append({
            "name": "full_image",
            "filename": full_filename,
            "left": 0, "top": 0,
            "right": resolution, "bottom": resolution,
        })

        # Save each layer (back to front order for PSD)
        for idx, layer in enumerate(layers):
            layer_name = f"layer_{idx + 1}"
            layer_filename = f"{filename_prefix}_{ts}_{uid}_{layer_name}.png"
            PILImage.fromarray(layer).save(os.path.join(output_dir, layer_filename))

            # Compute bounding box of non-transparent pixels
            alpha = layer[..., 3]
            nz = np.argwhere(alpha > 10)
            if len(nz) > 0:
                y_min, x_min = nz.min(axis=0)
                y_max, x_max = nz.max(axis=0)
                left, top, right, bottom = int(x_min), int(y_min), int(x_max) + 1, int(y_max) + 1
            else:
                left, top, right, bottom = 0, 0, resolution, resolution

            layer_info_list.append({
                "name": layer_name,
                "filename": layer_filename,
                "left": left, "top": top,
                "right": right, "bottom": bottom,
            })

        info_filename = f"{filename_prefix}_{ts}_{uid}_layers.json"
        info_path = os.path.join(output_dir, info_filename)
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump({
                "prefix": filename_prefix,
                "timestamp": f"{ts}_{uid}",
                "layers": layer_info_list,
                "width": int(resolution),
                "height": int(resolution),
            }, f, indent=2, ensure_ascii=False)

        # Write a pointer file for the frontend button
        log_path = os.path.join(output_dir, "qwen_layered_psd_info.log")
        with open(log_path, "w") as f:
            f.write(info_filename)

        print(f"[QwenLayered] {len(layer_info_list)} layers saved. Use 'Download PSD' button to generate PSD.", flush=True)
        return (info_path,)


# ---------------------------------------------------------------------------
# Node 4: Convert to ComfyUI LAYERS format
# ---------------------------------------------------------------------------
class QwenLayered_ToLayers:
    """Convert QWEN_LAYERED_LAYERS into ComfyUI core LAYERS document."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "layers_data": ("QWEN_LAYERED_LAYERS",),
            },
        }

    RETURN_TYPES = ("LAYERS",)
    RETURN_NAMES = ("layers",)
    FUNCTION = "convert"
    CATEGORY = "QwenLayered"

    def convert(self, layers_data):
        layers = layers_data.layers
        resolution = layers_data.resolution

        items = []
        # Layers are stored back-to-front; z_index 0 = bottom
        for z_index, layer in enumerate(layers):
            tensor = torch.from_numpy(layer.astype(np.float32) / 255.0).unsqueeze(0)
            items.append({
                "image": tensor,
                "type": "raster",
                "name": f"layer_{z_index + 1}",
                "x": 0,
                "y": 0,
                "z_index": z_index,
            })

        document = {
            "version": 1,
            "layers": items,
            "canvas": (int(resolution), int(resolution)),
        }
        return (document,)


# ---------------------------------------------------------------------------
# Node 5: RGBA Image Generation (Qwen-Image-2.1)
# ---------------------------------------------------------------------------
class QwenLayered_RGBAGenerate:
    """Generate a single RGBA transparent image using Qwen-Image-2.1."""

    @classmethod
    def INPUT_TYPES(s):
        local_models = _scan_model_dirs()
        model_list = local_models + [DEFAULT_QWEN21_REPO]
        return {
            "required": {
                "prompt": ("STRING", {"default": "A cute cartoon dragon sticker", "multiline": True}),
                "model": (model_list, {"default": DEFAULT_QWEN21_REPO}),
            },
            "optional": {
                "width": ("INT", {"default": 2048, "min": 512, "max": 4096, "step": 32}),
                "height": ("INT", {"default": 2048, "min": 512, "max": 4096, "step": 32}),
                "num_inference_steps": ("INT", {"default": 40, "min": 1, "max": 100, "step": 1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "dtype": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "generate"
    CATEGORY = "QwenLayered"

    def generate(self, prompt, model, width=2048, height=2048,
                 num_inference_steps=40, seed=42, dtype="bf16"):
        from diffusers import QwenImage21Pipeline

        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        torch_dtype = dtype_map.get(dtype, torch.bfloat16)

        # Wrap with RGBA prompt format
        rgba_prompt = (
            f"This is an RGBA image with transparency. {prompt}. "
            f"The image has alpha channel and the background is transparent."
        )

        pretrained = _resolve_model_path(model)
        is_local = os.path.isdir(pretrained)

        print(f"[QwenLayered] Generating RGBA image with Qwen-Image-2.1 from {pretrained}", flush=True)

        pipe = QwenImage21Pipeline.from_pretrained(
            pretrained,
            torch_dtype=torch_dtype,
            local_files_only=is_local,
        ).to("cuda")

        generator = torch.Generator(device="cuda").manual_seed(seed)
        with torch.inference_mode():
            result = pipe(
                prompt=rgba_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                generator=generator,
            )

        image = result.images[0]
        if image.mode != "RGBA":
            image = image.convert("RGBA")

        arr = np.array(image).astype(np.float32) / 255.0
        rgb = arr[..., :3]
        alpha = arr[..., 3]

        out_image = torch.from_numpy(rgb).unsqueeze(0)
        out_mask = torch.from_numpy(alpha).unsqueeze(0)

        # Free memory
        del pipe
        mm.soft_empty_cache()

        return (out_image, out_mask)


# ---------------------------------------------------------------------------
# Mappings
# ---------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "QwenLayered_LoadModel": QwenLayered_LoadModel,
    "QwenLayered_Decompose": QwenLayered_Decompose,
    "QwenLayered_SavePSD": QwenLayered_SavePSD,
    "QwenLayered_ToLayers": QwenLayered_ToLayers,
    "QwenLayered_RGBAGenerate": QwenLayered_RGBAGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenLayered_LoadModel": "Qwen Layered Load Model",
    "QwenLayered_Decompose": "Qwen Layered Decompose",
    "QwenLayered_SavePSD": "Qwen Layered Save PSD",
    "QwenLayered_ToLayers": "Qwen Layered To Layers",
    "QwenLayered_RGBAGenerate": "Qwen Layered RGBA Generate",
}

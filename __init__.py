"""
ComfyUI-Qwen-Layered: Qwen Image Layer Decomposition with PSD Export
Wraps Qwen-Image-Layered (and Qwen-Image-2.1 RGBA) as ComfyUI custom nodes.
Supports layer separation and PSD file format export.
"""
import traceback

WEB_DIRECTORY = "./web"

print("[ComfyUI-Qwen-Layered] __init__.py loading...", flush=True)

try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    print(f"[ComfyUI-Qwen-Layered] Loaded {len(NODE_CLASS_MAPPINGS)} nodes: {list(NODE_CLASS_MAPPINGS.keys())}", flush=True)
except Exception as e:
    print(f"[ComfyUI-Qwen-Layered] Failed to import nodes: {e}", flush=True)
    traceback.print_exc()
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
__version__ = "0.1.0"

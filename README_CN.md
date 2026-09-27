# ComfyUI-Qwen-Layered

一个将 [Qwen-Image-Layered](https://huggingface.co/Qwen/Qwen-Image-Layered) 集成到 ComfyUI 的插件 — 将单张图像自动分解为多个 RGBA 透明图层，并支持导出为 PSD 文件。

[English](README.md)

## 功能特性

- **图层分离** — 输入一张图像，输出多个 RGBA 透明图层（支持 1~16 层）
- **可变层数** — 可自定义图层数量，从 3 层到 16 层按需分解
- **PSD 导出** — 直接在浏览器下载分层 PSD 文件（前端 ag-psd 实现，无需 Python 依赖）
- **RGBA 生成** — 支持使用 Qwen-Image-2.1 生成带透明背景的图像
- **ComfyUI LAYERS 兼容** — 可转换为 ComfyUI 核心 LAYERS 格式，配合图层面板使用
- **HuggingFace 自动下载** — 首次使用时自动从 HuggingFace 下载模型
- **显存优化** — 支持模型 CPU 卸载，适配低显存显卡

## 关于 Qwen-Image-2.1 与 Qwen-Image-Layered

本插件使用 **Qwen-Image-Layered** 进行图层分解（这是专门用于图层分离的模型，基于 Qwen-Image 系列）。

| 模型 | 用途 | HuggingFace 仓库 |
|------|------|-------------------|
| Qwen-Image-Layered | 图层分解（推荐） | `Qwen/Qwen-Image-Layered` |
| Qwen-Image-2.1 | RGBA 透明图像生成 | `Qwen/Qwen-Image-2.1` |

> **注意**：图层分离功能需要 Qwen-Image-Layered 模型。Qwen-Image-2.1 不支持图层分解，仅用于 RGBA 透明图像生成。

## 节点说明

| 节点 | 说明 |
|------|------|
| **Qwen Layered Load Model** | 加载 Qwen-Image-Layered 模型（diffusers 格式） |
| **Qwen Layered Decompose** | 将输入图像分解为多个 RGBA 透明图层 |
| **Qwen Layered Save PSD** | 保存图层 PNG + 元数据；通过浏览器按钮下载 PSD |
| **Qwen Layered To Layers** | 转换为 ComfyUI 核心 LAYERS 格式 |
| **Qwen Layered RGBA Generate** | 使用 Qwen-Image-2.1 生成带透明背景的 RGBA 图像 |

## 安装

将此仓库克隆到 ComfyUI 的 `custom_nodes` 目录：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/wuxiaren/ComfyUI-Qwen-Layered.git
```

安装依赖：

```bash
cd ComfyUI-Qwen-Layered
pip install -r requirements.txt
```

重启 ComfyUI，**QwenLayered** 节点将出现在 `QwenLayered` 分类下。

### 依赖

- `diffusers>=0.33.0` — HuggingFace 扩散管线（包含 QwenImageLayeredPipeline）
- `transformers>=4.51.3` — 文本编码器（Qwen2.5-VL）
- `accelerate>=0.20.0` — 模型加载加速
- `Pillow` — 图像处理

### 模型

首次使用时自动从 HuggingFace 下载：

| 模型 | HuggingFace 仓库 | 用途 |
|------|-------------------|------|
| Qwen-Image-Layered | `Qwen/Qwen-Image-Layered` | 图像图层分解 |

#### 手动放置模型

也可以手动下载模型放到 `ComfyUI/models/qwen_layered/` 目录。加载器会递归扫描两层目录、自动识别所有含 `model_index.json` 的合法 diffusers 模型目录：

```
ComfyUI/models/qwen_layered/
├── model_index.json                                        # 文件直接放在顶层（单模型）
├── Qwen-Image-Layered/                                     # 仅 repo 名子目录
│   └── model_index.json
└── Qwen/                                                   # org/repo 子目录（与 HF 命名一致）
    └── Qwen-Image-Layered/
        └── model_index.json
```

## 使用方法

### 基本工作流

1. 添加 **Load Image** 节点 — 加载输入图像
2. 添加 **Qwen Layered Load Model** 节点 — 选择 `Qwen/Qwen-Image-Layered`
3. 添加 **Qwen Layered Decompose** 节点 — 连接图像和模型，设置图层数量
4. 添加 **Qwen Layered Save PSD** — 连接 `layers_data` 输出
5. 添加 **Preview Image** — 连接 `preview` 输出查看合成预览
6. 运行工作流
7. 点击 Save PSD 节点上的 **Download PSD** 按钮生成并下载 PSD 文件

### 示例工作流

`workflows/` 目录中提供了预设工作流：

| 工作流 | 说明 |
|--------|------|
| `qwen_layered_basic.json` | 基础图层分解 + PSD 导出 |

将 `.json` 文件拖入 ComfyUI 即可加载工作流。

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `layers` | 4 | 分解的图层数量（1~16） |
| `resolution` | 640 | 处理分辨率（640 或 1024） |
| `num_inference_steps` | 50 | 扩散去噪步数 |
| `true_cfg_scale` | 4.0 | Classifier-free guidance 缩放 |
| `cfg_normalize` | false | 是否启用 CFG 归一化 |
| `use_en_prompt` | true | 使用英文 prompt 自动描述图像 |
| `prompt` | "" | 可选的图像描述 prompt，留空则自动生成 |
| `seed` | 42 | 随机种子 |

### 显存需求

- **推荐**：24GB+ 显存（bf16，640 分辨率）
- **16GB 显存**：开启 CPU offload，使用 640 分辨率
- **8GB 显存**：可能需要使用量化版本

## 输出图层

分解产生的图层按从后到前的顺序排列（PSD 中底层在前）。每个图层都是带透明度的 RGBA 图像，定位在画布上的正确位置。

## PSD 导出

PSD 文件在浏览器端使用 [ag-psd](https://github.com/nicasiomg/ag-psd) 库生成，无需安装 Python PSD 库。每个图层保留其在画布上的位置信息。

## 致谢

- 图层分解使用 [Qwen-Image-Layered](https://huggingface.co/Qwen/Qwen-Image-Layered) 模型
- PSD 生成使用浏览器端的 [ag-psd](https://github.com/nicasiomg/ag-psd) 库
- 参考了 [ComfyUI-See-through](https://github.com/jtydhr88/ComfyUI-See-through) 的 PSD 导出实现

## 许可证

MIT

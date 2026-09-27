/**
 * QwenLayered PSD Generator
 * Uses ag-psd to generate PSD files in the browser from decomposed layers.
 */
const { app } = window.comfyAPI.app;
const { api } = window.comfyAPI.api;

let agPsdLoaded = false;
let agPsdLoadPromise = null;

const _ownScriptDir = (() => {
    const src = document.currentScript?.src;
    if (src) return src.substring(0, src.lastIndexOf("/") + 1);
    return null;
})();

async function ensureAgPsdLoaded() {
    if (agPsdLoaded) return;
    if (agPsdLoadPromise) return agPsdLoadPromise;
    agPsdLoadPromise = new Promise((resolve, reject) => {
        if (_ownScriptDir) {
            const script = document.createElement("script");
            script.src = _ownScriptDir + "ag-psd.bundle.js";
            script.onload = () => {
                agPsdLoaded = true;
                console.log("[QwenLayered] ag-psd bundle loaded");
                resolve();
            };
            script.onerror = (e) => {
                console.error("[QwenLayered] Failed to load ag-psd bundle:", e);
                reject(new Error("Failed to load ag-psd bundle"));
            };
            document.head.appendChild(script);
        } else {
            const variants = [
                "ComfyUI-Qwen-Layered",
                "comfyui-qwen-layered",
                "ComfyUI-qwen-layered",
            ];
            let idx = 0;
            function tryNext() {
                if (idx >= variants.length) {
                    reject(new Error("Failed to load ag-psd bundle from any path"));
                    return;
                }
                const s = document.createElement("script");
                s.src = api.fileURL(`/extensions/${variants[idx]}/ag-psd.bundle.js`);
                idx++;
                s.onload = () => {
                    agPsdLoaded = true;
                    console.log(`[QwenLayered] ag-psd bundle loaded from: ${s.src}`);
                    resolve();
                };
                s.onerror = () => {
                    s.remove();
                    tryNext();
                };
                document.head.appendChild(s);
            }
            tryNext();
        }
    });
    return agPsdLoadPromise;
}

function loadImage(url) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = "anonymous";
        img.onload = () => resolve(img);
        img.onerror = () => reject(new Error(`Failed to load: ${url}`));
        img.src = url;
    });
}

async function createPSD(layerInfo) {
    await ensureAgPsdLoaded();
    const { layers, width, height, prefix, timestamp } = layerInfo;

    console.log(`[QwenLayered] Creating PSD: ${width}x${height}, ${layers.length} layers`);

    // Create composite canvas
    const compositeCanvas = document.createElement("canvas");
    compositeCanvas.width = width;
    compositeCanvas.height = height;
    const compositeCtx = compositeCanvas.getContext("2d");

    const psdLayers = [];

    // Process layers: first layer in list is bottom (full_image), last is top
    // ag-psd children order: first child is top layer
    for (let i = layers.length - 1; i >= 0; i--) {
        const layer = layers[i];
        const filename = layer.filename;
        if (!filename) continue;

        const url = api.apiURL(`/view?filename=${encodeURIComponent(filename)}&type=output&t=${Date.now()}`);
        const img = await loadImage(url);

        const lw = layer.right - layer.left;
        const lh = layer.bottom - layer.top;

        // Create layer canvas at the layer's bounding box size
        const layerCanvas = document.createElement("canvas");
        layerCanvas.width = lw;
        layerCanvas.height = lh;
        const layerCtx = layerCanvas.getContext("2d");
        layerCtx.drawImage(img, 0, 0, lw, lh);

        // Draw to composite at the correct position
        compositeCtx.drawImage(img, layer.left, layer.top, lw, lh);

        psdLayers.push({
            name: layer.name,
            canvas: layerCanvas,
            left: layer.left,
            top: layer.top,
            right: layer.right,
            bottom: layer.bottom,
            blendMode: "normal",
            opacity: 1,
        });
    }

    const psd = {
        width,
        height,
        canvas: compositeCanvas,
        children: psdLayers,
    };

    const psdBuffer = window.AgPsd.writePsd(psd);
    const blob = new Blob([psdBuffer], { type: "application/octet-stream" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${prefix}_${timestamp}.psd`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    console.log(`[QwenLayered] PSD downloaded: ${prefix}_${timestamp}.psd`);
}

app.registerExtension({
    name: "ComfyUI-Qwen-Layered.SavePSD",
    async nodeCreated(node) {
        if (node.comfyClass !== "QwenLayered_SavePSD") return;
        console.log("[QwenLayered] Setting up frontend PSD generator");

        const dlBtn = node.addWidget(
            "button",
            "Download PSD",
            "Download PSD",
            async () => {
                try {
                    dlBtn.name = "Generating PSD...";
                    const logResp = await fetch(
                        api.apiURL("/view?filename=qwen_layered_psd_info.log&type=output&t=" + Date.now())
                    );
                    if (!logResp.ok) {
                        alert("Please run the workflow first to generate layers.");
                        return;
                    }
                    const infoFilename = (await logResp.text()).trim();
                    const infoResp = await fetch(
                        api.apiURL(`/view?filename=${encodeURIComponent(infoFilename)}&type=output&t=${Date.now()}`)
                    );
                    if (!infoResp.ok) {
                        alert("Failed to load layer information.");
                        return;
                    }
                    const layerInfo = await infoResp.json();
                    await createPSD(layerInfo);
                } catch (error) {
                    console.error("[QwenLayered] Error:", error);
                    alert(`Failed to generate PSD: ${error.message}`);
                } finally {
                    dlBtn.name = "Download PSD";
                }
            }
        );
        dlBtn.color = "#10B981";
        dlBtn.bgcolor = "#059669";
    },
});

console.log("[QwenLayered] PSD extension loaded");

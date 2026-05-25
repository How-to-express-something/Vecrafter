import json
import time
import random
import requests
import websocket
import base64
import os
import re
import logging
from io import BytesIO
from PIL import Image
from typing import Dict, Any, List
import threading
from pathlib import Path
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler

from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

from image_preprocessor import (
    ImagePreprocessor,
    PreprocessConfig,
    PreprocessResult,
    AspectRatioPreset,
    OutputFormat,
    ResizeMode,
)
from vector_converter import (
    VectorConverter,
    VectorizationConfig,
    VectorizationResult,
    ColorQuantMethod,
    PathFittingMethod,
    ContourMethod,
)


# ======================= 项目路径 =======================
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LOGS_DIR = _PROJECT_ROOT / "logs"
_LOGS_DIR.mkdir(parents=True, exist_ok=True)
_OUTPUT_DIR = _PROJECT_ROOT / "output"
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ======================= 日志配置 =======================
_logger = logging.getLogger("vecrafter.backend")
_logger.setLevel(logging.INFO)

_handler = RotatingFileHandler(
    _LOGS_DIR / "backend.log",
    maxBytes=5 * 1024 * 1024,
    backupCount=5,
    encoding="utf-8",
)
_handler.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))
_logger.addHandler(_handler)


# ======================= 工具函数 =======================
def make_slug(text: str, max_len: int = 16) -> str:
    """将任意文本转为文件系统安全的 ASCII slug"""
    safe = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    slug = re.sub(r'\s+', '_', safe.strip())
    return slug[:max_len] if slug else "untitled"


# ======================= ComfyUI 封装 =======================
class ComfyUIWrapper:
    def __init__(self, server_url: str = "http://10.195.153.60:8188"):
        self.server_url = server_url.rstrip("/")
        self.client_id = str(int(time.time() * 1000))
        self.ws_url = server_url.replace("http://", "ws://", 1).rstrip("/") + f"/ws?clientId={self.client_id}"

    def load_workflow(self, workflow_path: str) -> Dict[str, Any]:
        with open(workflow_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def queue_prompt(self, workflow: Dict[str, Any]) -> str:
        payload = {"prompt": workflow, "client_id": self.client_id}
        resp = requests.post(f"{self.server_url}/prompt", json=payload)
        if resp.status_code != 200:
            raise Exception(f"Queue prompt failed: {resp.text}")
        return resp.json()["prompt_id"]

    def get_history(self, prompt_id: str) -> Dict[str, Any]:
        resp = requests.get(f"{self.server_url}/history/{prompt_id}")
        if resp.status_code != 200:
            raise Exception(f"Get history failed: {resp.text}")
        return resp.json()

    def wait_for_prompt(self, prompt_id: str, timeout: int = 600):
        """通过 WebSocket 等待生成完成，超时后回退到轮询模式"""
        done = threading.Event()
        try:
            ws = websocket.WebSocket()
            ws.settimeout(10)
            ws.connect(self.ws_url)
            _logger.info("WebSocket connected, waiting for prompt_id=%s", prompt_id)
        except Exception as e:
            _logger.warning("WebSocket connect failed (%s), falling back to polling", e)
            self._poll_until_done(prompt_id, timeout)
            return

        start_time = time.time()
        try:
            while not done.is_set():
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    _logger.warning("WebSocket wait timed out after %ds, falling back to polling", timeout)
                    ws.close()
                    self._poll_until_done(prompt_id, timeout)
                    return
                try:
                    msg = ws.recv()
                except websocket.WebSocketTimeoutException:
                    continue
                except Exception as e:
                    _logger.warning("WebSocket recv error (%s), falling back to polling", e)
                    ws.close()
                    self._poll_until_done(prompt_id, timeout)
                    return
                if not msg:
                    continue
                try:
                    data = json.loads(msg)
                except json.JSONDecodeError:
                    continue
                msg_type = data.get("type", "")
                _logger.info("WS msg: type=%s, node=%s, prompt_id=%s",
                             msg_type, data.get("data", {}).get("node"),
                             data.get("data", {}).get("prompt_id"))
                if msg_type == "executing":
                    node = data.get("data", {}).get("node")
                    pid = data.get("data", {}).get("prompt_id")
                    if node is None and pid == prompt_id:
                        _logger.info("Generation completed via WebSocket: prompt_id=%s", prompt_id)
                        done.set()
                elif msg_type == "execution_error":
                    pid = data.get("data", {}).get("prompt_id")
                    if pid == prompt_id:
                        ws.close()
                        raise Exception(f"ComfyUI execution error: {data}")
        finally:
            try:
                ws.close()
            except Exception:
                pass

    def _poll_until_done(self, prompt_id: str, timeout: int = 600):
        """轮询 history API 直到生成完成"""
        _logger.info("Polling history for prompt_id=%s (timeout=%ds)", prompt_id, timeout)
        start = time.time()
        while time.time() - start < timeout:
            time.sleep(2)
            try:
                history = self.get_history(prompt_id)
                if prompt_id in history:
                    _logger.info("History found for prompt_id=%s via polling", prompt_id)
                    return
            except Exception:
                pass
        raise Exception(f"Timeout waiting for prompt_id={prompt_id} after {timeout}s")

    def get_output_images(self, prompt_id: str) -> List[Image.Image]:
        """从历史记录中提取输出图片，带重试"""
        max_retries = 5
        for attempt in range(max_retries):
            history = self.get_history(prompt_id)
            prompt_info = history.get(prompt_id, {})
            outputs = prompt_info.get("outputs", {})

            images = []
            for node_id, output in outputs.items():
                if "images" in output:
                    for img_info in output["images"]:
                        filename = img_info["filename"]
                        subfolder = img_info.get("subfolder", "")
                        img_type = img_info.get("type", "output")
                        params = {
                            "filename": filename,
                            "subfolder": subfolder,
                            "type": img_type,
                        }
                        resp = requests.get(f"{self.server_url}/view", params=params)
                        if resp.status_code == 200:
                            img = Image.open(BytesIO(resp.content))
                            images.append(img)
                        else:
                            _logger.warning("Failed to download %s (HTTP %d): %s",
                                            filename, resp.status_code, resp.text[:100])
            if images:
                _logger.info("Downloaded %d image(s) for prompt_id=%s (attempt %d)",
                             len(images), prompt_id, attempt + 1)
                return images

            if attempt < max_retries - 1:
                _logger.warning("No images in history yet for prompt_id=%s, retrying (%d/%d)...",
                                prompt_id, attempt + 1, max_retries)
                time.sleep(2)

        _logger.error("Failed to get images for prompt_id=%s after %d retries", prompt_id, max_retries)
        return []

    def generate(
        self,
        workflow_path: str,
        positive_prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        seed: int = -1,
        steps: int = 8,
        cfg: float = 1.1,
        sampler_name: str = "euler",
        scheduler: str = "sgm_uniform",
        denoise: float = 1.0,
        **extra_params
    ) -> Dict[str, Any]:
        t_start = time.time()

        workflow = self.load_workflow(workflow_path)

        if seed == -1:
            seed = random.randint(0, 2**32 - 1)

        # CFG_test.json 节点 ID 映射:
        #   "5" → positive prompt (CLIPTextEncode)
        #   "6" → negative prompt (CLIPTextEncode)
        #   "4" → KSampler
        #   "7" → EmptyLatentImage

        if "5" in workflow:
            workflow["5"]["inputs"]["text"] = positive_prompt

        if "6" in workflow:
            workflow["6"]["inputs"]["text"] = negative_prompt

        if "4" in workflow:
            workflow["4"]["inputs"]["seed"] = seed
            workflow["4"]["inputs"]["steps"] = steps
            workflow["4"]["inputs"]["cfg"] = cfg
            workflow["4"]["inputs"]["sampler_name"] = sampler_name
            workflow["4"]["inputs"]["scheduler"] = scheduler
            workflow["4"]["inputs"]["denoise"] = denoise

        if "7" in workflow:
            workflow["7"]["inputs"]["width"] = width
            workflow["7"]["inputs"]["height"] = height

        prompt_id = self.queue_prompt(workflow)
        self.wait_for_prompt(prompt_id)
        images = self.get_output_images(prompt_id)

        elapsed = round(time.time() - t_start, 2)

        return {
            "images": images,
            "prompt_id": prompt_id,
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "elapsed_seconds": elapsed,
        }


# ======================= FastAPI 服务 =======================
app = FastAPI(title="Vecrafter Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config")
DEFAULT_WORKFLOW = os.path.join(_config_dir, "CFG_test.json")

wrapper = ComfyUIWrapper()
preprocessor = ImagePreprocessor()
vector_converter = VectorConverter()


# ======================= 请求/响应模型 =======================
class GenerateRequest(BaseModel):
    text: str
    style_prompt: str = ""
    negative_prompt: str = "blurry, low quality, distorted text, extra letters, missing letters, messy background"
    seed: int = -1
    width: int = 1024
    height: int = 1024
    steps: int = 8
    cfg: float = 1.1
    sampler_name: str = "euler"
    scheduler: str = "sgm_uniform"


class GenerationMetadata(BaseModel):
    prompt_id: str
    text: str
    style_prompt: str = ""
    negative_prompt: str = ""
    seed: int
    width: int
    height: int
    steps: int
    cfg: float
    sampler_name: str
    scheduler: str
    timestamp_utc: str
    generation_time_seconds: float


class GenerateResponse(BaseModel):
    success: bool
    images: List[str] = []
    metadata: GenerationMetadata | None = None
    preview_path: str | None = None
    metadata_path: str | None = None


# ======================= 磁盘存储 =======================
def _save_result(
    images: List[Image.Image],
    metadata: GenerationMetadata,
) -> tuple[str | None, str | None]:
    """保存 preview.png 和 metadata.json 到磁盘。返回 (preview_path, metadata_path) 相对于项目根目录"""
    slug = make_slug(metadata.text)
    ts_local = datetime.fromisoformat(metadata.timestamp_utc.replace("Z", "+00:00"))
    ts_local = ts_local.astimezone()
    dir_name = f"{ts_local.strftime('%H%M%S')}_{metadata.seed}_{slug}"
    date_dir = ts_local.strftime("%Y-%m-%d")

    out_dir = _OUTPUT_DIR / date_dir / dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    preview_path = out_dir / "preview.png"
    images[0].save(preview_path, format="PNG")

    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata.model_dump(), f, ensure_ascii=False, indent=2)

    rel_preview = str(preview_path.relative_to(_PROJECT_ROOT))
    rel_meta = str(meta_path.relative_to(_PROJECT_ROOT))
    return rel_preview, rel_meta


# ======================= 端点 =======================
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/results")
def list_results(limit: int = Query(20, ge=1, le=100)):
    """列出 output/ 中所有历史生成结果，按时间降序"""
    results = []
    if not _OUTPUT_DIR.exists():
        return results

    for date_dir in sorted(_OUTPUT_DIR.iterdir(), reverse=True):
        if not date_dir.is_dir():
            continue
        for run_dir in sorted(date_dir.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue
            meta_file = run_dir / "metadata.json"
            if not meta_file.exists():
                continue
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, IOError):
                continue
            meta["preview_path"] = str((run_dir / "preview.png").relative_to(_PROJECT_ROOT))
            meta["metadata_path"] = str(meta_file.relative_to(_PROJECT_ROOT))
            results.append(meta)
            if len(results) >= limit:
                return results
    return results


@app.get("/results/image")
def serve_preview(path: str = Query(..., description="Relative path to preview.png")):
    """提供预览图"""
    target = (_PROJECT_ROOT / path).resolve()
    # 安全检查：确保不越出 output/ 目录
    if not str(target).startswith(str(_OUTPUT_DIR.resolve())):
        raise HTTPException(status_code=403, detail="Access denied")
    if not target.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return StreamingResponse(target.open("rb"), media_type="image/png")


@app.post("/generate")
def generate_image(req: GenerateRequest):
    _logger.info(
        "Request received: text=%r, style=%r, seed=%d, steps=%d, cfg=%.2f, sampler=%s, scheduler=%s, resolution=%dx%d",
        req.text, req.style_prompt[:60] if req.style_prompt else "",
        req.seed, req.steps, req.cfg, req.sampler_name, req.scheduler,
        req.width, req.height,
    )

    t_req_start = time.time()

    try:
        positive = req.text
        if req.style_prompt:
            positive = f"{req.style_prompt},{req.text}"

        result = wrapper.generate(
            workflow_path=DEFAULT_WORKFLOW,
            positive_prompt=positive,
            negative_prompt=req.negative_prompt,
            width=req.width,
            height=req.height,
            seed=req.seed,
            steps=req.steps,
            cfg=req.cfg,
            sampler_name=req.sampler_name,
            scheduler=req.scheduler,
        )

        images = result["images"]
        if not images:
            _logger.error("No images generated for prompt_id=%s", result["prompt_id"])
            raise HTTPException(status_code=500, detail="No images generated")

        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        metadata = GenerationMetadata(
            prompt_id=result["prompt_id"],
            text=req.text,
            style_prompt=req.style_prompt,
            negative_prompt=req.negative_prompt,
            seed=result["seed"],
            width=req.width,
            height=req.height,
            steps=result["steps"],
            cfg=result["cfg"],
            sampler_name=result["sampler_name"],
            scheduler=result["scheduler"],
            timestamp_utc=now_utc,
            generation_time_seconds=result["elapsed_seconds"],
        )

        result_images = []
        for img in images:
            buf = BytesIO()
            img.save(buf, format="PNG")
            result_images.append(base64.b64encode(buf.getvalue()).decode())

        # 保存到磁盘（失败不影响请求响应）
        preview_path = None
        metadata_path = None
        try:
            preview_path, metadata_path = _save_result(images, metadata)
            _logger.info("Saved: preview=%s, metadata=%s", preview_path, metadata_path)
        except Exception as disk_err:
            _logger.warning("Failed to save to disk (generation itself succeeded): %s", disk_err)

        total_elapsed = round(time.time() - t_req_start, 2)
        _logger.info(
            "Generation completed: prompt_id=%s, seed=%d, gen_time=%.2fs, total_time=%.2fs",
            result["prompt_id"], result["seed"],
            result["elapsed_seconds"], total_elapsed,
        )

        return {
            "success": True,
            "images": result_images,
            "metadata": metadata.model_dump(),
            "preview_path": preview_path,
            "metadata_path": metadata_path,
        }

    except HTTPException:
        raise
    except Exception as e:
        total_elapsed = round(time.time() - t_req_start, 2)
        _logger.error(
            "Generation failed after %.2fs: %s | request text=%r",
            total_elapsed, str(e), req.text,
        )
        raise HTTPException(status_code=500, detail=str(e))


# ======================= 预处理端点 =======================

class PreprocessRequest(BaseModel):
    """预处理请求"""
    image_b64: str | None = None                           # base64 编码图像（与 image_path 二选一）
    image_path: str | None = None                          # 服务器端文件路径（与 image_b64 二选一）
    aspect_ratio: str = "1:1"                              # "1:1" / "16:9" / "3:2" / "9:16"
    target_width: int = 1024
    target_height: int = 1024
    resize_mode: str = "fit"                               # "fit" / "fill" / "stretch"
    remove_background: bool = True
    edge_denoise: bool = True
    subject_crop: bool = True
    crop_padding: int = 16
    color_quantize: bool = False
    quantize_colors: int = 256
    anti_alias: bool = True
    output_format: str = "png_rgba"                        # "png_rgba" / "png_rgb" / "webp_rgba"


class PreprocessResponse(BaseModel):
    success: bool
    image_b64: str | None = None                           # 处理后图像的 base64
    original_size: tuple[int, int] | None = None
    output_size: tuple[int, int] | None = None
    bbox: tuple[int, int, int, int] | None = None         # 主体包围盒
    detail: str | None = None


@app.post("/preprocess")
def preprocess_image(req: PreprocessRequest):
    """对单张图像执行预处理流水线，返回带 Alpha 通道的结果"""
    try:
        # 加载图像
        if req.image_b64:
            img = Image.open(BytesIO(base64.b64decode(req.image_b64)))
        elif req.image_path:
            target = (_PROJECT_ROOT / req.image_path).resolve()
            if not str(target).startswith(str(_PROJECT_ROOT.resolve())):
                raise HTTPException(status_code=403, detail="Access denied")
            img = Image.open(target)
        else:
            raise HTTPException(status_code=400, detail="image_b64 or image_path required")

        # 构建配置
        ar_map = {"1:1": AspectRatioPreset.SQUARE, "16:9": AspectRatioPreset.WIDESCREEN,
                   "3:2": AspectRatioPreset.CLASSIC, "9:16": AspectRatioPreset.PORTRAIT}
        rm_map = {"fit": ResizeMode.FIT, "fill": ResizeMode.FILL, "stretch": ResizeMode.STRETCH}
        fmt_map = {"png_rgba": OutputFormat.PNG_RGBA, "png_rgb": OutputFormat.PNG_RGB, "webp_rgba": OutputFormat.WEBP_RGBA}

        config = PreprocessConfig(
            aspect_ratio=ar_map.get(req.aspect_ratio, AspectRatioPreset.SQUARE),
            target_width=req.target_width,
            target_height=req.target_height,
            resize_mode=rm_map.get(req.resize_mode, ResizeMode.FIT),
            remove_background=req.remove_background,
            edge_denoise=req.edge_denoise,
            subject_crop=req.subject_crop,
            crop_padding=req.crop_padding,
            color_quantize=req.color_quantize,
            quantize_colors=req.quantize_colors,
            anti_alias=req.anti_alias,
            output_format=fmt_map.get(req.output_format, OutputFormat.PNG_RGBA),
        )

        result = preprocessor.process(img, config)

        buf = BytesIO()
        fmt = "PNG" if config.output_format != OutputFormat.WEBP_RGBA else "WEBP"
        result.image.save(buf, format=fmt)
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        _logger.info("Preprocess done: %dx%d → %dx%d", *result.original_size, *result.output_size)

        return PreprocessResponse(
            success=True,
            image_b64=img_b64,
            original_size=result.original_size,
            output_size=result.output_size,
            bbox=result.bbox,
            detail=f"Preprocessed: {result.original_size} → {result.output_size}",
        )

    except HTTPException:
        raise
    except Exception as e:
        _logger.error("Preprocess failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ======================= 矢量化端点 =======================

class VectorizeRequest(BaseModel):
    """矢量化请求"""
    image_b64: str | None = None                           # base64 编码图像
    color_clusters: int = 8
    smooth_threshold: float = 1.2
    min_region_area: int = 16
    path_precision: float = 0.5
    preserve_gradient: bool = True
    preserve_shadow: bool = True
    embed_preview: bool = True                             # 是否在 SVG 中嵌入预览
    output_preview_png: bool = False                       # 是否额外返回回渲染 PNG


class VectorizeResponse(BaseModel):
    success: bool
    svg_string: str | None = None
    preview_b64: str | None = None                         # 回渲染 PNG base64（可选）
    total_paths: int = 0
    total_vertices: int = 0
    color_layer_count: int = 0
    region_type_counts: dict[str, int] | None = None
    warnings: list[str] | None = None
    detail: str | None = None


@app.post("/vectorize")
def vectorize_image(req: VectorizeRequest):
    """对艺术字 PNG/JPG 执行矢量化，返回 SVG 字符串"""
    try:
        if not req.image_b64:
            raise HTTPException(status_code=400, detail="image_b64 is required")
        img = Image.open(BytesIO(base64.b64decode(req.image_b64)))

        config = VectorizationConfig(
            color_clusters=req.color_clusters,
            smooth_threshold=req.smooth_threshold,
            min_region_area=req.min_region_area,
            path_precision=req.path_precision,
            preserve_gradient=req.preserve_gradient,
            preserve_shadow=req.preserve_shadow,
            embed_preview=req.embed_preview,
        )

        result = vector_converter.convert(img, config)

        preview_b64: str | None = None
        if req.output_preview_png and result.preview_image:
            buf = BytesIO()
            result.preview_image.save(buf, format="PNG")
            preview_b64 = base64.b64encode(buf.getvalue()).decode()

        _logger.info(
            "Vectorize done: %d layers, %d paths, %d vertices",
            len(result.color_layers), result.total_paths, result.total_vertices,
        )

        return VectorizeResponse(
            success=True,
            svg_string=result.svg_string,
            preview_b64=preview_b64,
            total_paths=result.total_paths,
            total_vertices=result.total_vertices,
            color_layer_count=len(result.color_layers),
            region_type_counts=result.region_type_counts,
            warnings=result.warnings if result.warnings else None,
            detail=f"{len(result.color_layers)} color layers, {result.total_paths} paths",
        )

    except HTTPException:
        raise
    except Exception as e:
        _logger.error("Vectorize failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

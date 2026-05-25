"""
图像预处理与透明背景输出模块（核心任务三）

职责：
  - 统一分辨率输出（默认 ≥1024×1024，支持 16:9 / 1:1 / 3:2 等比例）
  - 背景分离、边缘去噪、主体裁剪、颜色量化、抗锯齿保留
  - 输出带 Alpha 通道的背景透明 PNG
  - 批量任务自动建立输出目录并按规则命名

注意：本文件仅定义接口与数据模型，算法实现留空。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageFilter

_logger = logging.getLogger("vecrafter.image_preprocessor")

try:
    from PIL import ImageResampling
    RESAMPLE_MODE = ImageResampling.LANCZOS
except ImportError:
    # 兼容旧版 PIL
    RESAMPLE_MODE = Image.LANCZOS if hasattr(Image, "LANCZOS") else Image.ANTIALIAS


# ======================= 枚举 / 常量 =======================

class AspectRatioPreset(Enum):
    """预设宽高比"""
    SQUARE = "1:1"       # 1024×1024
    WIDESCREEN = "16:9"  # 1024×576 或更大
    CLASSIC = "3:2"      # 1024×683
    PORTRAIT = "9:16"    # 576×1024
    CUSTOM = "custom"    # 由调用方直接指定宽高


class OutputFormat(Enum):
    """输出格式"""
    PNG_RGBA = "png_rgba"          # 带 Alpha 通道 PNG
    PNG_RGB = "png_rgb"            # 无 Alpha 通道（回退用）
    WEBP_RGBA = "webp_rgba"        # 带透明通道 WebP


class ResizeMode(Enum):
    """缩放模式"""
    FIT = "fit"              # 等比缩放，不足处填透明
    FILL = "fill"            # 等比缩放，超出部分居中裁剪
    STRETCH = "stretch"      # 直接拉伸到目标尺寸（不推荐）


# ======================= 数据模型 =======================

@dataclass
class PreprocessConfig:
    """预处理参数配置"""
    # --- 分辨率 ---
    aspect_ratio: AspectRatioPreset = AspectRatioPreset.SQUARE
    target_width: int = 1024
    target_height: int = 1024
    resize_mode: ResizeMode = ResizeMode.FIT
    min_resolution: int = 1024  # 短边最低像素

    # --- 背景处理 ---
    remove_background: bool = True
    bg_tolerance: float = 0.05         # 背景色容差 (0–1)
    bg_edge_feather: int = 2           # 边缘羽化像素

    # --- 边缘去噪 ---
    edge_denoise: bool = True
    edge_denoise_radius: int = 2       # 去噪核半径
    edge_smooth_iterations: int = 1    # 边缘平滑迭代次数

    # --- 主体裁剪 ---
    subject_crop: bool = True
    crop_padding: int = 16             # 裁剪后四周留白像素

    # --- 颜色量化 ---
    color_quantize: bool = False
    quantize_colors: int = 256         # 量化后颜色数

    # --- 抗锯齿 ---
    anti_alias: bool = True
    aa_scale: int = 2                  # 超采样倍数（2x = 先放大再缩回）

    # --- 输出 ---
    output_format: OutputFormat = OutputFormat.PNG_RGBA
    output_quality: int = 95           # 仅对有损格式生效

    # --- 后处理 ---
    trim_transparent_border: bool = False  # 裁掉全透明边缘（与 FIT 模式互斥，FIT 时不建议启用）
    ensure_opaque_foreground: bool = True  # 保证前景主体完全不透明

    def __post_init__(self) -> None:
        """参数合法性校验，不合法的值直接抛 ValueError 阻止后续执行"""
        errors: list[str] = []

        # ---- 分辨率 ----
        if self.target_width < 1:
            errors.append(f"target_width must be >= 1, got {self.target_width}")
        if self.target_height < 1:
            errors.append(f"target_height must be >= 1, got {self.target_height}")
        if self.min_resolution < 1:
            errors.append(f"min_resolution must be >= 1, got {self.min_resolution}")

        # ---- 背景处理 ----
        if not (0.0 <= self.bg_tolerance <= 1.0):
            errors.append(f"bg_tolerance must be in [0.0, 1.0], got {self.bg_tolerance}")
        if self.bg_edge_feather < 0:
            errors.append(f"bg_edge_feather must be >= 0, got {self.bg_edge_feather}")

        # ---- 边缘去噪 ----
        if self.edge_denoise:
            if self.edge_denoise_radius < 1:
                errors.append(f"edge_denoise_radius must be >= 1 when edge_denoise=True, got {self.edge_denoise_radius}")
            if self.edge_smooth_iterations < 1:
                errors.append(f"edge_smooth_iterations must be >= 1 when edge_denoise=True, got {self.edge_smooth_iterations}")

        # ---- 主体裁剪 ----
        if self.crop_padding < 0:
            errors.append(f"crop_padding must be >= 0, got {self.crop_padding}")

        # ---- 颜色量化 ----
        if not (2 <= self.quantize_colors <= 256):
            errors.append(f"quantize_colors must be in [2, 256], got {self.quantize_colors}")

        # ---- 抗锯齿 ----
        if self.anti_alias:
            if not (1 <= self.aa_scale <= 8):
                errors.append(f"aa_scale must be in [1, 8] when anti_alias=True, got {self.aa_scale}")

        # ---- 输出 ----
        if not (1 <= self.output_quality <= 100):
            errors.append(f"output_quality must be in [1, 100], got {self.output_quality}")

        # ---- 互斥 / 冲突检测 ----
        if self.trim_transparent_border and self.resize_mode == ResizeMode.FIT:
            _logger.warning(
                "trim_transparent_border=True conflicts with resize_mode=FIT: "
                "FIT padding will be stripped by post-process. "
                "Consider setting trim_transparent_border=False or using FILL/STRETCH mode."
            )

        if errors:
            raise ValueError("PreprocessConfig validation failed:" + " ".join(errors))


@dataclass
class PreprocessResult:
    """单张图像预处理结果"""
    image: Image.Image                     # 处理后的 RGBA PIL Image
    original_size: Tuple[int, int]         # 原始宽高
    output_size: Tuple[int, int]           # 输出宽高
    bbox: Optional[Tuple[int, int, int, int]] = None  # 主体包围盒 (x, y, w, h)
    foreground_pixel_count: int = 0        # 前景像素数（alpha > 0，用于质量评估）
    config_snapshot: PreprocessConfig = field(default_factory=PreprocessConfig)


@dataclass
class BatchPreprocessResult:
    """批量预处理结果"""
    items: List[PreprocessResult] = field(default_factory=list)
    total_input: int = 0
    total_success: int = 0
    output_dir: Optional[Path] = None
    errors: List[str] = field(default_factory=list)


# ======================= 预处理器类 =======================

class ImagePreprocessor:
    """
    图像预处理引擎

    使用示例::

        config = PreprocessConfig(
            aspect_ratio=AspectRatioPreset.SQUARE,
            target_width=1024,
            target_height=1024,
            remove_background=True,
            edge_denoise=True,
        )
        preprocessor = ImagePreprocessor()
        result = preprocessor.process(pil_image, config)
        result.image.save("output.png", format="PNG")
    """

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def process(
        self,
        image: Image.Image,
        config: Optional[PreprocessConfig] = None,
    ) -> PreprocessResult:
        """
        对单张图像执行完整预处理流水线

        Args:
            image: 输入 PIL.Image（支持 RGB / RGBA 模式）
            config: 预处理配置，为 None 时使用默认值

        Returns:
            PreprocessResult，其 .image 为 RGBA 模式的 PIL.Image
        """
        cfg = config or PreprocessConfig()
        original_size = (image.width, image.height)

        _logger.info(
            "Preprocess start: mode=%s size=%dx%d target=%dx%d aspect=%s",
            image.mode, image.width, image.height,
            cfg.target_width, cfg.target_height, cfg.aspect_ratio.value,
        )

        # ---- 流水线（顺序很重要！）----
        # 1. _ensure_rgba          → 统一到 RGBA
        # 2. _remove_background    → 背景分离（rembg / 色度键）
        # 3. _subject_crop         → 先裁剪主体区域（避免后续对大图做无谓计算）
        # 4. _edge_denoise         → 在裁剪后的小图上做二值掩码形态学清理
        #                               注意：此步只做开/闭运算清理噪点，不产生半透明边缘
        #                               边缘柔化由 _anti_alias 超采样处理
        # 5. _resize_to_target     → 分辨率统一（FIT / FILL / STRETCH）
        # 6. _color_quantize       → 颜色量化（中值切分 / K-Means）
        # 7. _anti_alias           → 超采样抗锯齿（产生平滑 RGB + 自然半透明边缘）
        # 8. _post_process         → 后处理（裁透明边 / 前景不透明度）
        #                               其中 ensure_opaque 将收紧 Alpha 利于后续矢量化

        result = self._ensure_rgba(image)
        _logger.debug("Step 1/8 ensure_rgba: %s → %s", image.mode, result.mode)

        if cfg.remove_background:
            result = self._remove_background(result, cfg)
            _logger.debug("Step 2/8 remove_background: %dx%d", result.width, result.height)

        if cfg.subject_crop:
            result = self._subject_crop(result, cfg)
            _logger.debug("Step 3/8 subject_crop: %dx%d", result.width, result.height)

        if cfg.edge_denoise:
            result = self._edge_denoise(result, cfg)
            _logger.debug("Step 4/8 edge_denoise: %dx%d", result.width, result.height)

        # 计算前景像素数（用于质量评估，在 resize 之前统计）
        foreground_count = 0
        if result.mode == "RGBA":
            alpha_arr = np.array(result.getchannel("A"), dtype=np.uint8)
            foreground_count = int(np.sum(alpha_arr > 0))

        result = self._resize_to_target(result, cfg)
        _logger.debug("Step 5/8 resize_to_target: %dx%d", result.width, result.height)

        if cfg.color_quantize:
            result = self._color_quantize(result, cfg)
            _logger.debug("Step 6/8 color_quantize: %dx%d", result.width, result.height)

        if cfg.anti_alias:
            result = self._anti_alias(result, cfg)
            _logger.debug("Step 7/8 anti_alias: %dx%d", result.width, result.height)

        result = self._post_process(result, cfg)
        _logger.debug("Step 8/8 post_process: %dx%d", result.width, result.height)

        _logger.info("Preprocess done: %dx%d → %dx%d",
                     original_size[0], original_size[1],
                     result.width, result.height)

        return PreprocessResult(
            image=result,
            original_size=original_size,
            output_size=(result.width, result.height),
            bbox=self._compute_alpha_bbox(result),
            foreground_pixel_count=foreground_count,
            config_snapshot=cfg,
        )

    def batch_process(
        self,
        images: List[Image.Image],
        config: Optional[PreprocessConfig] = None,
        output_dir: Optional[Union[str, Path]] = None,
        task_ids: Optional[List[str]] = None,
        prompt_slugs: Optional[List[str]] = None,
        seeds: Optional[List[int]] = None,
    ) -> BatchPreprocessResult:
        """
        批量预处理，自动建立输出目录并按规则命名

        命名规则:
            {output_dir}/{task_id}_{prompt_slug}_{seed}.png

        Args:
            images: 输入图像列表
            config: 预处理配置（所有图像共用同一配置）
            output_dir: 输出根目录，为 None 时不落盘
            task_ids: 任务编号列表，长度需与 images 一致
            prompt_slugs: 提示词摘要列表，长度需与 images 一致
            seeds: 随机种子列表，长度需与 images 一致

        Returns:
            BatchPreprocessResult
        """
        cfg = config or PreprocessConfig()
        n = len(images)

        task_ids = task_ids or [f"{i:04d}" for i in range(n)]
        prompt_slugs = prompt_slugs or ["untitled"] * n
        seeds = seeds or [0] * n

        # 校验辅助列表长度，防止 zip 静默截断导致命名错乱
        if len(task_ids) != n:
            _logger.warning("task_ids length %d != images length %d, padding with defaults", len(task_ids), n)
            task_ids = (list(task_ids) + [f"{i:04d}" for i in range(len(task_ids), n)])[:n]
        if len(prompt_slugs) != n:
            _logger.warning("prompt_slugs length %d != images length %d, padding with defaults", len(prompt_slugs), n)
            prompt_slugs = (list(prompt_slugs) + ["untitled"] * (n - len(prompt_slugs)))[:n]
        if len(seeds) != n:
            _logger.warning("seeds length %d != images length %d, padding with defaults", len(seeds), n)
            seeds = (list(seeds) + [0] * (n - len(seeds)))[:n]

        out_path = Path(output_dir) if output_dir else None
        if out_path:
            out_path.mkdir(parents=True, exist_ok=True)

        result = BatchPreprocessResult(
            total_input=n,
            output_dir=out_path,
        )

        for idx, (img, tid, slug, seed) in enumerate(
            zip(images, task_ids, prompt_slugs, seeds)
        ):
            try:
                item = self.process(img, cfg)

                if out_path:
                    filename = f"{tid}_{slug}_{seed}.png"
                    self.save_result(item, out_path, filename)

                result.items.append(item)
                result.total_success += 1
            except Exception as exc:
                result.errors.append(f"[{idx}] {tid}: {exc}")
                _logger.error("Batch preprocess failed for item %d (%s): %s", idx, tid, exc)

        _logger.info(
            "Batch preprocess done: %d/%d success",
            result.total_success, result.total_input,
        )
        return result

    def save_result(
        self,
        result: PreprocessResult,
        output_dir: Union[str, Path],
        filename: str,
    ) -> Path:
        """
        将预处理结果保存为 PNG 文件

        Args:
            result: PreprocessResult
            output_dir: 输出目录
            filename: 文件名（不含路径）

        Returns:
            已保存文件的 Path
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        filepath = out_dir / filename

        # 根据配置选择输出格式和编码器
        fmt_config = result.config_snapshot.output_format
        if fmt_config == OutputFormat.WEBP_RGBA:
            result.image.save(filepath, format="WEBP", quality=result.config_snapshot.output_quality)
        elif fmt_config == OutputFormat.PNG_RGB:
            result.image.convert("RGB").save(filepath, format="PNG")
        else:
            result.image.save(filepath, format="PNG")

        _logger.info("Saved preprocessed image to %s", filepath)
        return filepath

    def compute_resolution(
        self,
        aspect_ratio: AspectRatioPreset,
        min_resolution: int = 1024,
    ) -> Tuple[int, int]:
        """
        根据宽高比预设计算目标分辨率

        Args:
            aspect_ratio: 宽高比预设
            min_resolution: 短边最低像素

        Returns:
            (width, height)
        """
        mapping = {
            AspectRatioPreset.SQUARE:    (min_resolution, min_resolution),
            AspectRatioPreset.WIDESCREEN: (max(min_resolution, 1024), max(min_resolution * 9 // 16, 576)),
            AspectRatioPreset.CLASSIC:   (max(min_resolution, 1024), max(min_resolution * 2 // 3, 683)),
            AspectRatioPreset.PORTRAIT:  (max(min_resolution * 9 // 16, 576), max(min_resolution, 1024)),
        }
        return mapping.get(aspect_ratio, (min_resolution, min_resolution))

    # ------------------------------------------------------------------
    # 私有方法桩（算法待实现）
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_alpha_bbox(image: Image.Image) -> Optional[Tuple[int, int, int, int]]:
        """计算 Alpha 通道非零区域的包围盒 (x, y, w, h)"""
        if image.mode != "RGBA":
            return None

        bbox = image.getchannel("A").getbbox()
        if bbox is None:
            return None
        left, upper, right, lower = bbox
        # PIL getbbox 返回 (left, upper, right, lower)，转换为 (x, y, w, h)
        return (left, upper, right - left, lower - upper)

    # 以下方法仅声明签名，作为流水线步骤的占位说明

    def _ensure_rgba(self, image: Image.Image) -> Image.Image:
        """统一图像模式到 RGBA"""
        if image.mode == "RGBA":
            return image
        return image.convert("RGBA")

    def _remove_background(self, image: Image.Image, cfg: PreprocessConfig) -> Image.Image:
        """
        背景分离

        策略：
          1. 优先使用 rembg 深度学习模型（效果最好，能识别任意背景）
          2. 若 rembg 不可用，回退到色度键（chroma-key）方法：
             检测图像四角的纯色区域作为背景色，将近似颜色的像素变透明

        Args:
            image: RGBA 模式的 PIL.Image
            cfg: 预处理配置

        Returns:
            背景已透明的 RGBA PIL.Image
        """
        # ---------- 策略 1: rembg ----------
        try:
            from rembg import remove
            _logger.info("Using rembg for background removal")
            # 确保rembg返回RGBA格式
            result = remove(image).convert("RGBA")
            if cfg.bg_edge_feather > 0:
                result = self._feather_alpha_edge(result, cfg.bg_edge_feather)
            _logger.info("rembg background removal done")
            return result
        except ImportError:
            _logger.warning("rembg not installed, falling back to chroma-key")
        except RuntimeError as e:
            _logger.warning(f"rembg runtime error (CUDA out of memory?): {e}, falling back to chroma-key")
        except Exception as e:
            _logger.warning(f"rembg failed: {e}, falling back to chroma-key")
    
        return self._chroma_key_remove(image, cfg)

    # ------------------------------------------------------------------
    # 内部辅助：色度键背景去除
    # ------------------------------------------------------------------

    def _chroma_key_remove(self, image: Image.Image, cfg: PreprocessConfig) -> Image.Image:
        """
        色度键（chroma-key）背景去除

        原理：
          1. 采样图像四角的像素，计算平均背景色
          2. 对每个像素，计算其与背景色的欧氏距离
          3. 距离小于阈值 * 背景亮度范围的像素 → 设为透明
        """
        if image.mode != "RGBA":
            image = image.convert("RGBA")

        arr = np.array(image, dtype=np.float32)
        h, w = arr.shape[:2]

        # 采样四角区域（每角取 5x5 或更小）来估计背景色
        corner_size = min(5, w // 4, h // 4)
        if corner_size < 2:
            # 图像过小（宽或高 < 8 像素），四角采样不可靠，跳过 chroma-key
            _logger.warning("Image too small (%dx%d) for chroma-key, returning unchanged", w, h)
            return image.convert("RGBA") if image.mode != "RGBA" else image

        corners = [
            arr[:corner_size, :corner_size, :3],          # 左上
            arr[:corner_size, -corner_size:, :3],         # 右上
            arr[-corner_size:, :corner_size, :3],         # 左下
            arr[-corner_size:, -corner_size:, :3],        # 右下
        ]
        bg_pixels = np.concatenate([c.reshape(-1, 3) for c in corners], axis=0)
        bg_color = bg_pixels.mean(axis=0)  # 平均背景色 (R, G, B)

        # 计算背景色的亮度范围用于自适应阈值
        bg_brightness = np.linalg.norm(bg_color)
        # tolerance 值越大 → 阈值越宽松 → 更多像素被移除
        threshold = cfg.bg_tolerance * (bg_brightness + 1.0) * 3.0
        threshold = max(1e-6, threshold)  # 最小阈值，防止除零

        # 计算每个像素与背景色的欧氏距离
        pixel_diff = arr[:, :, :3] - bg_color[None, None, :]
        distances = np.sqrt(np.sum(pixel_diff ** 2, axis=2))

        # 距离小于阈值的像素 → Alpha 设为 0（透明）
        is_background = distances < threshold
        arr[is_background, 3] = 0

        # 对半透明区域也渐变处理
        transition_zone = (distances >= threshold) & (distances < threshold * 1.5)
        if transition_zone.any():
            # 过渡区按距离做渐变透明度
            alpha_denominator = max(1e-6, threshold * 0.5)
            alpha_factor = (distances - threshold) / alpha_denominator
            alpha_factor = np.clip(alpha_factor, 0.0, 1.0)
            arr[transition_zone, 3] = arr[transition_zone, 3] * alpha_factor[transition_zone]

        result = Image.fromarray(arr.clip(0, 255).astype(np.uint8), mode="RGBA")

        # 羽化边缘
        if cfg.bg_edge_feather > 0:
            result = self._feather_alpha_edge(result, cfg.bg_edge_feather)

        _logger.info("Chroma-key background removal done (bg_color=%.0f,%.0f,%.0f threshold=%.1f)",
                      bg_color[0], bg_color[1], bg_color[2], threshold)
        return result

    # ------------------------------------------------------------------
    # 内部辅助：Alpha 边缘羽化
    # ------------------------------------------------------------------

    def _feather_alpha_edge(self, image: Image.Image, radius: int) -> Image.Image:
        """
        对 Alpha 通道做高斯模糊，柔化前景与透明区域的过渡

        Args:
            image: RGBA 图像
            radius: 高斯模糊半径（像素）

        Returns:
            羽化后的 RGBA 图像
        """
        if radius <= 0 or image.mode != "RGBA":
            return image

        # 只对 Alpha 通道做高斯模糊，RGB 保持不变
        r, g, b, a = image.split()
        a_blurred = a.filter(ImageFilter.GaussianBlur(radius))
        return Image.merge("RGBA", (r, g, b, a_blurred))

    def _edge_denoise(self, image: Image.Image, cfg: PreprocessConfig) -> Image.Image:
        """
        边缘去噪（纯二值掩码清理，不产生半透明边缘）

        策略：
          1. 将 Alpha 通道二值化为前景掩码（>=128 → 255，其余 → 0）
          2. 对二值掩码先开运算（去孤立噪点）再闭运算（填内部空洞）
          3. 用清理后的二值掩码替换原 Alpha 通道

        注意：
          本方法只做二值清理，不引入半透明边缘过滤。
          边缘柔化由 _anti_alias 的超采样处理完成，
          _post_process(ensure_opaque) 作为最后一步会收紧 Alpha。

        Args:
            image: RGBA 模式的 PIL.Image
            cfg: 预处理配置

        Returns:
            去噪后的 RGBA PIL.Image（Alpha 为 0 或 255 二值）
        """
        if not cfg.edge_denoise or image.mode != "RGBA":
            return image

        # 提前返回：半径为 0 时不处理
        if cfg.edge_denoise_radius <= 0:
            return image

        try:
            import cv2
        except ImportError:
            _logger.warning("cv2 (opencv-python) not installed, skipping edge denoise")
            return image

        r, g, b, a = image.split()
        alpha_arr = np.array(a, dtype=np.uint8)

        # 二值化
        binary_mask = (alpha_arr >= 128).astype(np.uint8) * 255

        # OpenCV 形态学：椭圆核 + 先开后闭
        kernel_size = cfg.edge_denoise_radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        # 开运算：去孤立噪点
        clean_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel,
                                       iterations=cfg.edge_smooth_iterations)
        # 闭运算：填内部空洞
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # 用清理后的二值掩码替换 Alpha（纯 0/255，不引入渐变）
        new_alpha = np.where(clean_mask == 255, 255, 0).astype(np.uint8)
        new_a = Image.fromarray(new_alpha, mode="L")

        _logger.info("Edge denoise done: kernel=%d iterations=%d",
                     kernel_size, cfg.edge_smooth_iterations)
        return Image.merge("RGBA", (r, g, b, new_a))

    def _subject_crop(self, image: Image.Image, cfg: PreprocessConfig) -> Image.Image:
        """
        基于 Alpha 通道包围盒的主体裁剪

        找到 Alpha 通道非零像素的包围盒，加上四周留白后裁剪，
        去除多余的透明边缘，使主体紧凑。

        Args:
            image: RGBA 模式的 PIL.Image
            cfg: 预处理配置

        Returns:
            裁剪后的 RGBA PIL.Image（可能与原图等大）
        """
        if not cfg.subject_crop or image.mode != "RGBA":
            return image
    
        alpha = image.getchannel("A")
        bbox = alpha.getbbox()
        if bbox is None:
            _logger.warning("Subject crop skipped: fully transparent image")
            return image
    
        left, upper, right, lower = bbox
        w, h = image.size
        padding = cfg.crop_padding
    
        # 计算新坐标并确保有效性
        new_left = max(0, left - padding)
        new_upper = max(0, upper - padding)
        new_right = min(w, right + padding)
        new_lower = min(h, lower + padding)
    
        # 修复：防止裁剪区域无效
        if new_left >= new_right or new_upper >= new_lower:
            _logger.warning("Subject crop skipped: invalid crop region")
            return image
    
        _logger.info(f"Subject crop: {bbox} + padding={padding} → ({new_left},{new_upper},{new_right},{new_lower})")
        return image.crop((new_left, new_upper, new_right, new_lower))

    def _resize_to_target(self, image: Image.Image, cfg: PreprocessConfig) -> Image.Image:
        """
        分辨率统一

        三种模式：
          - FIT:   等比缩放至完全进入目标画布，不足处填透明
          - FILL:  等比缩放至完全覆盖目标画布，超出部分居中裁剪
          - STRETCH: 直接拉伸到目标尺寸（通常不推荐）

        Args:
            image: RGBA 模式的 PIL.Image
            cfg: 预处理配置

        Returns:
            缩放后的 RGBA PIL.Image
        """
        tw, th = cfg.target_width, cfg.target_height
        iw, ih = image.size
    
        # 参数校验
        if tw <= 0 or th <= 0:
            _logger.warning(f"Invalid target size {tw}x{th}, using original size")
            return image
    
        if (iw, ih) == (tw, th) and cfg.resize_mode != ResizeMode.STRETCH:
            return image
    
        if cfg.resize_mode == ResizeMode.STRETCH:
            result = image.resize((tw, th), RESAMPLE_MODE)
            _logger.info(f"Resize STRETCH: {iw}x{ih} → {tw}x{th}")
            return result
    
        scale_fit = min(tw / iw, th / ih)
        scale_fill = max(tw / iw, th / ih)
    
        if cfg.resize_mode == ResizeMode.FIT:
            scale = scale_fit
            # 使用round()而非int()，减少尺寸偏差
            new_w, new_h = round(iw * scale), round(ih * scale)
            resized = image.resize((new_w, new_h), RESAMPLE_MODE)
            canvas = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
            # 更精确的居中计算
            offset_x = (tw - new_w) // 2
            offset_y = (th - new_h) // 2
            canvas.paste(resized, (offset_x, offset_y), resized)
            _logger.info(f"Resize FIT: {iw}x{ih} → {new_w}x{new_h} → canvas {tw}x{th}")
            return canvas
    
        elif cfg.resize_mode == ResizeMode.FILL:
            scale = scale_fill
            new_w, new_h = round(iw * scale), round(ih * scale)
            resized = image.resize((new_w, new_h), RESAMPLE_MODE)
            left = (new_w - tw) // 2
            top = (new_h - th) // 2
            result = resized.crop((left, top, left + tw, top + th))
            _logger.info(f"Resize FILL: {iw}x{ih} → {new_w}x{new_h} → crop {tw}x{th}")
            return result
    
        _logger.warning(f"Unsupported resize mode: {cfg.resize_mode}")
        return image

    # ------------------------------------------------------------------
    # 内部辅助：超采样缩放
    # ------------------------------------------------------------------

    @staticmethod
    def _resize_supersample(image: Image.Image, target_size: Tuple[int, int], scale: int) -> Image.Image:
        """
        超采样缩放：先放大 scale 倍再缩回目标尺寸，消除锯齿

        Args:
            image: 源图
            target_size: 目标 (w, h)
            scale: 放大倍数（如 2）

        Returns:
            缩放后的图像
        """
        if scale <= 1:
            return image.copy()  # 返回副本，避免 caller 意外修改原图
        tw, th = target_size
        temp_w, temp_h = tw * scale, th * scale
        return image.resize((temp_w, temp_h), RESAMPLE_MODE).resize((tw, th), RESAMPLE_MODE)

    # ------------------------------------------------------------------
    # 内部辅助：颜色量化
    # ------------------------------------------------------------------

    def _color_quantize(self, image: Image.Image, cfg: PreprocessConfig) -> Image.Image:
        """
        颜色量化

        将图像颜色数减少到 cfg.quantize_colors，用于简化后续矢量化。
        采用 PIL 内置的中值切分算法。

        Args:
            image: RGBA 模式的 PIL.Image
            cfg: 预处理配置

        Returns:
            量化后的 RGBA PIL.Image
        """
        if not cfg.color_quantize or image.mode != "RGBA":
            return image
    
        colors = max(2, min(cfg.quantize_colors, 256))
        rgb = image.convert("RGB")
        alpha = image.getchannel("A")
    
        # 兼容旧版PIL：MEDIANCUT的数值是0
        try:
            quantized_p = rgb.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
        except AttributeError:
            quantized_p = rgb.quantize(colors=colors, method=0)
    
        quantized_rgb = quantized_p.convert("RGB")
        result = Image.merge("RGBA", (*quantized_rgb.split(), alpha))
        _logger.info(f"Color quantize: {colors} colors")
        return result

    # ------------------------------------------------------------------
    # 内部辅助：抗锯齿
    # ------------------------------------------------------------------

    def _anti_alias(self, image: Image.Image, cfg: PreprocessConfig) -> Image.Image:
        """
        抗锯齿处理

        超采样策略：先放大 aa_scale 倍再 LANCZOS 缩回原尺寸。
        适用于在 resize/subject_crop 之后消除硬边缘锯齿。

        Args:
            image: RGBA 模式的 PIL.Image
            cfg: 预处理配置

        Returns:
            抗锯齿处理后的 RGBA PIL.Image（尺寸不变）
        """
        if not cfg.anti_alias or image.mode != "RGBA":
            return image
    
        scale = cfg.aa_scale  # 直接使用用户配置，不强制最小为2
        if scale <= 1:
            return image
    
        w, h = image.size
        result = self._resize_supersample(image, (w, h), scale)
        _logger.info(f"Anti-alias: {w}x{h} supersampled {scale}x")
        return result

    # ------------------------------------------------------------------
    # 内部辅助：后处理
    # ------------------------------------------------------------------

    def _post_process(self, image: Image.Image, cfg: PreprocessConfig) -> Image.Image:
        """
        后处理

        两步（按需执行）：
          1. trim_transparent_border: 裁掉四周全透明的边缘
          2. ensure_opaque_foreground: 将前景不透明区域的 Alpha 强制设为 255

        Args:
            image: RGBA 模式的 PIL.Image
            cfg: 预处理配置

        Returns:
            后处理后的 RGBA PIL.Image
        """
        if image.mode != "RGBA":
            return image
    
        result = image
    
        # 1. 裁掉四周全透明边缘（当用户显式启用且 _subject_crop 已被跳过时仍有意义）
        if cfg.trim_transparent_border:
            alpha = result.getchannel("A")
            bbox = alpha.getbbox()
            if bbox is not None and bbox != (0, 0, result.width, result.height):
                result = result.crop(bbox)
                _logger.info("Post-process: trimmed transparent border, %dx%d → %dx%d",
                             image.width, image.height, result.width, result.height)

        # 2. 将前景 Alpha 收紧为完全不透明（利于后续矢量化）
        if cfg.ensure_opaque_foreground:
            alpha = np.array(result.getchannel("A"), dtype=np.uint8)
            if np.any(alpha >= 128):
                arr = np.array(result, dtype=np.uint8)
                arr[:, :, 3] = np.where(arr[:, :, 3] >= 128, 255, arr[:, :, 3])
                result = Image.fromarray(arr, mode="RGBA")
                _logger.info("Post-process: foreground alpha clamped to 255")
    
        return result

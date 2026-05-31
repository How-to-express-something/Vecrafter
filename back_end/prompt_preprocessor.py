"""
Prompt 预处理器

职责：
  - 将用户输入的文字自动包装为 艺术字"xxx" 格式
  - 根据文本类型（中文/英文/数字/混合）选择最优包装策略
  - 根据风格预设（国风书法、海洋浪漫、促销卡通）自动增强 prompt
  - 参考 CFG_test.json 中 LoRA 栈特性，针对性添加触发词
  - 按文本长度推荐 CFG Scale
  - 按文本类型附加针对性负面提示词
  - 提高 ComfyUI 生成艺术字的准确率
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Dict, Optional, Tuple

_logger = logging.getLogger("vecrafter.backend")


# ======================= 文本类型检测 =======================

class TextCategory(Enum):
    CHINESE = "chinese"
    ENGLISH = "english"
    DIGIT = "digit"
    CHINESE_ENGLISH = "chinese_english"
    CHINESE_DIGIT = "chinese_digit"
    ENGLISH_DIGIT = "english_digit"
    MIXED = "mixed"


def _classify_text(text: str) -> TextCategory:
    """检测文本的语言/字符组成类别"""
    has_chinese = any('一' <= c <= '鿿' for c in text)
    has_english = any(c.isascii() and c.isalpha() for c in text)
    has_digit = any(c.isdigit() for c in text)

    if has_chinese and has_english and has_digit:
        return TextCategory.MIXED
    if has_chinese and has_english:
        return TextCategory.CHINESE_ENGLISH
    if has_chinese and has_digit:
        return TextCategory.CHINESE_DIGIT
    if has_english and has_digit:
        return TextCategory.ENGLISH_DIGIT
    if has_chinese:
        return TextCategory.CHINESE
    if has_english:
        return TextCategory.ENGLISH
    if has_digit:
        return TextCategory.DIGIT
    return TextCategory.MIXED


# ======================= 文本类型感知的包装策略 =======================

# 不同文本类型使用不同的包装格式，提高 SD 模型的文字生成准确率
_TEXT_WRAP_TEMPLATES: Dict[TextCategory, str] = {
    TextCategory.CHINESE:           '艺术字"{text}"',
    TextCategory.ENGLISH:           'Typography art "{text}", text-based logo design',
    TextCategory.DIGIT:             '促销艺术字设计，数字"{text}"，数字笔画完整清晰',
    TextCategory.CHINESE_ENGLISH:   '艺术字设计，文字内容为："{text}"，中英文混排，文字清晰可读',
    TextCategory.CHINESE_DIGIT:     '艺术字设计，文字内容为："{text}"，包含汉字与数字',
    TextCategory.ENGLISH_DIGIT:     'Typography art "{text}", alphanumeric design, clean readable text',
    TextCategory.MIXED:             '艺术字设计，文字内容包含汉字、英文和数字："{text}"',
}

# 长文本（>4字中文或>8字符）的额外约束
_LONG_TEXT_CONSTRAINT = "，文字笔画完整，结构清晰，无缺笔少画"
_SHORT_TEXT_CONSTRAINT = "，字体端正，笔画清晰"


# ======================= 文本类型感知的负面提示词 =======================

_TEXT_NEGATIVE: Dict[TextCategory, str] = {
    TextCategory.CHINESE:         "错误汉字，错别字，多余笔画，缺少笔画，汉字乱码，文字结构错误，偏旁错误",
    TextCategory.ENGLISH:         "misspelling, wrong letters, extra letters, missing letters, garbled text, weird characters",
    TextCategory.DIGIT:           "wrong number, missing digit, extra digit, broken number, number deformation, 拼写错误",
    TextCategory.CHINESE_ENGLISH: "中英文混排错误，对齐错误，中文字体与英文字体不协调，错别字，拼写错误",
    TextCategory.CHINESE_DIGIT:   "汉字与数字混排错误，数字变形，错别字，笔画残缺",
    TextCategory.ENGLISH_DIGIT:   "alphanumeric misalignment, wrong letters, wrong numbers, missing characters",
    TextCategory.MIXED:           "混排错误，文字重叠，文字混乱，乱码，错别字，拼写错误，数字错误",
}


# ======================= 风格预设增强库 =======================

# 根据 CFG_test.json 中加载的 LoRA 栈特性：
#   - 新中式字体 LoRA（青争Qwen）
#   - SDXLRonghua_v45（绒花风格）
#   - 毛笔字手写艺术字生成模型 V1
#   - Harrlogos_v2.0（Logo 风格）
# 不同风格预设对应不同的增强后缀

_STYLE_ENHANCE: Dict[str, dict] = {
    "国风书法": {
        "keywords": ["水墨", "书法", "墨色", "笔触"],
        "suffix": (
            "水墨风格书法艺术字，墨色浓淡相宜，毛笔笔触自然，"
            "传统宣纸纹理背景，印章点缀，留白构图，古朴典雅，"
            "金色或黑色墨迹，行书或楷书风格，高清矢量质感"
        ),
        "lora_trigger": "新中式字体，毛笔字手写艺术字",
    },
    "海洋浪漫": {
        "keywords": ["海洋", "海浪", "贝壳", "渐变"],
        "suffix": (
            "海洋主题艺术字设计，蓝青碧波渐变色彩，"
            "水波纹光影效果，贝壳海星珊瑚装饰元素，"
            "清新通透质感，气泡点缀，梦幻浪漫氛围，"
            "字体圆润流畅，高清渲染，矢量风格"
        ),
        "lora_trigger": "艺术字设计，装饰字体",
    },
    "促销卡通": {
        "keywords": ["促销", "卡通", "描边", "横幅"],
        "suffix": (
            "促销艺术字设计，卡通可爱风格，粗描边醒目，"
            "粉紫橙黄明快配色，弧形横幅飘带装饰，"
            "立体阴影效果，视觉冲击力强，标题字体突出，"
            "适合海报 banner，矢量风格，高清"
        ),
        "lora_trigger": "Logo设计，促销字体，Harrlogos",
    },
}

# 通用增强（所有风格都会带上）
_COMMON_ENHANCE = (
    "商业级艺术字设计，"
    "文字清晰可辨，笔画完整无缺失，"
    "主体居中，边缘干净，透明背景"
)

# 负面 prompt 增强
_NEGATIVE_ENHANCE = (
    "文字模糊，文字变形，笔画残缺，多余文字，"
    "拼写错误，乱码，重影，锯齿边缘，"
    "低质量，失真，扭曲，混乱背景，水印"
)


class PromptPreprocessor:
    """
    Prompt 预处理器

    使用示例::

        pp = PromptPreprocessor()
        positive = pp.build_positive("青山集", "国风书法，墨色渐变")
        # → 自动感知文本类型，匹配中文短文本包装策略
        negative = pp.build_negative("blurry, low quality", "青山集")
        # → blurry, low quality + 中文负面词 + 通用负面词
        cfg = pp.recommend_cfg("Hello World 2024")
        # → 1.3（中等长度混排文本）
    """

    @staticmethod
    def build_positive(text: str, style_prompt: str = "") -> str:
        """
        构建完整的 positive prompt（文本类型感知版）

        格式: <text_type_aware_wrap>，<style_prompt>，<style_enhance>，<text_length_constraint>，<common_enhance>

        Args:
            text: 用户输入的文字内容
            style_prompt: 风格提示词（来自前端风格预设或自定义）

        Returns:
            加工后的 positive prompt 字符串
        """
        category = _classify_text(text)
        parts: list[str] = []

        # 1. 文本类型感知的核心包装
        template = _TEXT_WRAP_TEMPLATES.get(category, '艺术字"{text}"')
        parts.append(template.format(text=text))

        # 2. 附加风格提示词
        if style_prompt and style_prompt != "默认艺术风格":
            parts.append(style_prompt)

        # 3. 根据风格关键词匹配增强
        enhanced = PromptPreprocessor._match_style_enhance(style_prompt)
        if enhanced:
            parts.append(enhanced)

        # 4. 按长度附加约束
        char_len = len(text)
        if char_len <= 4:
            parts.append(_SHORT_TEXT_CONSTRAINT.lstrip("，"))
        elif char_len > 8:
            parts.append(_LONG_TEXT_CONSTRAINT.lstrip("，"))

        # 5. 通用增强
        parts.append(_COMMON_ENHANCE)

        result = "，".join(parts)
        _logger.info(
            "Prompt built: text=%r category=%s style=%r → %s",
            text, category.value, style_prompt[:40] if style_prompt else "",
            result[:150],
        )
        return result

    @staticmethod
    def build_negative(base_negative: str = "", text: str = "") -> str:
        """
        构建完整的 negative prompt（文本类型感知版）

        Args:
            base_negative: 用户/系统默认的负面提示词
            text: 原始输入文字（用于匹配类型特定的负面词）

        Returns:
            增强后的 negative prompt
        """
        parts_list: list[str] = []

        if base_negative and base_negative != "模糊，杂乱背景，错误文字":
            parts_list.append(base_negative)

        # 追加通用负面增强
        parts_list.append(_NEGATIVE_ENHANCE)

        # 追加文本类型特定的负面词
        if text:
            category = _classify_text(text)
            text_specific = _TEXT_NEGATIVE.get(category)
            if text_specific:
                parts_list.append(text_specific)

        result = "，".join(parts_list)
        _logger.info("Negative prompt built (%d chars, text_category=%s)", len(result), _classify_text(text).value if text else "none")
        return result

    @staticmethod
    def recommend_cfg(text: str) -> float:
        """
        根据文本长度推荐 CFG Scale

        原则：
          - 短文本（≤2字符）：低 CFG 保留风格自由度 → 0.9
          - 中等文本（3-4字符）：默认 → 1.1
          - 较长文本（5-8字符）：提高约束 → 1.3
          - 超长文本（>8字符）：最高约束 → 1.5

        Args:
            text: 输入文字

        Returns:
            推荐 CFG Scale 值
        """
        l = len(text)
        if l <= 2:
            return 0.9
        elif l <= 4:
            return 1.1
        elif l <= 8:
            return 1.3
        else:
            return 1.5

    @staticmethod
    def get_preset_enhance(style_prompt: str) -> Optional[str]:
        """获取风格预设的增强描述（供外部查询用）"""
        for preset_name, cfg in _STYLE_ENHANCE.items():
            if any(kw in style_prompt for kw in cfg["keywords"]):
                return cfg["suffix"]
        return None

    @staticmethod
    def _match_style_enhance(style_prompt: str) -> Optional[str]:
        """
        根据风格提示词匹配预设增强

        匹配逻辑：检测 style_prompt 中是否包含预设的关键词
        """
        if not style_prompt:
            return None

        best_match = None
        best_score = 0

        for preset_name, cfg in _STYLE_ENHANCE.items():
            score = sum(1 for kw in cfg["keywords"] if kw in style_prompt)
            if score > best_score:
                best_score = score
                best_match = cfg["suffix"]

        if best_match:
            trigger = f"（风格匹配: {preset_name}）"
            _logger.info("Style enhanced matched: %s", trigger)
            return best_match

        return None

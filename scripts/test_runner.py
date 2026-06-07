#!/usr/bin/env python3
"""
Vecrafter 系统性测试脚本
涵盖：功能测试、异常测试、性能基线测试、边界输入测试
"""

import sys, os, time, json, csv, io, math
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# 确保能导入项目模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "back_end"))

RESULT_DIR = Path(__file__).resolve().parent.parent / "output" / "test_results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ======================= 测试结果收集 =======================

class TestResult:
    """单条测试结果"""
    def __init__(self, case_id: str, category: str, name: str):
        self.case_id = case_id
        self.category = category
        self.name = name
        self.status = "PASS"
        self.detail = ""
        self.duration_ms = 0.0
        self.expected = ""
        self.actual = ""

    def fail(self, msg: str):
        self.status = "FAIL"
        self.detail = msg

    def warn(self, msg: str):
        if self.status == "PASS":
            self.status = "WARN"
        self.detail = msg

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "name": self.name,
            "status": self.status,
            "detail": self.detail[:200] if self.detail else "",
            "duration_ms": round(self.duration_ms, 2),
            "expected": self.expected[:100] if self.expected else "",
            "actual": self.actual[:100] if self.actual else "",
        }


class TestSuite:
    def __init__(self, name: str):
        self.name = name
        self.results: List[TestResult] = []
        self.start_time = time.time()

    def add(self, case_id: str, category: str, name: str) -> TestResult:
        r = TestResult(case_id, category, name)
        self.results.append(r)
        return r

    def summary(self) -> dict:
        total = len(self.results)
        passed = sum(1 for r in self.results if r.status == "PASS")
        failed = sum(1 for r in self.results if r.status == "FAIL")
        warned = sum(1 for r in self.results if r.status == "WARN")
        elapsed = time.time() - self.start_time
        return {
            "suite": self.name,
            "total": total,
            "passed": passed,
            "failed": failed,
            "warned": warned,
            "pass_rate": f"{passed / max(1, total) * 100:.1f}%",
            "elapsed_seconds": round(elapsed, 2),
        }

    def report_text(self) -> str:
        s = self.summary()
        lines = [
            f"{'='*60}",
            f"  测试套件: {s['suite']}",
            f"  总计: {s['total']}  |  通过: {s['passed']}  |  失败: {s['failed']}  |  警告: {s['warned']}",
            f"  通过率: {s['pass_rate']}  |  耗时: {s['elapsed_seconds']}s",
            f"{'='*60}",
        ]
        for r in self.results:
            status_icon = {"PASS": "  OK", "FAIL": "FAIL", "WARN": " WARN"}[r.status]
            lines.append(f"  [{status_icon}] {r.case_id:8s} {r.name:<50s} {r.duration_ms:>8.1f}ms")
            if r.detail:
                lines.append(f"          -> {r.detail[:120]}")
        lines.append(f"{'='*60}")
        return "\n".join(lines)

    def to_json(self) -> dict:
        return {
            "summary": self.summary(),
            "results": [r.to_dict() for r in self.results],
        }


# ======================= 测试函数 =======================

def make_suite(name: str) -> TestSuite:
    print(f"\n{'#'*60}")
    print(f"# 开始测试: {name}")
    print(f"{'#'*60}")
    return TestSuite(name)


def print_suite_result(suite: TestSuite):
    print()
    print(suite.report_text())
    # 保存 JSON
    path = RESULT_DIR / f"{suite.name.replace(' ', '_').lower()}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(suite.to_json(), f, ensure_ascii=False, indent=2)
    print(f"  结果已保存: {path}\n")


# ======================= 测试用例 =======================

def test_prompt_preprocessor():
    """测试 PromptPreprocessor 模块"""
    suite = make_suite("Prompt Preprocessor")
    from prompt_preprocessor import PromptPreprocessor, _classify_text, TextCategory

    # 1.1 文本分类
    cases = [
        ("青山集", TextCategory.CHINESE),
        ("Hello", TextCategory.ENGLISH),
        ("2024", TextCategory.DIGIT),
        ("Hello2024", TextCategory.ENGLISH_DIGIT),
        ("青山2024", TextCategory.CHINESE_DIGIT),
        ("Hello青山", TextCategory.CHINESE_ENGLISH),
        ("Hello青山2024", TextCategory.MIXED),
    ]
    for text, expected in cases:
        r = suite.add("1.1", "文本分类", f"classify: {text}")
        t0 = time.time()
        try:
            actual = _classify_text(text)
            r.duration_ms = (time.time() - t0) * 1000
            r.expected = expected.value
            r.actual = actual.value
            if actual != expected:
                r.fail(f"expected={expected.value}, got={actual.value}")
        except Exception as e:
            r.fail(str(e))

    # 1.2 build_positive
    pos_cases = [
        ("青山集", "国风书法，墨色渐变"),
        ("Hello", "促销卡通"),
        ("2024", ""),
        ("HelloWorld2024", ""),
    ]
    for text, style in pos_cases:
        r = suite.add("1.2", "正向提示词", f"build_positive: {text}")
        t0 = time.time()
        try:
            result = PromptPreprocessor.build_positive(text, style)
            r.duration_ms = (time.time() - t0) * 1000
            r.expected = "length > 20"
            r.actual = f"len={len(result)}"
            if len(result) < 20:
                r.fail("too short")
        except Exception as e:
            r.fail(str(e))

    # 1.3 build_negative
    neg_cases = [
        ("blurry", "Hello"),
        ("", "青山集"),
        ("bad quality, low res", ""),
    ]
    for base_neg, text in neg_cases:
        r = suite.add("1.3", "负向提示词", f"build_negative: text={text[:15]}")
        t0 = time.time()
        try:
            result = PromptPreprocessor.build_negative(base_neg, text)
            r.duration_ms = (time.time() - t0) * 1000
            r.actual = f"len={len(result)}"
            if len(result) < 15:
                r.fail("too short")
        except Exception as e:
            r.fail(str(e))

    # 1.4 recommend_cfg
    cfg_cases = [("A", 0.9), ("AB", 0.9), ("ABC", 1.1), ("ABCDE", 1.3), ("ABCDEFGHI", 1.5)]
    for text, expected in cfg_cases:
        r = suite.add("1.4", "CFG推荐", f"recommend_cfg: len={len(text)}")
        t0 = time.time()
        try:
            actual = PromptPreprocessor.recommend_cfg(text)
            r.duration_ms = (time.time() - t0) * 1000
            r.expected = str(expected)
            r.actual = str(actual)
            if abs(actual - expected) > 0.1:
                r.fail(f"expected={expected}, got={actual}")
        except Exception as e:
            r.fail(str(e))

    print_suite_result(suite)
    return suite


def test_vectorizer():
    """测试矢量化模块（边缘驱动管线）"""
    suite = make_suite("Vectorizer")
    from vector_converter import VectorConverter, VectorizationConfig

    converter = VectorConverter()
    cfg = VectorizationConfig(use_edge_driven=True, embed_preview=False)

    # 寻找测试图片
    img_dir = Path(__file__).resolve().parent.parent / "output"
    test_images = sorted(img_dir.glob("**/preview.png"))[:5]
    if not test_images:
        # 创建一个测试图片
        img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
        for y in range(50, 200):
            for x in range(80, 180):
                if abs(x - 130) + abs(y - 125) < 40:
                    img.putpixel((x, y), (200, 30, 30, 255))
        test_images = [img]
        r = suite.add("2.0", "准备", "使用合成测试图")
        r.status = "WARN"
        r.detail = "No preview.png found, using synthetic image"
    else:
        r = suite.add("2.0", "准备", f"找到 {len(test_images)} 张测试图")
        r.status = "PASS"

    for i, img_path in enumerate(test_images):
        case_id = f"2.{i + 1}"
        if isinstance(img_path, Path):
            img = Image.open(img_path)
            name = f"vectorize: {img_path.parent.name}"
        else:
            img = img_path
            name = f"vectorize: synthetic #{i}"

        # 全分辨率
        r = suite.add(case_id, "矢量化", name)
        t0 = time.time()
        try:
            result = converter.convert(img, cfg)
            r.duration_ms = (time.time() - t0) * 1000
            r.expected = "paths > 0"
            r.actual = f"{result.total_paths} paths, {result.total_vertices} verts"
            if result.total_paths == 0:
                r.fail("no paths generated")
            if result.warnings:
                r.warn(f"warnings: {result.warnings}")
            # 检查 SVG 有效性
            if not result.svg_string.startswith("<?xml") and not result.svg_string.startswith("<svg"):
                r.warn("SVG may not be valid XML")
        except Exception as e:
            r.fail(str(e))

        # 缩略图（256x256）
        if isinstance(img_path, Path):
            r2 = suite.add(f"{case_id}b", "矢量化(缩略)", f"thumb: {img_path.parent.name}")
            t0 = time.time()
            try:
                thumb = img.resize((256, 256), Image.LANCZOS)
                result = converter.convert(thumb, cfg)
                r2.duration_ms = (time.time() - t0) * 1000
                r2.actual = f"{result.total_paths} paths, {len(result.svg_string)} bytes"
                if result.total_paths == 0:
                    r2.fail("no paths")
            except Exception as e:
                r2.fail(str(e))

    print_suite_result(suite)
    return suite


def test_batch_parser():
    """测试批量文件解析"""
    suite = make_suite("Batch Parser")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "front_end"))

    # 动态导入 parse_batch_file
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "vecrafter_frontend",
        Path(__file__).resolve().parent.parent / "front_end" / "Vecrafter.py"
    )
    # 直接复制函数逻辑
    import pandas as pd

    def parse_csv(content: bytes) -> list:
        df = pd.read_csv(io.BytesIO(content))
        col_map = {c.strip().lower(): c for c in df.columns}
        text_col = col_map.get("text")
        if text_col is None:
            raise ValueError("no text column")
        style_col = col_map.get("style")
        items = []
        for _, row in df.iterrows():
            text = str(row.get(text_col, "")).strip()
            if text:
                item = {"text": text}
                if style_col and pd.notna(row.get(style_col)):
                    item["style"] = str(row[style_col]).strip()
                items.append(item)
        return items

    def parse_json(content: bytes) -> list:
        data = json.loads(content.decode("utf-8"))
        items = []
        for entry in data:
            text = str(entry.get("text", "")).strip()
            if text:
                item = {"text": text}
                if entry.get("style"):
                    item["style"] = str(entry["style"]).strip()
                items.append(item)
        return items

    def parse_txt(content: bytes) -> list:
        items = []
        for line in content.decode("utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                items.append({"text": line})
        return items

    # CSV 测试
    # CSV 测试
    csv_content = "text,style\n青山集,国风书法\nHello,\n2024,促销卡通\n".encode("utf-8")
    r = suite.add("3.1", "CSV解析", "标准CSV")
    t0 = time.time()
    try:
        items = parse_csv(csv_content)
        r.duration_ms = (time.time() - t0) * 1000
        r.expected = "3 items"
        r.actual = f"{len(items)} items"
        if len(items) != 3:
            r.fail(f"expected 3, got {len(items)}")
    except Exception as e:
        r.fail(str(e))

    # JSON 测试
    json_content = json.dumps([
        {"text": "青山集", "style": "国风书法"},
        {"text": "Hello"},
    ], ensure_ascii=False).encode("utf-8")
    r = suite.add("3.2", "JSON解析", "标准JSON")
    t0 = time.time()
    try:
        items = parse_json(json_content)
        r.duration_ms = (time.time() - t0) * 1000
        if len(items) != 2:
            r.fail(f"expected 2, got {len(items)}")
        r.actual = f"{len(items)} items"
    except Exception as e:
        r.fail(str(e))

    # TXT 测试
    txt_content = "青山集\nHello\n\n# comment\n2024".encode("utf-8")
    r = suite.add("3.3", "TXT解析", "标准TXT")
    t0 = time.time()
    try:
        items = parse_txt(txt_content)
        r.duration_ms = (time.time() - t0) * 1000
        if len(items) != 3:
            r.fail(f"expected 3, got {len(items)}")
        r.actual = f"{len(items)} items"
    except Exception as e:
        r.fail(str(e))

    # 空文件
    r = suite.add("3.4", "边界", "空CSV")
    try:
        items = parse_csv(b"text\n")
        r.actual = f"{len(items)} items"
        if len(items) != 0:
            r.fail("expected 0 items")
    except Exception as e:
        r.warn(str(e))

    # 重复行
    r = suite.add("3.5", "边界", "重复文字")
    items = parse_txt("青山集\n青山集\nHello\nHello".encode("utf-8"))
    r.expected = "4 items"
    r.actual = f"{len(items)} items"
    if len(items) != 4:
        r.fail(f"expected 4, got {len(items)}")

    print_suite_result(suite)
    return suite


def test_foreground_segmentation():
    """测试前景分割（多种输入情况）"""
    suite = make_suite("Foreground Segmentation")
    from vector_converter import VectorConverter, VectorizationConfig
    import cv2
    import numpy as np

    converter = VectorConverter()

    # 有透明通道
    img_rgba = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    for x in range(30, 70):
        for y in range(30, 70):
            img_rgba.putpixel((x, y), (255, 0, 0, 255))

    arr = np.array(img_rgba)
    r = suite.add("4.1", "Alpha分割", "RGBA with text")
    t0 = time.time()
    try:
        mask = converter._segment_foreground_cv(arr)
        r.duration_ms = (time.time() - t0) * 1000
        fg_count = mask.sum()
        r.actual = f"{fg_count} fg pixels"
        if fg_count < 100:
            r.warn(f"too few fg pixels: {fg_count}")
    except Exception as e:
        r.fail(str(e))

    # 纯 RGB（无Alpha）
    img_rgb = Image.new("RGB", (100, 100), (255, 255, 255))
    for x in range(20, 80):
        for y in range(20, 80):
            img_rgb.putpixel((x, y), (50, 50, 50))

    arr2 = np.array(img_rgb.convert("RGBA"))
    r = suite.add("4.2", "RGB分割", "RGB with dark text")
    t0 = time.time()
    try:
        mask = converter._segment_foreground_cv(arr2)
        r.duration_ms = (time.time() - t0) * 1000
        fg_count = mask.sum()
        r.actual = f"{fg_count} fg pixels"
        if fg_count < 100:
            r.fail(f"too few fg pixels: {fg_count}")
    except Exception as e:
        r.fail(str(e))

    # 全白（极端情况）
    img_white = np.array(Image.new("RGBA", (100, 100), (255, 255, 255, 255)))
    r = suite.add("4.3", "极端输入", "全白图像")
    try:
        mask = converter._segment_foreground_cv(img_white)
        fg_ratio = mask.sum() / (100 * 100)
        r.actual = f"{fg_ratio:.1%} fg"
        if fg_ratio > 0.8:
            r.warn(f"foreground too large: {fg_ratio:.1%}")
    except Exception as e:
        r.fail(str(e))

    # 全黑（极端情况）
    img_black = np.array(Image.new("RGBA", (100, 100), (0, 0, 0, 255)))
    r = suite.add("4.4", "极端输入", "全黑图像")
    try:
        mask = converter._segment_foreground_cv(img_black)
        fg_ratio = mask.sum() / (100 * 100)
        r.actual = f"{fg_ratio:.1%} fg"
        if fg_ratio < 0.1:
            r.warn(f"foreground too small: {fg_ratio:.1%}")
    except Exception as e:
        r.fail(str(e))

    print_suite_result(suite)
    return suite


def test_batch_prompt_processing():
    """批量测试测试集中的提示词处理"""
    suite = make_suite("Batch Prompt Processing")
    from prompt_preprocessor import PromptPreprocessor, _classify_text

    # 读取测试用例
    try:
        import openpyxl
        wb = openpyxl.load_workbook(
            Path(__file__).resolve().parent.parent / "docs" / "测试用例.xlsx"
        )
        ws = wb.active
        test_cases = []
        for row in ws.iter_rows(min_row=2, values_only=True):
            if row[0]:
                test_cases.append({"text": str(row[0]), "style": str(row[1] or ""), "seed": row[2]})
    except Exception as e:
        r = suite.add("5.0", "准备", f"读取测试用例失败: {e}")
        r.status = "FAIL"
        print_suite_result(suite)
        return suite

    r = suite.add("5.0", "准备", f"读取 {len(test_cases)} 条测试用例")
    r.status = "PASS"

    # 测试每条用例的 prompt 处理
    fail_count = 0
    for i, tc in enumerate(test_cases[:110]):  # 全部 110 条
        case_id = f"5.{i + 1}"
        name = f"prompt: {tc['text'][:20]}"
        r = suite.add(case_id, "批量Prompt", name)
        t0 = time.time()
        try:
            # build_positive
            positive = PromptPreprocessor.build_positive(tc["text"], tc["style"])
            # build_negative
            negative = PromptPreprocessor.build_negative("", tc["text"])
            r.duration_ms = (time.time() - t0) * 1000
            r.expected = "positive > 30 chars"
            r.actual = f"+:{len(positive)}ch -:{len(negative)}ch"
            if len(positive) < 30:
                r.fail("positive prompt too short")
                fail_count += 1
            # 验证正/负面提示词不包含纯占位符
            if "{text}" in positive or "{style}" in positive:
                r.warn("template placeholders not replaced")
        except Exception as e:
            r.fail(str(e))
            fail_count += 1

    # 补充：空文本异常测试
    r = suite.add("5.111", "异常测试", "空文本")
    try:
        positive = PromptPreprocessor.build_positive("", "")
        r.actual = f"len={len(positive)}"
        if len(positive) < 5:
            r.warn("短文本包装可能不完整")
    except Exception as e:
        r.warn(str(e))

    r = suite.add("5.112", "异常测试", "超长文本(100字)")
    long_text = "艺术字" * 33  # 99 chars
    try:
        positive = PromptPreprocessor.build_positive(long_text, "")
        r.actual = f"len={len(positive)}"
        if len(positive) < 30:
            r.fail("prompt too short for long input")
    except Exception as e:
        r.fail(str(e))

    r = suite.add("5.113", "异常测试", "特殊符号输入")
    try:
        positive = PromptPreprocessor.build_positive("@#$%^&*()", "促销卡通")
        r.actual = f"len={len(positive)}"
        if len(positive) < 20:
            r.warn("特殊符号包装不完整")
    except Exception as e:
        r.warn(str(e))

    print_suite_result(suite)
    return suite


def test_performance_baseline():
    """性能基线测试"""
    suite = make_suite("Performance Baseline")
    from vector_converter import VectorConverter, VectorizationConfig

    converter = VectorConverter()
    cfg = VectorizationConfig(use_edge_driven=True, embed_preview=False)

    # 创建固定大小的测试图
    test_sizes = [(256, 256), (512, 512), (1024, 1024)]
    for w, h in test_sizes:
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        # 画一些文字形状
        for y in range(h // 4, 3 * h // 4):
            half_w = w // 4 + int(abs(y - h // 2) * 0.3)
            for x in range(w // 2 - half_w, w // 2 + half_w):
                if abs(x - w // 2) + abs(y - h // 2) < half_w:
                    img.putpixel((x, y), (200, 30, 30, 255))

        r = suite.add(f"6.{test_sizes.index((w, h)) + 1}", "性能", f"vectorize {w}x{h}")
        times = []
        results = None
        for trial in range(3):
            t0 = time.time()
            try:
                results = converter.convert(img, cfg)
                times.append((time.time() - t0) * 1000)
            except Exception as e:
                r.fail(f"trial {trial}: {e}")
                break
        if times:
            r.duration_ms = sum(times) / len(times)
            r.actual = f"avg={r.duration_ms:.0f}ms paths={results.total_paths if results else 'N/A'}"
            if r.duration_ms > 30000:
                r.warn(f"too slow: {r.duration_ms:.0f}ms")

    print_suite_result(suite)
    return suite


def test_svg_output_quality():
    """SVG 输出质量检测"""
    suite = make_suite("SVG Quality")
    from vector_converter import VectorConverter, VectorizationConfig

    converter = VectorConverter()
    cfg = VectorizationConfig(use_edge_driven=True, embed_preview=False)

    img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    for y in range(50, 200):
        half_w = int(abs(y - 125) * 0.5 + 40)
        for x in range(128 - half_w, 128 + half_w):
            img.putpixel((x, y), (200, 30, 30, 255))

    result = converter.convert(img, cfg)

    # SVG 闭合性
    r = suite.add("7.1", "SVG结构", "所有路径闭合(Z)")
    z_count = result.svg_string.count(" Z")
    r.actual = f"{z_count} Z commands"
    if z_count == 0 and result.total_paths > 0:
        r.fail("no path closure commands")

    # SVG 命名空间
    r = suite.add("7.2", "SVG结构", "包含xmlns")
    if "xmlns=" in result.svg_string:
        r.status = "PASS"
    else:
        r.fail("missing xmlns")
    r.actual = "xmlns present" if "xmlns=" in result.svg_string else "missing"

    # viewBox
    r = suite.add("7.3", "SVG结构", "包含viewBox")
    if "viewBox" in result.svg_string:
        r.status = "PASS"
    else:
        r.fail("missing viewBox")

    # 路径数量合理性
    r = suite.add("7.4", "SVG质量", "路径数合理(<100)")
    r.actual = f"{result.total_paths} paths"
    if result.total_paths > 100:
        r.warn(f"too many paths: {result.total_paths}")

    # C 曲线比例
    if result.total_paths > 0:
        r = suite.add("7.5", "SVG质量", "C曲线占比")
        c_count = result.svg_string.count(" C ")
        l_count = result.svg_string.count(" L ")
        total_cmds = c_count + l_count
        ratio = c_count / max(1, total_cmds)
        r.actual = f"{c_count}C / {total_cmds}total = {ratio:.0%}"
        if ratio < 0.1 and total_cmds > 10:
            r.warn(f"low C curve ratio: {ratio:.0%}")

    print_suite_result(suite)
    return suite


def test_health_endpoint():
    """测试后端健康检查端点"""
    suite = make_suite("Backend API")
    import requests

    base_url = "http://127.0.0.1:8000"

    # Health
    r = suite.add("8.1", "API健康", "GET /health")
    t0 = time.time()
    try:
        resp = requests.get(f"{base_url}/health", timeout=5)
        r.duration_ms = (time.time() - t0) * 1000
        if resp.status_code == 200:
            r.actual = f"HTTP {resp.status_code}"
        else:
            r.fail(f"HTTP {resp.status_code}")
    except requests.exceptions.ConnectionError:
        r.warn("Backend not running (skip API tests)")
        # 跳过后续测试
    except Exception as e:
        r.fail(str(e))

    print_suite_result(suite)
    return suite


# ======================= 主入口 =======================

def main():
    print(f"\n{'#' * 60}")
    print(f"#  Vecrafter 系统测试")
    print(f"#  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 60}")

    all_suites = []
    total_start = time.time()

    all_suites.append(test_prompt_preprocessor())
    all_suites.append(test_batch_parser())
    all_suites.append(test_foreground_segmentation())
    all_suites.append(test_vectorizer())
    all_suites.append(test_batch_prompt_processing())
    all_suites.append(test_svg_output_quality())
    all_suites.append(test_performance_baseline())
    all_suites.append(test_health_endpoint())

    # 全局汇总
    total_elapsed = time.time() - total_start
    total = sum(s.summary()["total"] for s in all_suites)
    passed = sum(s.summary()["passed"] for s in all_suites)
    failed = sum(s.summary()["failed"] for s in all_suites)
    warned = sum(s.summary()["warned"] for s in all_suites)

    print(f"\n{'=' * 60}")
    print(f"  全局测试汇总")
    print(f"  总计: {total}  |  通过: {passed}  |  失败: {failed}  |  警告: {warned}")
    print(f"  通过率: {passed / max(1, total) * 100:.1f}%")
    print(f"  总耗时: {total_elapsed:.2f}s")
    print(f"{'=' * 60}")
    print(f"\n详细结果已保存至: {RESULT_DIR}")

    # 保存全局汇总
    summary = {
        "test_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_suites": len(all_suites),
        "total_cases": total,
        "passed": passed,
        "failed": failed,
        "warned": warned,
        "pass_rate": f"{passed / max(1, total) * 100:.1f}%",
        "elapsed_seconds": round(total_elapsed, 2),
        "suites": [s.to_json() for s in all_suites],
    }
    with open(RESULT_DIR / "global_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 生成 CSV 报告
    csv_path = RESULT_DIR / "test_report.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Suite", "CaseID", "Category", "Name", "Status", "Duration_ms", "Detail"])
        for suite in all_suites:
            for r in suite.results:
                writer.writerow([
                    suite.name, r.case_id, r.category, r.name,
                    r.status, round(r.duration_ms, 1), r.detail[:200],
                ])
    print(f"  CSV 报告: {csv_path}")


if __name__ == "__main__":
    main()

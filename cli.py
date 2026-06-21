#!/usr/bin/env python3
"""
Vecrafter CLI 入口

三种模式：
  generate  — 单条/批量生成艺术字
  vectorize — 对已有图片矢量化
  batch     — 从 CSV/JSON/TXT 批量生成并矢量化

使用示例:
  python cli.py generate --text "青山集" --style "国风书法" --output out/
  python cli.py vectorize --input input.png --output output.svg
  python cli.py batch --file prompts.csv --output out/
  python cli.py check-env                    # 环境检测
"""

import sys, os, time, json, base64, csv, io, argparse, glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

BACKEND_URL = "http://127.0.0.1:8000"

# 绕过系统代理（后端 & ComfyUI 均为本地/LAN 服务）
import requests as _requests
_req_session = _requests.Session()
_req_session.trust_env = False



def cmd_generate(args):
    """单条生成"""
    import requests
    print(f"[生成] text={args.text} style={args.style} seed={args.seed}")
    payload = {
        "text": args.text,
        "style_prompt": args.style,
        "seed": args.seed,
        "width": args.width,
        "height": args.height,
    }
    t0 = time.time()
    r = _req_session.post(f"{BACKEND_URL}/generate", json=payload, timeout=900)
    print(f"  耗时: {time.time()-t0:.1f}s, HTTP {r.status_code}")
    if r.status_code != 200:
        print(f"  失败: {r.json().get('detail', r.text)}")
        return
    data = r.json()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 保存图片
    for i, b64 in enumerate(data.get("images", [])):
        png = base64.b64decode(b64)
        (out_dir / f"result_{i}.png").write_bytes(png)
    print(f"  已保存: {out_dir}/")

    # 保存元数据
    if data.get("metadata"):
        (out_dir / "metadata.json").write_text(
            json.dumps(data["metadata"], ensure_ascii=False, indent=2), encoding="utf-8"
        )


def cmd_vectorize(args):
    """单张矢量化"""
    import requests
    from PIL import Image

    img = Image.open(args.input)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    payload = {
        "image_b64": b64,
        "color_clusters": args.clusters,
        "use_edge_driven": True,
        "embed_preview": False,
    }
    t0 = time.time()
    r = _req_session.post(f"{BACKEND_URL}/vectorize", json=payload, timeout=600)
    print(f"[矢量化] {args.input.name} -> {args.output}")
    print(f"  耗时: {time.time()-t0:.1f}s")
    if r.status_code != 200:
        print(f"  失败: {r.json().get('detail', r.text)}")
        return
    data = r.json()
    if data.get("svg_string"):
        Path(args.output).write_text(data["svg_string"], encoding="utf-8")
        print(f"  SVG: {args.output}")
    print(f"  路径数: {data.get('total_paths', '?')}")


def cmd_batch(args):
    """批量处理"""
    import requests, csv

    ext = Path(args.file).suffix.lower()
    items = []
    if ext == ".csv":
        with open(args.file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("text"):
                    items.append({
                        "text": row["text"].strip(),
                        "style": row.get("style", args.style or "").strip(),
                        "seed": int(row["seed"]) if row.get("seed") else args.seed,
                    })
    elif ext == ".json":
        data = json.loads(Path(args.file).read_text(encoding="utf-8"))
        for entry in data:
            if entry.get("text"):
                items.append({
                    "text": entry["text"].strip(),
                    "style": entry.get("style", args.style or ""),
                    "seed": entry.get("seed", args.seed),
                })
    elif ext == ".txt":
        for line in Path(args.file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                items.append({"text": line, "style": args.style, "seed": args.seed})

    print(f"[批量] 读取 {len(items)} 条")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for i, item in enumerate(items):
        print(f"  [{i+1}/{len(items)}] {item['text'][:20]}...", end=" ", flush=True)
        t0 = time.time()
        try:
            payload = {
                "text": item["text"],
                "style_prompt": item.get("style", ""),
                "seed": int(item.get("seed", args.seed)),
                "width": args.width,
                "height": args.height,
            }
            r = _req_session.post(f"{BACKEND_URL}/generate", json=payload, timeout=900)
            if r.status_code == 200:
                data = r.json()
                item_dir = out_dir / f"{i:04d}_{item['text'][:16]}"
                item_dir.mkdir(exist_ok=True)
                for j, b64 in enumerate(data.get("images", [])):
                    (item_dir / f"result_{j}.png").write_bytes(base64.b64decode(b64))
                if data.get("metadata"):
                    (item_dir / "metadata.json").write_text(
                        json.dumps(data["metadata"], ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                results.append({
                    "index": i, "text": item["text"], "status": "success",
                    "time_s": round(time.time()-t0, 1)
                })
                print(f"OK")
            else:
                results.append({
                    "index": i, "text": item["text"], "status": "failed",
                    "error": r.json().get("detail", r.text)[:100]
                })
                print(f"FAIL")
        except Exception as e:
            results.append({"index": i, "text": item["text"], "status": "failed", "error": str(e)[:100]})
            print(f"ERR")

    (out_dir / "batch_report.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    success = sum(1 for r in results if r["status"] == "success")
    print(f"\n  完成: {success}/{len(items)} 成功")


def cmd_check_env(args):
    """检测运行环境"""
    print("=== Vecrafter 环境检测 ===")
    ok = True

    # Python 版本
    print(f"  Python: {sys.version.split()[0]}")
    if sys.version_info < (3, 10):
        print("    [警告] 推荐 Python 3.10+")
        ok = False

    # 核心依赖
    deps = {
        "opencv-python": "cv2",
        "Pillow": "PIL",
        "numpy": "numpy",
        "scikit-image": "skimage",
        "svgwrite": "svgwrite",
        "requests": "requests",
        "streamlit": "streamlit",
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
    }
    print(f"\n  核心依赖:")
    for name, mod in deps.items():
        try:
            __import__(mod)
            print(f"    {name:20s} [OK]")
        except ImportError:
            print(f"    {name:20s} [NO] 未安装")
            ok = False

    # ComfyUI 配置检测
    config_dir = Path(__file__).resolve().parent / "config"
    print(f"\n  配置文件:")
    for f in sorted(glob.glob(str(config_dir / "*.json"))):
        try:
            data = json.loads(Path(f).read_text(encoding="utf-8"))
            # 提取模型名
            models = []
            for nid in ["1", "2", "3"]:
                if nid in data and "inputs" in data[nid]:
                    for k in ["unet_name", "clip_name", "vae_name"]:
                        if k in data[nid]["inputs"]:
                            models.append(data[nid]["inputs"][k])
            print(f"    {Path(f).name:30s} [OK] 模型: {', '.join(models)}")
        except Exception as e:
            print(f"    {Path(f).name:30s} [NO] {e}")

    # ComfyUI API 连通性
    print(f"\n  ComfyUI:")
    comfyui_url = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
    print(f"     配置地址: {comfyui_url}")
    print(f"     (可通过环境变量 COMFYUI_URL 修改)")
    import requests as req
    # 检测代理干扰
    proxy = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
    if proxy:
        print(f"     检测到系统代理: {proxy}")
        if "127.0.0.1" not in (os.environ.get("NO_PROXY", "") + os.environ.get("no_proxy", "")):
            print(f"     [WARN] 代理可能干扰本地连接，建议设置: set NO_PROXY=127.0.0.1,localhost")
    try:
        # 绕过代理测试（ComfyUI 不应走代理）
        r = req.get(comfyui_url, timeout=3, proxies={"http": None, "https": None})
        if r.status_code == 200 or r.status_code == 404:
            print(f"     连通性: [OK] 可达")
        else:
            print(f"     连通性: [WARN] 返回 {r.status_code}")
    except:
        print(f"     连通性: [NO] 不可达，请检查 ComfyUI 是否已启动")
        print(f"     如 ComfyUI 在其他地址，请设置: set COMFYUI_URL=http://你的IP:8188")

    # GPU 检测
    print(f"\n  GPU:")
    try:
        import torch
        print(f"     PyTorch: {torch.__version__} [OK]")
        if torch.cuda.is_available():
            print(f"     CUDA: [OK] ({torch.cuda.get_device_name(0)})")
        else:
            print(f"     CUDA: [NO] (CPU 模式)")
    except ImportError:
        print(f"     PyTorch: [NO] 未安装 (部分功能可选)")

    print(f"\n  环境状态: {'[OK] 就绪' if ok else '[WARN] 部分缺失'}")


def cmd_repl():
    """交互式命令行（类似 Claude Code 风格）"""
    import shutil
    width = shutil.get_terminal_size().columns

    print("=" * min(width, 60))
    print("  Vecrafter - 矢量艺术字工坊 (交互模式)")
    print("  输入 help 查看命令，exit 退出")
    print("=" * min(width, 60))
    print()
    print("! 首次使用先运行 check-env 检测环境")
    print()

    while True:
        try:
            cmd = input("vecrafter> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见!")
            break

        if not cmd:
            continue

        if cmd in ("exit", "quit", "q"):
            print("再见!")
            break

        if cmd in ("help", "?"):
            print("""可用命令:
  generate <文字> [--style 风格] [--seed 种子]  生成艺术字
  vectorize <图片路径>                           矢量化图片
  batch <文件路径>                                批量生成
  check-env                                      环境检测
  status                                         查看后端状态
  exit/quit                                       退出""")
            continue

        if cmd == "check-env":
            cmd_check_env(argparse.Namespace())
            continue

        if cmd == "status":
            _cmd_status()
            continue

        if cmd.startswith("generate "):
            _cmd_interactive_generate(cmd[9:].strip())
            continue

        if cmd.startswith("vectorize "):
            _cmd_interactive_vectorize(cmd[10:].strip())
            continue

        if cmd.startswith("batch "):
            _cmd_interactive_batch(cmd[6:].strip())
            continue

        print(f"未知命令: {cmd}  输入 help 查看可用命令")


def _cmd_status():
    try:
        import requests
        t0 = time.time()
        r = _req_session.get(f"{BACKEND_URL}/health", timeout=5)
        print(f"  后端状态: {'运行中' if r.status_code==200 else '异常'} ({time.time()-t0:.1f}s)")
    except:
        print("  后端状态: 未运行 (启动: python back_end/main.py)")


def _cmd_interactive_generate(text_input: str):
    import requests, re
    style = ""; seed = 42
    parts = re.split(r' (--\w+) ', text_input)
    text = parts[0].strip() if parts else text_input
    for i in range(1, len(parts), 2):
        flag = parts[i].strip()
        val = parts[i+1].strip() if i+1 < len(parts) else ""
        if flag == "--style": style = val
        elif flag == "--seed":
            try: seed = int(val)
            except: pass
    print(f"  text={text} style={style} seed={seed}")
    payload = {"text": text, "style_prompt": style, "seed": seed, "width": 1024, "height": 600}
    t0 = time.time()
    try:
        r = _req_session.post(f"{BACKEND_URL}/generate", json=payload, timeout=900)
        if r.status_code == 200:
            data = r.json()
            out_dir = Path("output") / f"int_{int(time.time())}"
            out_dir.mkdir(parents=True, exist_ok=True)
            for j, b64 in enumerate(data.get("images", [])):
                (out_dir / f"result_{j}.png").write_bytes(base64.b64decode(b64))
            if data.get("metadata"):
                (out_dir / "metadata.json").write_text(json.dumps(data["metadata"], ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  [OK] {time.time()-t0:.1f}s -> {out_dir}/")
        else:
            print(f"  [X] {r.json().get('detail', r.text)[:100]}")
    except requests.exceptions.ConnectionError:
        print("  [X] 后端未运行，请先启动 python back_end/main.py")
    except Exception as e:
        print(f"  [X] {str(e)[:60]}")


def _cmd_interactive_vectorize(path: str):
    import requests
    p = Path(path)
    if not p.exists(): print(f"  [X] 文件不存在: {path}"); return
    try:
        from PIL import Image
        img = Image.open(p); buf = io.BytesIO(); img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        t0 = time.time()
        r = _req_session.post(f"{BACKEND_URL}/vectorize", json={"image_b64": b64, "use_edge_driven": True, "embed_preview": False}, timeout=600)
        if r.status_code == 200:
            data = r.json()
            out_path = p.with_suffix(".svg")
            out_path.write_text(data["svg_string"], encoding="utf-8")
            print(f"  [OK] {time.time()-t0:.1f}s -> {out_path}  路径数: {data.get('total_paths', '?')}")
        else:
            print(f"  [X] {r.json().get('detail', r.text)[:100]}")
    except requests.exceptions.ConnectionError:
        print("  [X] 后端未运行")
    except Exception as e:
        print(f"  [X] {str(e)[:60]}")


def _cmd_interactive_batch(path: str):
    import requests, csv
    ext = Path(path).suffix.lower()
    if not Path(path).exists(): print(f"  [X] 文件不存在: {path}"); return
    items = []
    try:
        if ext == ".csv":
            with open(path, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("text"): items.append({"text": row["text"].strip(), "style": row.get("style",""), "seed": int(row["seed"]) if row.get("seed") else 42})
        elif ext == ".json":
            for entry in json.loads(Path(path).read_text(encoding="utf-8")):
                if entry.get("text"): items.append({"text": entry["text"], "style": entry.get("style",""), "seed": entry.get("seed",42)})
        elif ext == ".txt":
            for line in Path(path).read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"): items.append({"text": line, "style": "", "seed": 42})
    except Exception as e: print(f"  [X] 解析失败: {e}"); return
    if not items: print("  [X] 无有效提示词"); return
    print(f"  共 {len(items)} 条，开始生成...")
    out_dir = Path("output") / f"batch_{int(time.time())}"
    out_dir.mkdir(parents=True, exist_ok=True)
    success = 0
    for i, item in enumerate(items):
        print(f"    [{i+1}/{len(items)}] {item['text'][:20]}...", end=" ", flush=True)
        try:
            r = _req_session.post(f"{BACKEND_URL}/generate", json={"text": item["text"], "style_prompt": item["style"], "seed": item["seed"], "width": 1024, "height": 600}, timeout=900)
            if r.status_code == 200:
                data = r.json()
                item_dir = out_dir / f"{i:04d}_{item['text'][:16]}"
                item_dir.mkdir(exist_ok=True)
                for j, b64 in enumerate(data.get("images", [])): (item_dir / f"result_{j}.png").write_bytes(base64.b64decode(b64))
                if data.get("metadata"): (item_dir / "metadata.json").write_text(json.dumps(data["metadata"], ensure_ascii=False, indent=2), encoding="utf-8")
                success += 1; print("OK")
            else: print("FAIL")
        except: print("ERR")
    print(f"  完成: {success}/{len(items)} 成功 -> {out_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Vecrafter - 矢量艺术字生成与转换工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    # generate
    gen = sub.add_parser("generate", help="生成艺术字")
    gen.add_argument("--text", required=True, help="文字内容")
    gen.add_argument("--style", default="", help="风格提示词")
    gen.add_argument("--seed", type=int, default=42, help="随机种子")
    gen.add_argument("--width", type=int, default=1024)
    gen.add_argument("--height", type=int, default=600)
    gen.add_argument("--output", default="./output", help="输出目录")

    # vectorize
    vec = sub.add_parser("vectorize", help="矢量化图片")
    vec.add_argument("--input", type=argparse.FileType("rb"), required=True, help="输入图片路径")
    vec.add_argument("--output", default="output.svg", help="输出 SVG 路径")
    vec.add_argument("--clusters", type=int, default=8, help="颜色聚类数")

    # batch
    bat = sub.add_parser("batch", help="批量生成（CSV/JSON/TXT）")
    bat.add_argument("--file", required=True, help="提示词文件路径")
    bat.add_argument("--style", default="", help="默认风格")
    bat.add_argument("--seed", type=int, default=42, help="默认种子")
    bat.add_argument("--width", type=int, default=1024)
    bat.add_argument("--height", type=int, default=600)
    bat.add_argument("--output", default="./batch_output", help="输出目录")

    # check-env
    sub.add_parser("check-env", help="检测运行环境")

    # repl (交互模式)
    sub.add_parser("repl", help="交互式命令行")

    args = parser.parse_args()
    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "vectorize":
        cmd_vectorize(args)
    elif args.command == "batch":
        cmd_batch(args)
    elif args.command == "check-env":
        cmd_check_env(args)
    elif args.command == "repl":
        cmd_repl()
    elif args.command is None:
        cmd_repl()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

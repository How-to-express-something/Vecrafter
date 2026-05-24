import json
import time
import requests
import websocket
from PIL import Image
from io import BytesIO
from typing import Optional, Dict, Any
import threading

class ComfyUIWrapper:
    def __init__(self, server_url: str = "http://127.0.0.1:8188"):
        self.server_url = server_url.rstrip("/")
        self.ws_url = server_url.replace("http://", "ws://", 1).rstrip("/") + "/ws"
        self.client_id = str(int(time.time() * 1000))  # 唯一客户端 ID

    def load_workflow(self, workflow_path: str) -> Dict[str, Any]:
        with open(workflow_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def queue_prompt(self, workflow: Dict[str, Any]) -> str:
        """提交工作流，返回 prompt_id"""
        payload = {"prompt": workflow, "client_id": self.client_id}
        resp = requests.post(f"{self.server_url}/prompt", json=payload)
        if resp.status_code != 200:
            raise Exception(f"Queue prompt failed: {resp.text}")
        return resp.json()["prompt_id"]

    def get_history(self, prompt_id: str) -> Dict[str, Any]:
        """获取指定 prompt_id 的历史记录"""
        resp = requests.get(f"{self.server_url}/history/{prompt_id}")
        if resp.status_code != 200:
            raise Exception(f"Get history failed: {resp.text}")
        return resp.json()

    def wait_for_prompt(self, prompt_id: str):
        """通过 WebSocket 等待生成完成"""
        done = threading.Event()
        ws = websocket.WebSocket()
        ws.connect(self.ws_url)

        try:
            while not done.is_set():
                msg = ws.recv()
                if not msg:
                    continue
                data = json.loads(msg)
                if data["type"] == "executing":
                    if data["data"]["node"] is None and data["data"]["prompt_id"] == prompt_id:
                        done.set()  # 生成完成
        finally:
            ws.close()

    def get_output_images(self, prompt_id: str) -> list[Image.Image]:
        """从历史记录中提取输出图片"""
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
                    # 下载图片
                    params = {
                        "filename": filename,
                        "subfolder": subfolder,
                        "type": img_type,
                    }
                    resp = requests.get(f"{self.server_url}/view", params=params)
                    if resp.status_code == 200:
                        img = Image.open(BytesIO(resp.content))
                        images.append(img)
        return images

    def generate(
        self,
        workflow_path: str,
        positive_prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        seed: int = -1,
        steps: int = 20,
        cfg: float = 7.0,
        sampler_name: str = "euler",
        scheduler: str = "normal",
        denoise: float = 1.0,
        **extra_params
    ) -> list[Image.Image]:
        """
        对用户透明的生成接口
        :param workflow_path: 基础工作流 API JSON 文件路径
        :param positive_prompt: 正向提示词
        :param negative_prompt: 负向提示词
        :param width: 图片宽度
        :param height: 图片高度
        :param seed: 随机种子，-1 表示随机
        :param steps: 采样步数
        :param cfg: CFG scale
        :param sampler_name: 采样器名称
        :param scheduler: 调度器
        :param denoise: 降噪强度
        :return: PIL Image 对象列表
        """
        workflow = self.load_workflow(workflow_path)

        # ---------- 根据你的节点 ID 修改参数 ----------
        # 这里假设了常见的节点 ID，你需要根据实际工作流调整
        # 可以通过打印 workflow 的 key 来确定节点 ID
        # 节点 ID 是字符串数字，如 "6"、"7"、"3"、"8"

        # 正向提示词节点 (假设 id 为 "6")
        if "6" in workflow:
            workflow["6"]["inputs"]["text"] = positive_prompt

        # 负向提示词节点 (假设 id 为 "7")
        if "7" in workflow:
            workflow["7"]["inputs"]["text"] = negative_prompt

        # 图像尺寸节点 (假设 id 为 "8"，EmptyLatentImage)
        if "8" in workflow:
            workflow["8"]["inputs"]["width"] = width
            workflow["8"]["inputs"]["height"] = height

        # KSampler 节点 (假设 id 为 "3")
        if "3" in workflow:
            workflow["3"]["inputs"]["seed"] = seed
            workflow["3"]["inputs"]["steps"] = steps
            workflow["3"]["inputs"]["cfg"] = cfg
            workflow["3"]["inputs"]["sampler_name"] = sampler_name
            workflow["3"]["inputs"]["scheduler"] = scheduler
            workflow["3"]["inputs"]["denoise"] = denoise

        # 提交生成任务
        prompt_id = self.queue_prompt(workflow)
        # 等待完成
        self.wait_for_prompt(prompt_id)
        # 取回图片
        return self.get_output_images(prompt_id)
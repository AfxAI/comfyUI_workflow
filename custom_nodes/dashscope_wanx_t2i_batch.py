import json
import time
import io
import urllib.request
import urllib.error

import numpy as np
import torch
from PIL import Image


def _http_json(url, method="GET", headers=None, body_obj=None, timeout=60):
    headers = headers or {}
    data = None
    if body_obj is not None:
        data = json.dumps(body_obj, ensure_ascii=False).encode("utf-8")
        headers = {**headers, "Content-Type": "application/json"}

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw)


def _download_bytes(url, timeout=120):
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _bytes_to_comfy_image(img_bytes: bytes) -> torch.Tensor:
    # ComfyUI IMAGE: float32, [B,H,W,3], range 0..1
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]  # [1,H,W,3]


class DashScopeWanxText2ImageBatch:
    """
    输入：JSON数组字符串（例如：["描述1","描述2"]）
    输出：IMAGE(batch)，以及 urls_json 方便你调试
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False}),
                "prompts_json": ("STRING", {"multiline": True}),
                "model": ("STRING", {"default": "wan2.5-t2i-preview"}),
                "size": ("STRING", {"default": "1024*1024"}),
                "n_per_prompt": ("INT", {"default": 1, "min": 1, "max": 4}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "prompt_extend": ("BOOLEAN", {"default": True}),
                "watermark": ("BOOLEAN", {"default": False}),
                "base_host": ("STRING", {"default": "dashscope.aliyuncs.com"}),
                "poll_interval_sec": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 10.0}),
                "timeout_sec": ("INT", {"default": 120, "min": 10, "max": 600}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("images", "urls_json", "image_count")
    FUNCTION = "run"
    CATEGORY = "api/dashscope"

    def run(
        self,
        api_key,
        prompts_json,
        model,
        size,
        n_per_prompt,
        negative_prompt,
        prompt_extend,
        watermark,
        base_host,
        poll_interval_sec,
        timeout_sec,
    ):
        # 1) 解析 prompts_json -> list[str]
        try:
            prompts = json.loads(prompts_json)
        except Exception as e:
            raise RuntimeError(f"prompts_json 不是合法JSON：{e}")

        if isinstance(prompts, str):
            prompts = [prompts]
        if not isinstance(prompts, list) or not all(isinstance(x, str) for x in prompts):
            raise RuntimeError("prompts_json 必须是 JSON 字符串数组，例如：[\"描述1\",\"描述2\"]")

        # 2) 逐条 prompt 调万相
        headers_post = {
            "Authorization": f"Bearer {api_key}",
            "X-DashScope-Async": "enable",
        }
        headers_get = {
            "Authorization": f"Bearer {api_key}",
        }

        create_url = f"https://{base_host}/api/v1/services/aigc/text2image/image-synthesis"
        images_tensors = []
        urls = []

        for prompt in prompts:
            payload = {
                "model": model,
                "input": {
                    "prompt": prompt,
                },
                "parameters": {
                    "size": size,
                    "n": int(n_per_prompt),
                    "prompt_extend": bool(prompt_extend),
                    "watermark": bool(watermark),
                },
            }
            if negative_prompt.strip():
                payload["input"]["negative_prompt"] = negative_prompt

            # 2.1 创建任务 -> task_id
            try:
                resp = _http_json(create_url, method="POST", headers=headers_post, body_obj=payload, timeout=60)
            except urllib.error.HTTPError as e:
                msg = e.read().decode("utf-8", errors="ignore") if hasattr(e, "read") else str(e)
                raise RuntimeError(f"创建任务失败(HTTP {e.code}): {msg}")
            except Exception as e:
                raise RuntimeError(f"创建任务失败: {e}")

            task_id = None
            # 常见结构：resp["output"]["task_id"]
            if isinstance(resp, dict):
                task_id = (resp.get("output") or {}).get("task_id") or resp.get("task_id")
            if not task_id:
                raise RuntimeError(f"拿不到 task_id，返回：{resp}")

            # 2.2 轮询任务
            status_url = f"https://{base_host}/api/v1/tasks/{task_id}"
            t0 = time.time()
            final = None
            while True:
                if time.time() - t0 > timeout_sec:
                    raise RuntimeError(f"任务超时：{task_id}")

                try:
                    sresp = _http_json(status_url, method="GET", headers=headers_get, timeout=60)
                except Exception as e:
                    # 网络抖动就等等再试
                    time.sleep(poll_interval_sec)
                    continue

                out = (sresp.get("output") or {}) if isinstance(sresp, dict) else {}
                task_status = out.get("task_status") or sresp.get("task_status")

                if task_status == "SUCCEEDED":
                    final = sresp
                    break
                if task_status in ("FAILED", "CANCELED", "CANCELLED"):
                    raise RuntimeError(f"任务失败：{task_id}，返回：{sresp}")

                time.sleep(poll_interval_sec)

            # 2.3 下载结果图片
            out = final.get("output") or {}
            results = out.get("results") or []
            if not isinstance(results, list) or len(results) == 0:
                raise RuntimeError(f"任务成功但没有 results：{final}")

            for r in results:
                u = r.get("url") if isinstance(r, dict) else None
                if not u:
                    continue
                img_bytes = _download_bytes(u, timeout=120)
                images_tensors.append(_bytes_to_comfy_image(img_bytes))
                urls.append(u)

        if not images_tensors:
            raise RuntimeError("没有生成任何图片（images_tensors 为空）")

        batch = torch.cat(images_tensors, dim=0)  # [B,H,W,3]
        return (batch, json.dumps(urls, ensure_ascii=False), int(batch.shape[0]))


NODE_CLASS_MAPPINGS = {
    "DashScopeWanxText2ImageBatch": DashScopeWanxText2ImageBatch
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DashScopeWanxText2ImageBatch": "DashScope 万相文生图(批量)"
}

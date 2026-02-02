import os
import re
import base64
from io import BytesIO

import numpy as np
from PIL import Image


class ArticleEmbedImagesHTML:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "article": ("STRING",),
                "images": ("IMAGE",),
            },
            "optional": {
                # 匹配占位符：🖼️【图片位置：...】 / 【图片位置: ...】等
                "pattern": ("STRING", {"default": r"(?:🖼️\s*)?【图片位置[:：]\s*.*?】"}),
                "title": ("STRING", {"default": "preview"}),
                # 是否显示占位符那一行（红框那行）
                "show_placeholder": ("BOOLEAN", {"default": False}),
            },
        }

    # 关键：让 ComfyUI 不对 batch/list 做“逐个元素映射执行”，而是一次把所有输入传进来
    INPUT_IS_LIST = True

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("html_path", "html_text", "matched_placeholders", "image_count")
    FUNCTION = "run"
    CATEGORY = "utils"

    def _first(self, v, default=None):
        """INPUT_IS_LIST=True 时，输入可能是 list；这里取第一个值。"""
        if isinstance(v, list):
            return v[0] if len(v) > 0 else default
        return v

    def _to_numpy_batch(self, images):
        """
        把 ComfyUI 的 IMAGE 输入统一成 numpy float32 batch: [B,H,W,C] 范围 0..1
        images 可能是：
          - 一个 batch tensor [B,H,W,C]
          - 单张 [H,W,C]
          - list[...]（上游产生了 list）
        """
        if isinstance(images, list):
            batches = []
            for item in images:
                # item 可能是 torch tensor 或 numpy
                arr = item.cpu().numpy() if hasattr(item, "cpu") else np.asarray(item)
                if arr.ndim == 3:
                    arr = arr[None, ...]
                batches.append(arr)
            if len(batches) == 0:
                return np.zeros((0, 64, 64, 3), dtype=np.float32)
            return np.concatenate(batches, axis=0).astype(np.float32)

        arr = images.cpu().numpy() if hasattr(images, "cpu") else np.asarray(images)
        if arr.ndim == 3:
            arr = arr[None, ...]
        return arr.astype(np.float32)

    def run(
        self,
        article,
        images,
        pattern=r"(?:🖼️\s*)?【图片位置[:：]\s*.*?】",
        title="preview",
        show_placeholder=False,
    ):
        # 处理 INPUT_IS_LIST=True 时的 list 输入
        article = self._first(article, "")
        pattern = self._first(pattern, r"(?:🖼️\s*)?【图片位置[:：]\s*.*?】")
        title = self._first(title, "preview")
        show_placeholder = self._first(show_placeholder, False)

        # 兼容 show_placeholder 传成 0/1 或 "true"/"false"
        if isinstance(show_placeholder, str):
            show_placeholder = show_placeholder.strip().lower() in ("1", "true", "yes", "y", "on")

        # 统一图片 batch
        imgs = self._to_numpy_batch(images)
        batch = int(imgs.shape[0])

        idx = 0
        matched = 0

        def repl(m):
            nonlocal idx, matched
            matched += 1

            # 如果占位符比图片多，多出来的不替换
            if idx >= batch:
                return m.group(0)

            arr = np.clip(imgs[idx] * 255.0, 0, 255).astype(np.uint8)
            pil = Image.fromarray(arr)

            buf = BytesIO()
            pil.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            placeholder = m.group(0)
            idx += 1

            img_html = f"<img style='max-width:100%;border-radius:12px;' src='data:image/png;base64,{b64}'/>"

            if show_placeholder:
                # 调试模式：显示占位符文本（红框那行）
                return (
                    f"<div style='margin:16px 0;padding:12px;border:1px solid #333;border-radius:12px;'>"
                    f"<div style='font-size:14px;opacity:.8;margin-bottom:8px;'>{placeholder}</div>"
                    f"{img_html}"
                    f"</div>"
                )

            # 展示模式：只插图片，不显示占位符那行
            return f"<div style='margin:16px 0;'>{img_html}</div>"

        # 用正则顺序替换，确保第1/2/...占位符对应第1/2/...张图
        html_article = re.sub(str(pattern), repl, str(article), flags=re.DOTALL)

        # 换行转 <br/>
        html_body = html_article.replace("\n", "<br/>")

        html = f"""<!doctype html>
<html><head><meta charset="utf-8"/>
<title>{title}</title>
</head>
<body style="font-family:system-ui;line-height:1.6;padding:24px;max-width:900px;margin:0 auto;">
<div style="white-space:normal;">{html_body}</div>
</body></html>"""

        out_dir = os.path.join(os.getcwd(), "output")
        os.makedirs(out_dir, exist_ok=True)
        html_path = os.path.join(out_dir, f"{title}.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)

        return (html_path, html, matched, batch)


NODE_CLASS_MAPPINGS = {"ArticleEmbedImagesHTML": ArticleEmbedImagesHTML}
NODE_DISPLAY_NAME_MAPPINGS = {"ArticleEmbedImagesHTML": "Article → HTML with Embedded Images"}

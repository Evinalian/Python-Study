"""
练习: PII 检测与脱敏服务 (FastAPI)
==================================

需求:
  用 FastAPI 实现一个 PII（个人身份信息）检测和脱敏的 API 服务。

要求:
  1. API 端点:
     - POST /v1/pii/detect: 检测文本中的 PII，返回找到的 PII 类型和位置
     - POST /v1/pii/mask: 脱敏文本中的 PII，返回脱敏后的文本
     - GET /v1/pii/types: 返回当前支持检测的 PII 类型列表
     - GET /health: 健康检查

  2. PII 类型（至少支持）:
     - 中国大陆手机号
     - 中国大陆身份证号
     - 邮箱地址
     - IPv4/IPv6 地址
     - API Key（OpenAI sk- 格式、AWS AKIA 格式）
     - 信用卡号
     - 车牌号（中国大陆格式）

  3. 脱敏策略（可配置）:
     - "mask": 替换为星号（默认）
     - "hash": 替换为 SHA256 哈希值
     - "redact": 替换为 [REDACTED]
     - 每种 PII 类型可单独指定策略

  4. 请求/响应格式:
     detect 请求:
     {
       "text": "我的手机号是13800138000",
       "pii_types": ["phone_cn", "email"]  // 可选，指定要检测的类型
     }
     detect 响应:
     {
       "findings": [
         {"type": "phone_cn", "value": "13800138000", "start": 6, "end": 17}
       ],
       "count": 1
     }

     mask 请求:
     {
       "text": "我的手机号是13800138000",
       "strategy": "mask"  // "mask" / "hash" / "redact"
     }
     mask 响应:
     {
       "masked_text": "我的手机号是138****8000",
       "findings": [...],
       "count": 1
     }

  5. 使用 Pydantic 严格校验输入输出
  6. 编写单元测试（用 pytest + fastapi.testclient.TestClient）

TODO:
  - [ ] 实现 PIIDetector 类（检测逻辑 + 脱敏逻辑）
  - [ ] 定义 Pydantic 请求/响应模型
  - [ ] 实现 /v1/pii/detect 端点
  - [ ] 实现 /v1/pii/mask 端点
  - [ ] 实现 /v1/pii/types 端点
  - [ ] 编写至少 5 个测试用例
  - [ ] 在 main() 中使用 uvicorn 启动服务

提示:
  - 使用 re 模块进行正则匹配
  - 身份证校验码算法可做（作为 bonus）
  - SHA256: hashlib.sha256(value.encode()).hexdigest()
  - FastAPI TestClient 不需要实际启动服务器
  - 从后往前替换避免索引偏移
"""
import re
import hashlib
import json
from typing import Optional, List
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from dotenv import load_dotenv

load_dotenv()


class PIIDetector:
    """PII 检测器"""

    # TODO: 定义 PII 正则模式字典

    def __init__(self):
        # TODO: 初始化 PII 模式
        pass

    def detect(self, text: str, pii_types: Optional[List[str]] = None) -> list[dict]:
        # TODO: 返回 [{"type": ..., "value": ..., "start": ..., "end": ...}, ...]
        pass

    def mask(self, text: str, strategy: str = "mask") -> tuple[str, list[dict]]:
        # TODO: 脱敏文本，返回 (脱敏后文本, findings列表)
        pass

    def get_supported_types(self) -> list[dict]:
        # TODO: 返回支持的 PII 类型及描述
        pass


# ====== Pydantic 模型 ======
class DetectRequest(BaseModel):
    # TODO
    pass


class DetectResponse(BaseModel):
    # TODO
    pass


class MaskRequest(BaseModel):
    # TODO
    pass


class MaskResponse(BaseModel):
    # TODO
    pass


# ====== FastAPI 应用 ======
app = FastAPI(title="PII Service", version="1.0.0")
detector = PIIDetector()


@app.post("/v1/pii/detect", response_model=DetectResponse)
async def detect_pii(request: DetectRequest):
    # TODO
    pass


@app.post("/v1/pii/mask", response_model=MaskResponse)
async def mask_pii(request: MaskRequest):
    # TODO
    pass


@app.get("/v1/pii/types")
async def list_types():
    # TODO
    pass


@app.get("/health")
async def health():
    # TODO
    pass


# ====== 测试 ======
def test_detect_phone():
    # TODO: 使用 TestClient 测试
    pass


def test_mask_email():
    # TODO
    pass


# 更多测试用例...


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

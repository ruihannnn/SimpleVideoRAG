import os
import logging
import warnings
import multiprocessing

warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.WARNING)

from videorag import VideoRAG, QueryParam
from videorag._llm import deepseek_bge_config


def build_bailian_vlm_addon():
    api_key = os.environ.get("BAILIAN_API_KEY", "sk-***")
    base_url = os.environ.get("BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    model = os.environ.get("VLM_MODEL", "qwen-vl-plus")
    vlm_cfg = {
        "provider": "bailian",
        "model": model,
        "api_key": api_key,
        "base_url": base_url,
        "max_tokens": 4096,
        # 可按需追加："timeout": 60, "max_tokens": 1024, "temperature": 0.7
    }
    addon_params = {
        "use_vlm_reasoning": True,
        "vlm": vlm_cfg,
    }
    return addon_params


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    query = os.environ.get("VLM_QUERY", '视频的主要内容是什么？')

    param = QueryParam(mode="videorag")
    param.wo_reference = True  # 新版路径直接由 VLM 生成，参考标志无影响

    addon_params = build_bailian_vlm_addon()
    videorag = VideoRAG(llm=deepseek_bge_config, working_dir=f"./videorag-workdir", addon_params=addon_params)
    
    response = videorag.query(query=query, param=param)
    print(response)



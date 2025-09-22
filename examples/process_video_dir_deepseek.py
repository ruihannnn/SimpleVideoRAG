import os
import logging
import warnings
import multiprocessing
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.WARNING)


from videorag._llm import deepseek_bge_config
from videorag import VideoRAG, QueryParam


if __name__ == '__main__':
    # 必须的设置
    multiprocessing.set_start_method('spawn')

    # 视频目录和工作目录
    video_dir = '/root/autodl-fs/MMBench-Video/video'
    work_base_dir = '/root/autodl-fs/MMBench-Video/video_work_dir'

    # 确保工作目录存在
    os.makedirs(work_base_dir, exist_ok=True)

    # 收集所有mp4视频文件
    video_files = [
        filename for filename in os.listdir(video_dir)
        if os.path.isfile(os.path.join(video_dir, filename)) and filename.lower().endswith('.mp4')
    ]

    # 遍历所有视频文件，使用tqdm进度条
    for filename in tqdm(video_files, desc="Processing videos"):
        video_path = os.path.join(video_dir, filename)

        # 针对每个视频，创建独立的工作目录
        video_name, _ = os.path.splitext(filename)
        video_work_dir = os.path.join(work_base_dir, video_name)
        os.makedirs(video_work_dir, exist_ok=True)

        # 初始化 VideoRAG，指定每个视频的工作目录
        videorag = VideoRAG(llm=deepseek_bge_config, working_dir=video_work_dir)

        # 处理当前视频
        videorag.insert_video(video_path_list=[video_path])
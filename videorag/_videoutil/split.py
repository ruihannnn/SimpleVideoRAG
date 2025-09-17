import os
import time
import shutil
import numpy as np
from tqdm import tqdm
from moviepy.video import fx as vfx
from moviepy.video.io.VideoFileClip import VideoFileClip
from .._utils import logger
import concurrent.futures

def split_video(
    video_path,
    working_dir,
    segment_length,
    num_frames_per_segment,
    audio_output_format='mp3',
):  
    """按固定时长切分视频，并为每段生成采样帧时间与音频文件。

    本函数会将输入视频按 `segment_length` 秒划分为若干子段：
    - 若最后一段长度小于 5 秒，则与上一段合并（避免产生过短片段）。
    - 为每个子段在全局时间轴上均匀采样 `num_frames_per_segment` 个时间点（用于后续抽帧/描述）。
    - 将每段音频提取为 `audio_output_format` 格式并写入缓存目录 `working_dir/_cache/<video_name>`。

    Args:
        video_path (str): 视频文件的绝对或相对路径。
        working_dir (str): 工作目录，用于保存缓存与索引文件。
        segment_length (int): 每个视频片段的目标长度（单位：秒）。
        num_frames_per_segment (int): 每段需均匀采样的帧时间点数量。
        audio_output_format (str): 导出音频的格式后缀，默认 'mp3'。

    Returns:
        tuple[dict[str, str], dict[str, dict]]: 返回二元组 `(segment_index2name, segment_times_info)`。
            - `segment_index2name`：索引到唯一段名的映射，形如 `{ "0": "<timestamp>-0-<start>-<end>", ... }`。
            - `segment_times_info`：每段的时间信息，形如 `{ "0": { "frame_times": np.ndarray, "timestamp": (start, end) }, ... }`，
              其中 `frame_times` 为全局时间轴上的采样时间（秒）。
    """
    unique_timestamp = str(int(time.time() * 1000))
    video_name = os.path.basename(video_path).split('.')[0]
    video_segment_cache_path = os.path.join(working_dir, '_cache', video_name)
    if os.path.exists(video_segment_cache_path):
        shutil.rmtree(video_segment_cache_path)
    os.makedirs(video_segment_cache_path, exist_ok=False)
    
    segment_index = 0
    segment_index2name, segment_times_info = {}, {}
    with VideoFileClip(video_path) as video:
    
        total_video_length = int(video.duration)
        start_times = list(range(0, total_video_length, segment_length))
        # if the last segment is shorter than 5 seconds, we merged it to the last segment
        if len(start_times) > 1 and (total_video_length - start_times[-1]) < 5:
            start_times = start_times[:-1]
        
        for start in tqdm(start_times, desc=f"Spliting Video {video_name}"):
            if start != start_times[-1]:
                end = min(start + segment_length, total_video_length)
            else:
                end = total_video_length
            
            subvideo = video.subclip(start, end)
            subvideo_length = subvideo.duration
            frame_times = np.linspace(0, subvideo_length, num_frames_per_segment, endpoint=False)
            frame_times += start
            
            segment_index2name[f"{segment_index}"] = f"{unique_timestamp}-{segment_index}-{start}-{end}"
            segment_times_info[f"{segment_index}"] = {"frame_times": frame_times, "timestamp": (start, end)}
            
            # save audio
            audio_file_base_name = segment_index2name[f"{segment_index}"]
            audio_file = f'{audio_file_base_name}.{audio_output_format}'
            try:
                subaudio = subvideo.audio
                subaudio.write_audiofile(os.path.join(video_segment_cache_path, audio_file), codec='mp3', verbose=False, logger=None)
            except Exception as e:
                logger.warning(f"Warning: Failed to extract audio for video {video_name} ({start}-{end}). Probably due to lack of audio track.")

            segment_index += 1

    return segment_index2name, segment_times_info

def saving_video_segments(
    video_name,
    video_path,
    working_dir,
    segment_index2name,
    segment_times_info,
    error_queue,
    video_output_format='mp4',
):
    """保存视频片段到缓存目录。

    本函数会遍历 `segment_index2name` 中的每个片段，并使用 `video_path` 作为源视频，
    将每个片段按 `segment_times_info` 中的时间信息切分，并保存到缓存目录 `working_dir/_cache/<video_name>`。

    Args:
        video_name (str): 视频文件名。
        video_path (str): 视频文件的绝对或相对路径。
        working_dir (str): 工作目录，用于保存缓存与索引文件。
        segment_index2name (dict[str, str]): 索引到唯一段名的映射，形如 `{ "0": "<timestamp>-0-<start>-<end>", ... }`。
        segment_times_info (dict[str, dict]): 每段的时间信息，形如 `{ "0": { "frame_times": np.ndarray, "timestamp": (start, end) }, ... }`，
              其中 `frame_times` 为全局时间轴上的采样时间（秒）。
        error_queue (Queue): 错误队列，用于记录处理过程中发生的错误。
        video_output_format (str): 导出视频的格式后缀，默认 'mp4'。
    """
    try:
        with VideoFileClip(video_path) as video:
            video_segment_cache_path = os.path.join(working_dir, '_cache', video_name)
            for index in tqdm(segment_index2name, desc=f"Saving Video Segments {video_name}"):
                start, end = segment_times_info[index]["timestamp"][0], segment_times_info[index]["timestamp"][1]
                video_file = f'{segment_index2name[index]}.{video_output_format}'
                subvideo = video.subclip(start, end)
                subvideo.write_videofile(os.path.join(video_segment_cache_path, video_file), codec='libx264', verbose=False, logger=None)
    except Exception as e:
        error_queue.put(f"Error in saving_video_segments:\n {str(e)}")
        raise RuntimeError
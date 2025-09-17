import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from moviepy.video.io.VideoFileClip import VideoFileClip
import concurrent.futures

def encode_video(video, frame_times):
    frames = []
    for t in frame_times:
        frames.append(video.get_frame(t))
    frames = np.stack(frames, axis=0)
    frames = [Image.fromarray(v.astype('uint8')).resize((1280, 720)) for v in frames]
    return frames
    
def segment_caption(video_name, video_path, segment_index2name, transcripts, segment_times_info, caption_result, error_queue):
    """
    使用 MiniCPM-V-2_6-int4 模型进行视频片段描述。

    本函数会遍历 `segment_index2name` 中的每个片段，并使用 `transcripts` 作为转录结果，
    将每个片段的描述结果保存到 `caption_result`。

    Args:
        video_name (str): 视频文件名。
        video_path (str): 视频文件的绝对或相对路径。
        segment_index2name (dict[str, str]): 索引到唯一段名的映射，形如 `{ "0": "<timestamp>-0-<start>-<end>", ... }`。
        transcripts (dict[str, str]): 每个片段的转录结果，形如 `{ "0": "<timestamp>-0-<start>-<end> 转录结果", ... }`。
        segment_times_info (dict[str, dict]): 每段的时间信息，形如 `{ "0": { "frame_times": np.ndarray, "timestamp": (start, end) }, ... }`，
              其中 `frame_times` 为全局时间轴上的采样时间（秒）。
        caption_result (dict[str, str]): 每个片段的描述结果，形如 `{ "0": "<timestamp>-0-<start>-<end> 描述结果", ... }`。
        error_queue (Queue): 错误队列，用于记录处理过程中发生的错误。
    """
    try:
        caption_model_path = os.environ.get("CAPTION_MODEL_PATH", "./MiniCPM-V-2_6-int4")
        model = AutoModel.from_pretrained(caption_model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(caption_model_path, trust_remote_code=True)
        model.eval()
        
        with VideoFileClip(video_path) as video:
            for index in tqdm(segment_index2name, desc=f"Captioning Video {video_name}"):
                frame_times = segment_times_info[index]["frame_times"]
                video_frames = encode_video(video, frame_times)
                segment_transcript = transcripts[index]
                query = f"The transcript of the current video:\n{segment_transcript}.\nNow provide a description (caption) of the video in English."
                msgs = [{'role': 'user', 'content': video_frames + [query]}]
                params = {}
                params["use_image_id"] = False
                params["max_slice_nums"] = 2
                segment_caption = model.chat(
                    image=None,
                    msgs=msgs,
                    tokenizer=tokenizer,
                    **params
                )
                caption_result[index] = segment_caption.replace("\n", "").replace("<|endoftext|>", "")
                torch.cuda.empty_cache()
    except Exception as e:
        error_queue.put(f"Error in segment_caption:\n {str(e)}")
        raise RuntimeError

def merge_segment_information(segment_index2name, segment_times_info, transcripts, captions):
    """
    合并视频片段的描述结果和转录结果。

    Args:
        segment_index2name (dict[str, str]): 索引到唯一段名的映射，形如 `{ "0": "<timestamp>-0-<start>-<end>", ... }`。
        segment_times_info (dict[str, dict]): 每段的时间信息，形如 `{ "0": { "frame_times": np.ndarray, "timestamp": (start, end) }, ... }`，
              其中 `frame_times` 为全局时间轴上的采样时间（秒）。
        transcripts (dict[str, str]): 每个片段的转录结果，形如 `{ "0": "<timestamp>-0-<start>-<end> 转录结果", ... }`。
        captions (dict[str, str]): 每个片段的描述结果，形如 `{ "0": "<timestamp>-0-<start>-<end> 描述结果", ... }`。

    Returns:
        dict[str, dict]: 返回每个片段的描述结果和转录结果，形如 `{ "0": {"content": "<timestamp>-0-<start>-<end> 描述结果", "time": "<timestamp>-0-<start>-<end>", "transcript": "<timestamp>-0-<start>-<end> 转录结果", "frame_times": np.ndarray} }`。
    """
    inserting_segments = {}
    for index in segment_index2name:
        inserting_segments[index] = {"content": None, "time": None}
        segment_name = segment_index2name[index]
        inserting_segments[index]["time"] = '-'.join(segment_name.split('-')[-2:])
        inserting_segments[index]["content"] = f"Caption:\n{captions[index]}\nTranscript:\n{transcripts[index]}\n\n"
        inserting_segments[index]["transcript"] = transcripts[index]
        inserting_segments[index]["frame_times"] = segment_times_info[index]["frame_times"].tolist()
    return inserting_segments
        
def retrieved_segment_caption(caption_model, caption_tokenizer, refine_knowledge, retrieved_segments, video_path_db, video_segments, num_sampled_frames):
    # model = AutoModel.from_pretrained('./MiniCPM-V-2_6-int4', trust_remote_code=True)
    # tokenizer = AutoTokenizer.from_pretrained('./MiniCPM-V-2_6-int4', trust_remote_code=True)
    # model.eval()
    
    caption_result = {}
    for this_segment in tqdm(retrieved_segments, desc='Captioning Segments for Given Query'):
        video_name = '_'.join(this_segment.split('_')[:-1])
        index = this_segment.split('_')[-1]
        video_path = video_path_db._data[video_name]
        timestamp = video_segments._data[video_name][index]["time"].split('-')
        start, end = eval(timestamp[0]), eval(timestamp[1])
        video = VideoFileClip(video_path)
        frame_times = np.linspace(start, end, num_sampled_frames, endpoint=False)
        video_frames = encode_video(video, frame_times)
        segment_transcript = video_segments._data[video_name][index]["transcript"]
        # query = f"The transcript of the current video:\n{segment_transcript}.\nGiven a question: {query}, you have to extract relevant information from the video and transcript for answering the question."
        query = f"The transcript of the current video:\n{segment_transcript}.\nNow provide a very detailed description (caption) of the video in English and extract relevant information about: {refine_knowledge}'"
        msgs = [{'role': 'user', 'content': video_frames + [query]}]
        params = {}
        params["use_image_id"] = False
        params["max_slice_nums"] = 2
        segment_caption = caption_model.chat(
            image=None,
            msgs=msgs,
            tokenizer=caption_tokenizer,
            **params
        )
        this_caption = segment_caption.replace("\n", "").replace("<|endoftext|>", "")
        caption_result[this_segment] = f"Caption:\n{this_caption}\nTranscript:\n{segment_transcript}\n\n"
        torch.cuda.empty_cache()
    
    return caption_result
import os
import torch
import logging
from tqdm import tqdm
from faster_whisper import WhisperModel
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

def speech_to_text(video_name, working_dir, segment_index2name, audio_output_format):
    """
    使用 Faster Whisper 模型进行语音转文本。

    本函数会遍历 `segment_index2name` 中的每个片段，并使用 `audio_output_format` 作为音频格式，
    将每个片段的音频文件转录为文本，并保存到缓存目录 `working_dir/_cache/<video_name>`。

    Args:
        video_name (str): 视频文件名。
        working_dir (str): 工作目录，用于保存缓存与索引文件。
        segment_index2name (dict[str, str]): 索引到唯一段名的映射，形如 `{ "0": "<timestamp>-0-<start>-<end>", ... }`。
        audio_output_format (str): 导出音频的格式后缀，默认 'mp3'。

    Returns:
        dict[str, str]: 返回每个片段的转录结果，形如 `{ "0": "<timestamp>-0-<start>-<end> 转录结果", ... }`。
    """
    asr_model_path = os.environ.get("ASR_MODEL_PATH", "./faster-distil-whisper-large-v3")
    model = WhisperModel(asr_model_path)
    model.logger.setLevel(logging.WARNING)
    
    cache_path = os.path.join(working_dir, '_cache', video_name)
    
    transcripts = {}
    for index in tqdm(segment_index2name, desc=f"Speech Recognition {video_name}"):
        segment_name = segment_index2name[index]
        audio_file = os.path.join(cache_path, f"{segment_name}.{audio_output_format}")

        # if the audio file does not exist, skip it
        if not os.path.exists(audio_file):
            transcripts[index] = ""
            continue
        
        segments, info = model.transcribe(audio_file)
        result = ""
        for segment in segments:
            result += "[%.2fs -> %.2fs] %s\n" % (segment.start, segment.end, segment.text)
        transcripts[index] = result
    
    return transcripts
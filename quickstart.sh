# deepseek and siliconflow api key
export DEEPSEEK_API_KEY="sk-<your-api-key>"
export SILICONFLOW_API_KEY="sk-<your-api-key>"

# asr and caption model path
export ASR_MODEL_PATH=<faster-distil-whisper-large-v3-path>
export CAPTION_MODEL_PATH=<MiniCPM-V-2_6-int4-path>

# python path
export PYTHONPATH=<your-project-path>:$PYTHONPATH

# cd to project path
cd <your-project-path>

python examples/process_videos_deepseek.py
python examples/query_videos_deepseek.py

import re
import json
import asyncio
import tiktoken
from typing import Union
from collections import defaultdict
from ._splitter import SeparatorSplitter
from ._utils import (
    logger,
    list_of_list_to_csv,
    truncate_list_by_token_size,
    compute_mdhash_id,
)
from .base import (
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
    QueryParam,
)
from .prompt import PROMPTS
from ._videoutil import (
    retrieved_segment_caption,
)

def chunking_by_token_size(
    tokens_list: list[list[int]],
    doc_keys,
    tiktoken_model,
    overlap_token_size=128,
    max_token_size=1024,
):
    """按固定窗口将 tokens 列表切分为带重叠的文本块。

    对每个文档使用大小为 ``max_token_size``、重叠为 ``overlap_token_size`` 的滑动窗口分块。
    每个分块会通过 ``tiktoken_model`` 解码为文本，并附带相关元信息。

    Args:
        tokens_list: 文档级的 tokens 列表（每个元素是一篇文档的 tokens）。
        doc_keys: 与 ``tokens_list`` 一一对应的文档标识列表。
        tiktoken_model: 用于将 tokens 解码为文本的 tiktoken 编解码器/模型。
        overlap_token_size: 相邻分块之间的 token 重叠数。
        max_token_size: 单个分块允许的最大 token 数。

    Returns:
        list[dict]: 分块字典列表，包含以下字段：
            - ``tokens``: 分块中 token 的数量。
            - ``content``: 分块解码后的文本内容。
            - ``chunk_order_index``: 该分块在其所属文档中的序号。
            - ``full_doc_id``: 分块对应的文档标识。
    """

    results = []
    for index, tokens in enumerate(tokens_list):
        chunk_token = []
        lengths = []
        for start in range(0, len(tokens), max_token_size - overlap_token_size):

            chunk_token.append(tokens[start : start + max_token_size])
            lengths.append(min(max_token_size, len(tokens) - start))

        # here somehow tricky, since the whole chunk tokens is list[list[list[int]]] for corpus(doc(chunk)),so it can't be decode entirely
        chunk_token = tiktoken_model.decode_batch(chunk_token)
        for i, chunk in enumerate(chunk_token):

            results.append(
                {
                    "tokens": lengths[i],
                    "content": chunk.strip(),
                    "chunk_order_index": i,
                    "full_doc_id": doc_keys[index],
                }
            )

    return results


def chunking_by_video_segments(
    tokens_list: list[list[int]],
    doc_keys,
    tiktoken_model,
    max_token_size=1024,
):
    """按顺序拼接相邻视频片段的 tokens，直至接近上限形成分块。

    按原有顺序将片段依次加入当前分块，直到即将超过 ``max_token_size`` 为止。
    每个完成的分块会被解码，并记录由哪些片段组成。

    Args:
        tokens_list: 已按顺序排列的视频片段 tokens 列表。
        doc_keys: 与 ``tokens_list`` 对齐的片段标识列表。
        tiktoken_model: 用于将 tokens 解码为文本的 tiktoken 编解码器/模型。
        max_token_size: 单个分块允许的最大 token 数。

    Returns:
        list[dict]: 分块字典列表，包含 ``tokens``、``content``、
        ``chunk_order_index`` 以及 ``video_segment_id``（组成该分块的片段 id 列表）。
    """
    # make sure each segment is not larger than max_token_size
    for index in range(len(tokens_list)):
        if len(tokens_list[index]) > max_token_size:
            tokens_list[index] = tokens_list[index][:max_token_size]
    
    results = []
    chunk_token = []
    chunk_segment_ids = []
    chunk_order_index = 0
    for index, tokens in enumerate(tokens_list):
        
        if len(chunk_token) + len(tokens) <= max_token_size:
            # add new segment
            chunk_token += tokens.copy()
            chunk_segment_ids.append(doc_keys[index])
        else:
            # save the current chunk
            chunk = tiktoken_model.decode(chunk_token)
            results.append(
                {
                    "tokens": len(chunk_token),
                    "content": chunk.strip(),
                    "chunk_order_index": chunk_order_index,
                    "video_segment_id": chunk_segment_ids,
                }
            )
            # new chunk with current segment as begin
            chunk_token = []
            chunk_segment_ids = []
            chunk_token += tokens.copy()
            chunk_segment_ids.append(doc_keys[index])
            chunk_order_index += 1
    
    # save the last chunk
    if len(chunk_token) > 0:
        chunk = tiktoken_model.decode(chunk_token)
        results.append(
            {
                "tokens": len(chunk_token),
                "content": chunk.strip(),
                "chunk_order_index": chunk_order_index,
                "video_segment_id": chunk_segment_ids,
            }
        )
    
    return results
    
    
def chunking_by_seperators(
    tokens_list: list[list[int]],
    doc_keys,
    tiktoken_model,
    overlap_token_size=128,
    max_token_size=1024,
):
    """基于分隔符进行带重叠的分块。

    使用 ``SeparatorSplitter``，结合 Prompt 中定义的分隔符 token 模式，
    对每个文档的 tokens 进行切分，并施加 ``chunk_overlap`` 与 ``chunk_size`` 约束。

    Args:
        tokens_list: 文档级的 tokens 列表。
        doc_keys: 每个文档对应的标识列表。
        tiktoken_model: 用于将 tokens 解码为文本的 tiktoken 编解码器/模型。
        overlap_token_size: 相邻分块之间的 token 重叠数。
        max_token_size: 单个分块允许的最大 token 数。

    Returns:
        list[dict]: 分块字典列表，含 ``tokens``、``content``、
        ``chunk_order_index`` 与 ``full_doc_id``。
    """

    splitter = SeparatorSplitter(
        separators=[
            tiktoken_model.encode(s) for s in PROMPTS["default_text_separator"]
        ],
        chunk_size=max_token_size,
        chunk_overlap=overlap_token_size,
    )
    results = []
    for index, tokens in enumerate(tokens_list):
        chunk_token = splitter.split_tokens(tokens)
        lengths = [len(c) for c in chunk_token]

        # here somehow tricky, since the whole chunk tokens is list[list[list[int]]] for corpus(doc(chunk)),so it can't be decode entirely
        chunk_token = tiktoken_model.decode_batch(chunk_token)
        for i, chunk in enumerate(chunk_token):

            results.append(
                {
                    "tokens": lengths[i],
                    "content": chunk.strip(),
                    "chunk_order_index": i,
                    "full_doc_id": doc_keys[index],
                }
            )

    return results


def get_chunks(new_videos, chunk_func=chunking_by_video_segments, **chunk_func_params):
    """使用给定的分块策略为视频生成文本分块。

    先用 tiktoken 模型对片段文本进行编码，再调用 ``chunk_func`` 产出分块，
    并将稳定的哈希 id 映射到分块内容上。

    Args:
        new_videos: 结构为 ``video_name -> {segment_index -> {"content": str}}`` 的映射。
        chunk_func: 对已编码文档进行分块的函数。
        **chunk_func_params: 传递给 ``chunk_func`` 的额外参数。

    Returns:
        dict[str, dict]: 从分块 id 到分块字典的映射。
    """
    inserting_chunks = {}

    new_videos_list = list(new_videos.keys())
    for video_name in new_videos_list:
        segment_id_list = list(new_videos[video_name].keys())
        docs = [new_videos[video_name][index]["transcript"] for index in segment_id_list]
        doc_keys = [f'{video_name}_{index}' for index in segment_id_list]

        ENCODER = tiktoken.encoding_for_model("gpt-4o")
        tokens = ENCODER.encode_batch(docs, num_threads=16)
        chunks = chunk_func(
            tokens, doc_keys=doc_keys, tiktoken_model=ENCODER, **chunk_func_params
        )

        for chunk in chunks:
            inserting_chunks.update(
                {compute_mdhash_id(chunk["content"], prefix="chunk-"): chunk}
            )

    return inserting_chunks



async def _extract_keywords_query(
    query,
    query_param: QueryParam,
    global_config: dict,
):
    """从查询中提取用于字幕/描述生成的关键关键词。

    调用较廉价的 LLM 获取关键词，以指导视频片段的字幕选择与生成。

    Args:
        query: 原始用户查询。
        query_param: 控制检索行为的查询参数。
        global_config: 全局配置，包含各类 LLM 调用函数。

    Returns:
        str: 模型返回的关键词字符串或短语列表。
    """
    use_llm_func: callable = global_config["llm"]["cheap_model_func"]
    keywords_prompt = PROMPTS["keywords_extraction"]
    keywords_prompt = keywords_prompt.format(input_text=query)
    final_result = await use_llm_func(keywords_prompt)
    return final_result

async def videorag_query(
    query,
    text_chunks_db,
    chunks_vdb,
    video_path_db,
    video_segments,
    video_segment_feature_vdb,
    caption_model,
    caption_tokenizer,
    query_param: QueryParam,
    global_config: dict,
) -> str:
    """结合文本分块与视觉检索，回答开放式问题。

    流程包括：检索文本分块、视觉检索、生成聚焦字幕，并将文本与视频证据组合成最终回答。

    Args:
        query: 用户问题。
        text_chunks_db: 文本分块内容的键值存储。
        chunks_vdb: 文本分块向量检索库。
        video_path_db: 视频名称到文件路径的键值存储。
        video_segments: 视频片段元数据与粗粒度字幕的存储。
        video_segment_feature_vdb: 片段级特征的向量检索库。
        caption_model: 用于字幕生成的视觉语言模型。
        caption_tokenizer: 与字幕模型配套的分词器/标记器。
        query_param: 包含 top-k 与 token 限制等的查询参数。
        global_config: 全局配置，包含 LLM 可调用函数与其他控制项。

    Returns:
        str: 生成的自然语言回答。
    """
    use_model_func = global_config["llm"]["best_model_func"]
    query = query
    
    # naive chunks
    results = await chunks_vdb.query(query, top_k=query_param.top_k)
    if not len(results):
        return PROMPTS["fail_response"]
    chunks_ids = [r["id"] for r in results]
    chunks = await text_chunks_db.get_by_ids(chunks_ids)

    maybe_trun_chunks = truncate_list_by_token_size(
        chunks,
        key=lambda x: x["content"],
        max_token_size=query_param.naive_max_token_for_text_unit,
    )
    logger.info(f"Truncate {len(chunks)} to {len(maybe_trun_chunks)} chunks")
    section = "-----New Chunk-----\n".join([c["content"] for c in maybe_trun_chunks])
    retreived_chunk_context = section
    
    # visual retrieval
    segment_results = await video_segment_feature_vdb.query(query)
    visual_retrieved_segments = {n['__id__'] for n in segment_results} if len(segment_results) else set()
    
    # caption
    retrieved_segments = list(visual_retrieved_segments)
    retrieved_segments = sorted(
        retrieved_segments,
        key=lambda x: (
            '_'.join(x.split('_')[:-1]), # video_name
            eval(x.split('_')[-1]) # index
        )
    )
    print(query)
    print(f"Retrieved Visual Segments {visual_retrieved_segments}")
    
    # 直接使用所有检索到的视频片段，不再进行LLM过滤
    remain_segments = retrieved_segments
    print(f"Using all {len(remain_segments)} retrieved segments")
    
    # visual retrieval
    keywords_for_caption = await _extract_keywords_query(
        query,
        query_param,
        global_config,
    )
    print(f"Keywords: {keywords_for_caption}")
    caption_results = retrieved_segment_caption(
        caption_model,
        caption_tokenizer,
        keywords_for_caption,
        remain_segments,
        video_path_db,
        video_segments,
        num_sampled_frames=global_config['fine_num_frames_per_segment']
    )

    ## data table
    text_units_section_list = [["video_name", "start_time", "end_time", "content"]]
    for s_id in caption_results:
        video_name = '_'.join(s_id.split('_')[:-1])
        index = s_id.split('_')[-1]
        start_time = eval(video_segments._data[video_name][index]["time"].split('-')[0])
        end_time = eval(video_segments._data[video_name][index]["time"].split('-')[1])
        start_time = f"{start_time // 3600}:{(start_time % 3600) // 60}:{start_time % 60}"
        end_time = f"{end_time // 3600}:{(end_time % 3600) // 60}:{end_time % 60}"
        text_units_section_list.append([video_name, start_time, end_time, caption_results[s_id]])
    text_units_context = list_of_list_to_csv(text_units_section_list)

    retreived_video_context = f"\n-----Retrieved Knowledge From Videos-----\n```csv\n{text_units_context}\n```\n"
    
    if query_param.wo_reference:
        sys_prompt_temp = PROMPTS["videorag_response_wo_reference"]
    else:
        sys_prompt_temp = PROMPTS["videorag_response"]
        
    sys_prompt = sys_prompt_temp.format(
        video_data=retreived_video_context,
        chunk_data=retreived_chunk_context,
        response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    return response

async def videorag_query_multiple_choice(
    query,
    text_chunks_db,
    chunks_vdb,
    video_path_db,
    video_segments,
    video_segment_feature_vdb,
    caption_model,
    caption_tokenizer,
    query_param: QueryParam,
    global_config: dict,
) -> str:
    """结合文本与视觉证据，回答选择题（MCQ）。

    与 ``videorag_query`` 类似，但面向选择题：构建证据、
    使用专门的选择题 Prompt，并强制输出包含答案与简短解释的 JSON 结构。

    Args:
        query: 用户的选择题问题（含选项）。
        text_chunks_db: 文本分块内容的键值存储。
        chunks_vdb: 文本分块向量检索库。
        video_path_db: 视频名称到文件路径的键值存储。
        video_segments: 视频片段元数据与粗粒度字幕的存储。
        video_segment_feature_vdb: 片段级特征的向量检索库。
        caption_model: 用于字幕生成的视觉语言模型。
        caption_tokenizer: 与字幕模型配套的分词器/标记器。
        query_param: 包含 top-k 与 token 限制等的查询参数。
        global_config: 全局配置，包含 LLM 可调用函数与其他控制项。

    Returns:
        dict: 至少包含 ``Answer`` 与 ``Explanation`` 的 JSON 风格字典。
    """
    use_model_func = global_config["llm"]["best_model_func"]
    query = query
    
    # naive chunks
    results = await chunks_vdb.query(query, top_k=query_param.top_k)
    # NOTE: I update here, not len results can also process
    if len(results):
        chunks_ids = [r["id"] for r in results]
        chunks = await text_chunks_db.get_by_ids(chunks_ids)

        maybe_trun_chunks = truncate_list_by_token_size(
            chunks,
            key=lambda x: x["content"],
            max_token_size=query_param.naive_max_token_for_text_unit,
        )
        logger.info(f"Truncate {len(chunks)} to {len(maybe_trun_chunks)} chunks")
        section = "-----New Chunk-----\n".join([c["content"] for c in maybe_trun_chunks])
        retreived_chunk_context = section
    else:
        retreived_chunk_context = "No Content"
        
    # visual retrieval
    segment_results = await video_segment_feature_vdb.query(query)
    visual_retrieved_segments = {n['__id__'] for n in segment_results} if len(segment_results) else set()
    
    # caption
    retrieved_segments = list(visual_retrieved_segments)
    retrieved_segments = sorted(
        retrieved_segments,
        key=lambda x: (
            '_'.join(x.split('_')[:-1]), # video_name
            eval(x.split('_')[-1]) # index
        )
    )
    print(query)
    print(f"Retrieved Visual Segments {visual_retrieved_segments}")
    
    # 直接使用所有检索到的视频片段，不再进行LLM过滤
    remain_segments = retrieved_segments
    print(f"Using all {len(remain_segments)} retrieved segments")
    
    # visual retrieval
    keywords_for_caption = await _extract_keywords_query(
        query,
        query_param,
        global_config,
    )
    print(f"Keywords: {keywords_for_caption}")
    caption_results = retrieved_segment_caption(
        caption_model,
        caption_tokenizer,
        keywords_for_caption,
        remain_segments,
        video_path_db,
        video_segments,
        num_sampled_frames=global_config['fine_num_frames_per_segment']
    )

    ## data table
    text_units_section_list = [["video_name", "start_time", "end_time", "content"]]
    for s_id in caption_results:
        video_name = '_'.join(s_id.split('_')[:-1])
        index = s_id.split('_')[-1]
        start_time = eval(video_segments._data[video_name][index]["time"].split('-')[0])
        end_time = eval(video_segments._data[video_name][index]["time"].split('-')[1])
        start_time = f"{start_time // 3600}:{(start_time % 3600) // 60}:{start_time % 60}"
        end_time = f"{end_time // 3600}:{(end_time % 3600) // 60}:{end_time % 60}"
        text_units_section_list.append([video_name, start_time, end_time, caption_results[s_id]])
    text_units_context = list_of_list_to_csv(text_units_section_list)

    retreived_video_context = f"\n-----Retrieved Knowledge From Videos-----\n```csv\n{text_units_context}\n```\n"
    
    # NOTE: I update here to use a different prompt
    sys_prompt_temp = PROMPTS["videorag_response_for_multiple_choice_question"]
        
    sys_prompt = sys_prompt_temp.format(
        video_data=retreived_video_context,
        chunk_data=retreived_chunk_context,
        response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        use_cache=False,
    )
    while True:
        try:
            json_response = json.loads(response)
            assert "Answer" in json_response and "Explanation" in json_response
            return json_response
        except Exception as e:
            logger.info(f"Response is not valid JSON for query {query}. Found {e}. Retrying...")
            response = await use_model_func(
                query,
                system_prompt=sys_prompt,
                use_cache=False,
            )
    
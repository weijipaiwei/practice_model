
from email import message
from functools import cache
import os
import random
import torch
import time
import numpy as np
import torch.nn.functional as F
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers.cache_utils import DynamicCache
from qwen_vl_utils import process_vision_info


def logits_to_next_token_id(
    prompt_len,
    input_ids,
    logits, 
    temperature, 
    repeat_penalty, 
    repeat_penalty_length,
    no_repeat_ngram_size,
    topk,
    topp
):
    # 1. 取出最新预测的token，并施加温度
    next_token_logits = logits[:, -1, :].clone()
    next_token_logits = next_token_logits / temperature

    # 2. 重复惩罚，最近出现过的词，logits变小
    if repeat_penalty != 1.0:
        prev_tokens = input_ids[0, -repeat_penalty_length:] if input_ids.shape[1] > repeat_penalty_length else input_ids[0]
        score_tensor = torch.ones_like(next_token_logits)
        score_tensor.scatter_(1, prev_tokens.unsqueeze(0), repeat_penalty)
        next_token_logits = torch.where(next_token_logits>0, next_token_logits/score_tensor, next_token_logits*score_tensor)

    # 3. ngrams惩罚，减少重复模式出现的概率
    if no_repeat_ngram_size > 0:
        banned_tokens = []
        generated_ids = input_ids[0, prompt_len:]
        if len(generated_ids) >= no_repeat_ngram_size - 1:
            ngrams = [tuple(generated_ids[i: i+no_repeat_ngram_size-1]) for i in range(len(generated_ids) - no_repeat_ngram_size + 2)]
            prefix = tuple(generated_ids[-(no_repeat_ngram_size-1):].tolist())
            for ngram in ngrams[:-1]:
                if ngram == prefix:
                    banned_tokens.append(generated_ids[ngrams.index(ngram) + no_repeat_ngram_size -1].item())
        if banned_tokens:
            next_token_logits[0, banned_tokens] = -float('inf')
    next_token_probs = F.softmax(next_token_logits, dim=-1)

    # 4. 使用top k筛选
    topk_probs, topk_indices = torch.topk(next_token_probs, topk)

    # 5. 使用top p进一步筛选
    cumulative_probs = torch.cumsum(topk_probs, dim=-1)

    sorted_indices_to_remove = cumulative_probs > topp
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    probs_to_keep = topk_probs.clone()
    probs_to_keep[sorted_indices_to_remove] = 0.0

    # 6. top k和top p破坏了概率的分布，因此重新归一化
    final_probs = probs_to_keep / probs_to_keep.sum(dim=-1, keepdim=True)

    # 7. 选取下一个token id
    sampled_relative_index = torch.multinomial(final_probs, num_samples=1)
    next_token_id = torch.gather(topk_indices, -1, sampled_relative_index)
    return next_token_id



def generation_v4_topk_topp_repeatPenalty_ngrams_kvCache_mllm_instruct(
    model, 
    processor, 
    messages, 
    max_new_tokens, 
    temperature=1e-6, 
    topk=50, 
    topp=1.0, 
    repeat_penalty=1.05, 
    repeat_penalty_length=1,
    eos_token_ids=None,
    no_repeat_ngram_size=0
):
    device = model.device
    # 1. 使用chat template处理messages，得到模型输入的文本
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # 2. 使用process_vision_info载入messages中出现过的图片和视频
    image_inputs, video_inputs = process_vision_info(messages)
    
    # 3. 将文本嵌入为token ids，并且为图片和视频token留出相应的位置
    inputs = processor(
        text = [text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )

    # 4. 准备好prefilling阶段模型的输入
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    pixel_values = inputs.pixel_values.to(device)
    image_grid_thw = inputs.image_grid_thw.to(device)
    cache_position = torch.arange(0,input_ids.shape[1], dtype=torch.long, device=device)
    past_key_values = DynamicCache()
    
    if eos_token_ids is None:
        eos_token_ids = [processor.tokenizer.eos_token_id, processor.tokenizer.pad_token_id]

    all_token_ids = input_ids.clone()
    prompt_len = input_ids.shape[1]

    # 5. 进行第一次forward，完成prefilling
    with torch.no_grad():
        outputs = model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            image_grid_thw=image_grid_thw,
            use_cache=True,
            past_key_values=past_key_values,
            cache_position=cache_position,
            pixel_values=pixel_values,
            return_dict = True
        )
    logits = outputs.logits

    # 6. 准备好进入循环所需要的参数
    past_key_values = outputs.past_key_values
    next_token_id = logits_to_next_token_id(
        prompt_len = prompt_len,
        input_ids = input_ids,
        logits = logits, 
        temperature = temperature, 
        repeat_penalty = repeat_penalty, 
        repeat_penalty_length = repeat_penalty_length,
        no_repeat_ngram_size = no_repeat_ngram_size,
        topk = topk,
        topp = topp
    )
    all_token_ids = torch.cat([all_token_ids, next_token_id], dim=1)
    cache_position = torch.tensor([all_token_ids.shape[1]-1], dtype = torch.long, device=device)
    attention_mask = torch.cat([attention_mask, torch.ones([1,1], device=device, dtype=torch.long)], dim=-1)

    # 7. 预测下一个token
    for _ in range(max_new_tokens-1):
        with torch.no_grad():
            outputs = model(
                input_ids = next_token_id,
                attention_mask = attention_mask,
                image_grid_thw = image_grid_thw,
                use_cache = True,
                past_key_values = past_key_values,
                cache_position = cache_position,
                return_dict = True
            )

        logits = outputs.logits
        past_key_values = outputs.past_key_values

        next_token_id = logits_to_next_token_id(
            prompt_len = prompt_len,
            input_ids = all_token_ids,
            logits = logits, 
            temperature = temperature, 
            repeat_penalty = repeat_penalty, 
            repeat_penalty_length = repeat_penalty_length,
            no_repeat_ngram_size = no_repeat_ngram_size,
            topk = topk,
            topp = topp
        )

        all_token_ids = torch.cat([all_token_ids, next_token_id], dim=1)
        cache_position = torch.tensor([all_token_ids.shape[1]-1], dtype = torch.long, device=device)
        attention_mask = torch.cat([attention_mask, torch.ones([1,1], device=device, dtype=torch.long)], dim=-1)
        

        if next_token_id.item() in eos_token_ids:
            print("已生成结束符，停止生成。")
            break

    return processor.tokenizer.decode(all_token_ids[0][prompt_len:])


if __name__ == "__main__":

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
    ###################################################################################################################
    # text = processor.apply_chat_template(
    #     messages, tokenize=False, add_generation_prompt=True
    # )
    # image_inputs, video_inputs = process_vision_info(messages)
    # inputs = processor(
    #     text=[text],
    #     images=image_inputs,
    #     videos=video_inputs,
    #     padding=True,
    #     return_tensors="pt",
    # )
    # inputs = inputs.to("cuda")

    # generated_ids = model.generate(**inputs, max_new_tokens=128)
    # generated_ids_trimmed = [
    #     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    # ]
    # output_text = processor.batch_decode(
    #     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    # )
    # print(output_text)
    ###################################################################################################################

    generated_texts = generation_v4_topk_topp_repeatPenalty_ngrams_kvCache_mllm_instruct(
        model, 
        processor, 
        messages, 
        max_new_tokens=128, 
        temperature=1.0, 
        topk=50, 
        topp=0.85, 
        repeat_penalty=1.0, 
        repeat_penalty_length=64,
        eos_token_ids=None,
        no_repeat_ngram_size=0
    )

    print(generated_texts)


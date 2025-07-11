
import os
import random
import torch
import time
import numpy as np
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def set_seed(seed: int = 42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)


def generate_v1_gready(model, tokenizer, prompt, max_new_tokens):
    '''
    最简单的贪心搜索算法
    '''
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.cuda()
    attention_mask = inputs.attention_mask.cuda()

    for _ in range(max_new_tokens):
        outputs = model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            return_dict = True
        )

        logits = outputs.logits
        next_token_logits = logits[:, -1, :]

        next_token_id = torch.argmax(next_token_logits, dim=-1)
        input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones([1,1], device="cuda", dtype=torch.long)], dim=-1)

        if next_token_id.item() == tokenizer.eos_token_id:
            print("已生成结束符，停止生成。")
            break

    return tokenizer.decode(input_ids[0])


def generate_v2_sampling(model, tokenizer, prompt, max_new_tokens, temperature=1.0):
    '''
    引入温度与随机采样
    '''

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.cuda()
    attention_mask = inputs.attention_mask.cuda()

    for _ in range(max_new_tokens):
        outputs = model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            return_dict = True
        )

        logits = outputs.logits
        next_token_logits = logits[:, -1, :]

        ######################################################################################
        next_token_logits = next_token_logits / temperature
        probs = F.softmax(next_token_logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token_id], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones([1,1], device="cuda", dtype=torch.long)], dim=-1)
        ######################################################################################

        if next_token_id.item() == tokenizer.eos_token_id:
            print("已生成结束符，停止生成。")
            break

    return tokenizer.decode(input_ids[0])


def generate_v3_topk(model, tokenizer, prompt, max_new_tokens, temperature=1.0, topk=50):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.cuda()
    attention_mask = inputs.attention_mask.cuda()

    for _ in range(max_new_tokens):
        outputs = model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            return_dict = True
        )

        logits = outputs.logits
        next_token_logits = logits[:, -1, :]
        next_token_logits = next_token_logits / temperature
        next_token_probs = F.softmax(next_token_logits, dim=-1)

        ######################################################################################
        topk_probs, topk_indices = torch.topk(next_token_probs, topk)
        filtered_probs = torch.zeros_like(next_token_probs, dtype=next_token_probs.dtype)
        filtered_probs.scatter_(1, topk_indices, topk_probs)
        probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
        ######################################################################################

        
        next_token_id = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token_id], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones([1,1], device="cuda", dtype=torch.long)], dim=-1)
        

        if next_token_id.item() == tokenizer.eos_token_id:
            print("已生成结束符，停止生成。")
            break

    return tokenizer.decode(input_ids[0])


def generation_v4_topk_topp(model, tokenizer, prompt, max_new_tokens, temperature=1.0, topk=50, topp=0.7):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.cuda()
    attention_mask = inputs.attention_mask.cuda()

    for _ in range(max_new_tokens):
        outputs = model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            return_dict = True
        )

        logits = outputs.logits
        next_token_logits = logits[:, -1, :].clone()
        next_token_logits = next_token_logits / temperature
        next_token_probs = F.softmax(next_token_logits, dim=-1)

        ######################################################################################
        '''直观的实现'''
        # topk_probs, topk_indices = torch.topk(next_token_probs, topk)
        # filtered_probs = torch.zeros_like(next_token_probs, dtype=next_token_probs.dtype)
        # filtered_probs.scatter_(1, topk_indices, topk_probs)
        
        # sorted_probs, sorted_indices = torch.sort(filtered_probs, descending=True, dim=-1)
        # cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
 
        # sorted_indices_to_remove = cumulative_probs > topp
        # sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        # sorted_indices_to_remove[..., 0] = 0

        # filtered_probs.scatter_(-1, sorted_indices, sorted_probs * (~sorted_indices_to_remove).float())
        # probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
        # next_token_id = torch.multinomial(probs, num_samples=1)
        ######################################################################################


        ######################################################################################
        '''更高效的实现'''
        topk_probs, topk_indices = torch.topk(next_token_probs, topk)

        cumulative_probs = torch.cumsum(topk_probs, dim=-1)
 
        sorted_indices_to_remove = cumulative_probs > topp
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        probs_to_keep = topk_probs.clone()
        probs_to_keep[sorted_indices_to_remove] = 0.0

        final_probs = probs_to_keep / probs_to_keep.sum(dim=-1, keepdim=True)
        sampled_relative_index = torch.multinomial(final_probs, num_samples=1)

        next_token_id = torch.gather(topk_indices, -1, sampled_relative_index)
        ######################################################################################


        input_ids = torch.cat([input_ids, next_token_id], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones([1,1], device="cuda", dtype=torch.long)], dim=-1)
        

        if next_token_id.item() == tokenizer.eos_token_id:
            print("已生成结束符，停止生成。")
            break

    return tokenizer.decode(input_ids[0])



def generation_v4_topk_topp_repeatPenalty(model, tokenizer, prompt, max_new_tokens, temperature=1.0, topk=50, topp=0.7, repeat_penalty=1.2):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.cuda()
    attention_mask = inputs.attention_mask.cuda()

    for _ in range(max_new_tokens):
        outputs = model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            return_dict = True
        )

        logits = outputs.logits
        next_token_logits = logits[:, -1, :].clone()
        next_token_logits = next_token_logits / temperature

        ######################################################################################
        if repeat_penalty != 1.0:
            prev_tokens = input_ids[0]
            score_tensor = torch.ones_like(next_token_logits)
            score_tensor.scatter_(1, prev_tokens.unsqueeze(0), repeat_penalty)
            next_token_logits = torch.where(next_token_logits>0, next_token_logits/score_tensor, next_token_logits*score_tensor)
        next_token_probs = F.softmax(next_token_logits, dim=-1)


        topk_probs, topk_indices = torch.topk(next_token_probs, topk)
        cumulative_probs = torch.cumsum(topk_probs, dim=-1)
 
        sorted_indices_to_remove = cumulative_probs > topp
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        probs_to_keep = topk_probs.clone()
        probs_to_keep[sorted_indices_to_remove] = 0.0

        final_probs = probs_to_keep / probs_to_keep.sum(dim=-1, keepdim=True)
        sampled_relative_index = torch.multinomial(final_probs, num_samples=1)

        next_token_id = torch.gather(topk_indices, -1, sampled_relative_index)
        ######################################################################################


        input_ids = torch.cat([input_ids, next_token_id], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones([1,1], device="cuda", dtype=torch.long)], dim=-1)
        

        if next_token_id.item() == tokenizer.eos_token_id:
            print("已生成结束符，停止生成。")
            break

    return tokenizer.decode(input_ids[0])


def generation_v4_topk_topp_repeatPenalty_ngrams(
    model, 
    tokenizer, 
    prompt, 
    max_new_tokens, 
    temperature=1.0, 
    topk=50, 
    topp=0.7, 
    repeat_penalty=1.2, 
    repeat_penalty_length=64,
    eos_token_id=None,
    no_repeat_ngram_size=0
):
    inputs = tokenizer(prompt, return_tensors="pt")
    device = model.device
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id

    prompt_len = input_ids.shape[1]


    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(
                input_ids = input_ids,
                attention_mask = attention_mask,
                return_dict = True
            )

        logits = outputs.logits
        next_token_logits = logits[:, -1, :].clone()
        next_token_logits = next_token_logits / temperature

        ######################################################################################
        if repeat_penalty != 1.0:
            prev_tokens = input_ids[0, -repeat_penalty_length:] if input_ids.shape[1] > repeat_penalty_length else input_ids[0]
            score_tensor = torch.ones_like(next_token_logits)
            score_tensor.scatter_(1, prev_tokens.unsqueeze(0), repeat_penalty)
            next_token_logits = torch.where(next_token_logits>0, next_token_logits/score_tensor, next_token_logits*score_tensor)

        
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


        topk_probs, topk_indices = torch.topk(next_token_probs, topk)
        cumulative_probs = torch.cumsum(topk_probs, dim=-1)
 
        sorted_indices_to_remove = cumulative_probs > topp
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        probs_to_keep = topk_probs.clone()
        probs_to_keep[sorted_indices_to_remove] = 0.0

        final_probs = probs_to_keep / probs_to_keep.sum(dim=-1, keepdim=True)
        sampled_relative_index = torch.multinomial(final_probs, num_samples=1)

        next_token_id = torch.gather(topk_indices, -1, sampled_relative_index)
        ######################################################################################


        input_ids = torch.cat([input_ids, next_token_id], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones([1,1], device=device, dtype=torch.long)], dim=-1)
        

        if next_token_id.item() == tokenizer.eos_token_id:
            print("已生成结束符，停止生成。")
            break

    return tokenizer.decode(input_ids[0])


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
    next_token_logits = logits[:, -1, :].clone().float()
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



def generation_v4_topk_topp_repeatPenalty_ngrams_kvCache(
    model, 
    tokenizer, 
    prompt, 
    max_new_tokens, 
    temperature=1.0, 
    topk=50, 
    topp=0.7, 
    repeat_penalty=1.2, 
    repeat_penalty_length=64,
    eos_token_id=None,
    no_repeat_ngram_size=0
):
    # 1. 得到input ids和attention mask
    inputs = tokenizer(prompt, return_tensors="pt")
    device = model.device
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id

    # 记录每一步预测的token id
    all_token_ids = input_ids.clone()
    prompt_len = input_ids.shape[1]

    # 2. prefilling，运行第一次，得到kv cache
    with torch.no_grad():
        outputs = model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            use_cache=True,
            return_dict = True
        )
    logits = outputs.logits
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
    attention_mask = torch.cat([attention_mask, torch.ones([1,1], device=device, dtype=torch.long)], dim=-1)
    
    # 3. 自回归生成
    for _ in range(max_new_tokens-1):
        with torch.no_grad():
            outputs = model(
                input_ids = next_token_id,
                attention_mask = attention_mask,
                use_cache=True,
                past_key_values=past_key_values,
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
        attention_mask = torch.cat([attention_mask, torch.ones([1,1], device=device, dtype=torch.long)], dim=-1)
        

        if next_token_id.item() == tokenizer.eos_token_id:
            print("已生成结束符，停止生成。")
            break

    return tokenizer.decode(all_token_ids[0])





if __name__ == "__main__":

    # 定义要加载的模型名称
    set_seed(142)
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name, padding_side = "left")
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.cuda()

    print("模型和分词器加载成功！")
    prompt_text = "In a world where dragons fly and magic is real,"

    print("开始贪心搜索解码")
    generated_text = generate_v1_gready(model, tokenizer, prompt_text, max_new_tokens=512)
    print(f"使用贪心搜索算法，结果为：{generated_text}")

    print("开始温度+采样解码")
    generated_text = generate_v2_sampling(model, tokenizer, prompt_text, max_new_tokens=512, temperature=1.5)
    print(f"使用温度+采样的算法，结果为：{generated_text}")

    print("开始温度+采样+topk解码")
    generated_text = generate_v3_topk(model, tokenizer, prompt_text, max_new_tokens=512, temperature=1.5, topk=50)
    print(f"使用温度+采样+topk的算法，结果为：{generated_text}")

    print("开始温度+采样+topk+topp解码")
    generated_text = generation_v4_topk_topp(model, tokenizer, prompt_text, max_new_tokens=512, temperature=1.5, topk=50, topp=0.7)
    print(f"使用温度+采样+topk+topp的算法，结果为：{generated_text}")

    print("开始温度+采样+topk+topp+repeat penalty解码")
    generated_text = generation_v4_topk_topp_repeatPenalty(model, tokenizer, prompt_text, max_new_tokens=512, temperature=1.5, topk=50, topp=0.7, repeat_penalty=1.2)
    print(f"使用温度+采样+topk+topp+repeat penalty的算法，结果为：{generated_text}")


    print("开始温度+采样+topk+topp+repeat penalty+ngrams解码")
    generated_text = generation_v4_topk_topp_repeatPenalty_ngrams(
        model, tokenizer, prompt_text, max_new_tokens=512, 
        temperature=1., topk=50, topp=0.7, repeat_penalty=1.2,
        repeat_penalty_length=64,eos_token_id=None,no_repeat_ngram_size=8
    )
    print(f"使用温度+采样+topk+topp+repeat penalty+ngrams的算法，结果为：{generated_text}")


    print("开始温度+采样+topk+topp+repeat penalty+ngrams+kv cache解码")
    generated_text = generation_v4_topk_topp_repeatPenalty_ngrams_kvCache(
        model, tokenizer, prompt_text, max_new_tokens=512, 
        temperature=1.1, topk=80, topp=0.8, repeat_penalty=1.1,
        repeat_penalty_length=64,eos_token_id=None,no_repeat_ngram_size=8
    )
    print(f"使用温度+采样+topk+topp+repeat penalty+ngrams+kv cache的算法，结果为：{generated_text}")

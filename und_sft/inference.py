import re
import torch
from accelerate.utils import set_seed
from accelerate import Accelerator
import torch.nn.functional as F


from model import MyModel
from dataset import MyDataset_eval
from torch.utils.data import DataLoader
from transformers.cache_utils import DynamicCache
from qwen_vl_utils import process_vision_info
from tqdm import tqdm


from evaluation.evaluation import calculate_metrics_at_confidence_threshold


def check_bbox(bboxes):
    max_num = -1
    for bbox in bboxes:
        if bbox[0] + 9 > bbox[1]:
            return False
    return True


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
    input_ids,
    attention_mask,
    pixel_values,
    image_grid_thw,
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

    # 4. 准备好prefilling阶段模型的输入
    cache_position = torch.arange(0,input_ids.shape[1], dtype=torch.long, device=device)
    past_key_values = DynamicCache()
    
    if eos_token_ids is None:
        raise ValueError("eos_token_ids is None!")

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
            labels=None
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
                pixel_values = None,
                labels = None
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
            break

    return all_token_ids[0][prompt_len:]


def inference_main():

    model_path = "/slurm/home/yrd/kanlab/zhangchenfeng/program/practice_model/und_sft/outputs/result2/final_checkpoint_2000_steps"
    annotation_path = "/slurm/home/yrd/kanlab/zhangchenfeng/program/practice_model/und_sft/detection_dataset/test.jsonl"
    processor_path = "Qwen/Qwen2.5-VL-3B-Instruct"
    eval_batch_size = 1
    num_workers = 2
    seed = 42
    
    set_seed(seed)
    accelerator = Accelerator(mixed_precision="bf16")

    model = MyModel.from_pretrained(model_path)
    val_dataset = MyDataset_eval(annotation_path = annotation_path, processor_path = processor_path)
    eos_token_ids = val_dataset.eos_token_ids
    val_dataloader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, collate_fn=val_dataset.collate_fn)

    model, val_dataloader = accelerator.prepare(model, val_dataloader)

    results = []

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Inference"):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            pixel_values = batch["pixel_values"]
            image_grid_thw = batch["image_grid_thw"]
            # breakpoint()

            generated_ids_trimmed = []
            output_text = []

            generated_ids = generation_v4_topk_topp_repeatPenalty_ngrams_kvCache_mllm_instruct(
                model = model,
                input_ids = input_ids,
                attention_mask = attention_mask,
                pixel_values = pixel_values,
                image_grid_thw = image_grid_thw,
                max_new_tokens = 1024,
                temperature = 1e-6,
                topk = 50,
                topp = 1.0,
                repeat_penalty = 1.05,
                repeat_penalty_length = 1,  
                eos_token_ids = eos_token_ids,
                no_repeat_ngram_size = 0,
            )
            generated_ids_trimmed.append(generated_ids)
            output_text.append(val_dataset.processor.tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False))

            ###############################################################################################
            '''
                generated_ids_trimmed: 生成的ids，去掉input_ids
                output_text: 使用生成的ids，使用tokenizer.decode得到的文本（skip_special_tokens=True, clean_up_tokenization_spaces=False）
            '''

            bbox_start_idx = val_dataset.processor.tokenizer.encode("<|bbox_start|>")[0]
            bbox_end_idx = val_dataset.processor.tokenizer.encode("<|bbox_end|>")[0]

            for index, answer in enumerate(generated_ids_trimmed):
                answer_list = answer.cpu().tolist()
                bbox_start = None
                bbox_end = None
                bboxes = []
                for idx, generated_id in enumerate(answer_list):
                    if generated_id == bbox_start_idx:
                        bbox_start = idx
                    if generated_id == bbox_end_idx:
                        bbox_end = idx
                        bboxes.append((bbox_start, bbox_end))

                # breakpoint()
                if not check_bbox(bboxes):
                    print(f"bbox is not valid, llm answer: {output_text[index]}, skip this sample.")
                    continue

                sample_dict = {}
                sample_dict["image_id"] = batch['image_paths']
                sample_dict["obj_names"] = batch['obj_names']
                sample_dict["ground_truth"] = batch['positions_original']
                
                bbox_predictions = []
                for (bbox_start, bbox_end) in bboxes:
                    try:
                        bbox_str = val_dataset.processor.tokenizer.decode(generated_ids_trimmed[bbox_start+1:bbox_end])
                        matches = re.findall(r"\d+\.\d+", bbox_str)
                        bbox_predictions.append([float(match) for match in matches])
                    except:
                        print(f"bbox is not valid, llm answer: {bbox_str}, skip this bbox.")
                        continue

                if len(bbox_predictions) == 0:
                    print(f"no bbox is valid, llm answer: {output_text[index]}, skip this sample.")
                    continue

                sample_dict["predictions"] = bbox_predictions
                results.append(sample_dict)   
    
    ###########################################################################################################################################

    all_gts = []
    all_preds = []
    

    for sample_dict in results:
        image_id = sample_dict["image_id"]
        gt = sample_dict["ground_truth"]
        pred = sample_dict["predictions"]

        all_gts.append({image_id: gt})
        all_preds.append({image_id: pred})


    conf_threshold = 0.7
    p, r, f1 = calculate_metrics_at_confidence_threshold(all_gts, all_preds, conf_threshold=conf_threshold, iou_threshold=0.5)
    print(f"conf_threshold: {conf_threshold}, iou_threshold: 0.5")
    print(f"Precision: {p}, Recall: {r}, F1: {f1}")


if __name__ == "__main__":
    inference_main()




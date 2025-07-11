import re

import torch
from accelerate.utils import set_seed
from accelerate import Accelerator

from model import MyModel
from dataset import MyDataset_eval
from torch.utils.data import DataLoader

from evaluation.evaluation import calculate_metrics_at_confidence_threshold


def check_bbox(bboxes):
    max_num = -1
    for bbox in bboxes:
        if bbox[0] + 9 > bbox[1]:
            return False
        

    return True

def inference_main():

    model_path = "/slurm/home/yrd/kanlab/zhangchenfeng/program/practice_model/und_sft/outputs/result2/final_checkpoint_2000_steps"
    annotation_path = "/slurm/home/yrd/kanlab/zhangchenfeng/program/practice_model/und_sft/detection_dataset/test.jsonl"
    processor_path = "Qwen/Qwen2.5-VL-3B-Instruct"
    eval_batch_size = 32
    num_workers = 2
    seed = 42

    set_seed(seed)
    accelerator = Accelerator(mixed_precision="bf16")

    model = MyModel.from_pretrained(model_path)
    val_dataset = MyDataset_eval(annotation_path = annotation_path, processor_path = processor_path)
    val_dataloader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, collate_fn=val_dataset.collate_fn)

    model, val_dataloader = accelerator.prepare(model, val_dataloader)

    result_texts = []
    results = []

    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            pixel_values = batch["pixel_values"]
            image_grid_thw = batch["image_grid_thw"]

            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                max_new_tokens=1024,
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, generated_ids)
            ]

            output_text = val_dataset.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            result_texts.extend(output_text)


            bbox_start_idx = val_dataset.processor.tokenizer.encode("<|bbox_start|>")[0]
            bbox_end_idx = val_dataset.processor.tokenizer.encode("<|bbox_end|>")[0]

            for index, answer in enumerate(generated_ids_trimmed):
                answer_list = answer.cpu().tolist()
                bbox_start = None
                bbox_end = None
                bboxes = []
                for generated_id in answer_list:
                    if generated_id == bbox_start_idx:
                        bbox_start = generated_id
                    if generated_id == bbox_end_idx:
                        bbox_end = generated_id
                        bboxes.append((bbox_start, bbox_end))

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




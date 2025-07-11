import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


import json
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import List, Dict, Tuple, Optional, Union
from transformers import AutoProcessor, Qwen2_5_VLProcessor, Qwen2Tokenizer, AutoTokenizer
from qwen_vl_utils import process_vision_info


class MyDataset(Dataset):
    """
    YOLO格式目标检测数据集类
    """
    
    def __init__(self, annotation_path: str, processor_path: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
        """
        Args:
            annotation_path (str): 标注文件路径
        """

        self.annotation_path = annotation_path
        self.processor = self.load_processor(processor_path)
        self.image_root = os.path.join(os.path.dirname(annotation_path), "JPEGImages")

        with open(self.annotation_path, 'r', encoding='utf-8') as f:
            self.data_samples = [json.loads(line) for line in f]
    

    def load_processor(self, processor_path: str):
        """
        加载processor
        """
        processor = AutoProcessor.from_pretrained(processor_path, use_fast=False, padding_side="left")
        
        #########################################################################################
        '''给processor中的tokenizer添加特殊标记'''
        #########################################################################################
        special_tokens = [
            '<|tool_call_start|>',
            '<|tool_call_end|>',
            '<|tool_result_start|>', 
            '<|tool_result_end|>',
            '<|bbox_start|>', 
            '<|bbox_end|>'
        ]
        
        default_special_tokens = processor.tokenizer.additional_special_tokens
        all_special_tokens = default_special_tokens + special_tokens

        # 过滤掉已经存在的特殊标记
        existing_tokens = set(processor.tokenizer.get_vocab().keys())
        new_special_tokens = [token for token in all_special_tokens if token not in existing_tokens]
        
        if new_special_tokens:
            # 添加特殊标记到tokenizer
            num_added = processor.tokenizer.add_special_tokens({
                'additional_special_tokens': new_special_tokens + default_special_tokens
            })


        #########################################################################################
        '''因为使用了新的special tokens，因此需要使用新的chat_template，并进行测试'''
        #########################################################################################
        new_chat_template = """{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% set object_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system
        You are a helpful assistant.<|im_end|>
        {% endif %}<|im_start|>{{ message['role'] }}
        {% if message['content'] is string %}{{ message['content'] }}<|im_end|>
        {% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif content['type'] == 'bbox' %}{% for box in content['bbox'] %}{% set object_count.value = object_count.value + 1 %}Obj {{ object_count.value }}: <|bbox_start|>{{ box }}<|bbox_end|>{% endfor %}{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>
        {% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
        {% endif %}"""

        processor.chat_template = new_chat_template

        return processor
    
    
    
    def __len__(self) -> int:
        return len(self.data_samples)
    
    def __getitem__(self, idx: int):
        """
        获取单个样本
        Args:
            idx: 样本索引
        Returns:
            包含图像和标签的字典
        """
        data_sample = self.data_samples[idx]
        
        image_path = data_sample['filename']

        # (x min, y min, x max, y max)
        positions = data_sample['positions']
        obj_name = data_sample['name']
        width, height = data_sample['size']

        frac_positions = []
        for position in positions:
            x_min, y_min, x_max, y_max = position
            # frac_position = f'(x_min={x_min/width:.4f}, y_min={y_min/height:.4f}, x_max={x_max/width:.4f}, y_max={y_max/height:.4f})'
            # NOTE 此处改为像素绝对坐标了
            frac_position = f'(x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max})'
            frac_positions.append(frac_position)
        
        # 构造message
        message = [
            {
                'role': 'system',
                'content': '你是一个很有帮助的小助手'
            },
            {
                'role': 'user',
                'content': [
                    {'image_url': os.path.join(self.image_root ,image_path)},
                    {'type': 'text', 'text': f'这张图片中有多少"{obj_name}"？它们的具体的位置在哪里？'}
                ]
            },
            {
                'role': 'assistant',
                'content': [
                    {'type': 'text', 'text': f'图片中一共有{len(positions)}个"{obj_name}", 位置如下：'},
                    {'type': 'bbox', 'bbox': frac_positions}
                ]
            }
        ]

        return {
            'image_path': os.path.join(self.image_root ,image_path),
            'obj_name': obj_name,
            'positions': frac_positions,
            'positions_original': positions,
            'image_size': (width, height),
            'message': message
        }
    
    def collate_fn(self, batch):
        """
        自定义批处理函数
        Args:
            batch: 批次数据列表
        Returns:
            批处理后的数据
        """
        image_paths = [item['image_path'] for item in batch]
        obj_names = [item['obj_name'] for item in batch]
        positions = [item['positions'] for item in batch]
        image_sizes = [item['image_size'] for item in batch]
        messages = [item['message'] for item in batch]
        positions_original = [item['positions_original'] for item in batch]

        texts = self.processor.apply_chat_template(
            conversation=messages,
            add_generation_prompt=False,
            add_vision_id=True,
            tokenize=False
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        labels = self.get_labels(inputs['input_ids'])
        
        return {
            'image_paths': image_paths,
            'obj_names': obj_names,
            'positions': positions,
            'positions_original': positions_original,
            'image_sizes': image_sizes,
            'messages': messages,
            'texts': texts,
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'pixel_values': inputs['pixel_values'],
            'image_grid_thw': inputs['image_grid_thw'],
            'labels': labels
        }


    def get_labels(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        根据 input_ids 生成 labels，用于计算损失。
        此函数会将用户输入部分（包括 system prompt）的标签设置为 -100，将padding部分也设置为-100
        只保留 assistant 回答部分的标签。


        第一步：找到 “<|im_start|>assistant\n” 的位置
        """
        labels = input_ids.clone().numpy()
        assistant_start = self.processor.tokenizer.encode("<|im_start|>assistant\n")
        assistant_end = self.processor.tokenizer.encode("<|im_end|>\n")
        
        # 遍历批处理中的每个样本
        for i, sample_input_ids in enumerate(labels):

            # 找到所有可能的 assistant 分隔符位置
            matches = self.find_pattern_indices(
                sample_input_ids, 
                assistant_start, 
                assistant_end
            )
            for (start, end) in matches:
                labels[i, start:end+len(assistant_end)] = -100

        labels2 = input_ids.clone().numpy()
        labels2[labels != -100] = -100

        return torch.tensor(labels2)



    def find_pattern_indices(self, a, pattern_start, pattern_end):
        """
        优化后的模式匹配函数，使用向量化操作提高性能
        
        参数:
        a -- NumPy一维数组
        pattern_start -- 序列开头的固定部分(前三个数字)
        pattern_end -- 序列结尾的固定部分(最后两个数字)
        
        返回:
        列表，包含所有匹配序列的(起始索引, 结束索引)元组
        """
        n = len(a)
        start_len = len(pattern_start)
        end_len = len(pattern_end)
        
        # 使用滑动窗口和向量化操作快速找到所有可能的起始位置
        # 创建所有可能的start_len长度的窗口
        start_windows = np.lib.stride_tricks.sliding_window_view(a, start_len)
        # 找到匹配pattern_start的位置
        start_matches = np.where(np.all(start_windows == pattern_start, axis=1))[0]
        
        # 同样方法找到所有可能的结束位置
        end_windows = np.lib.stride_tricks.sliding_window_view(a, end_len)
        end_matches = np.where(np.all(end_windows == pattern_end, axis=1))[0]
        end_matches += end_len - 1  # 转换为结束索引
        
        matches = []
        for start in start_matches:
            # 找到第一个在 start+start_len 之后的 end_matches
            valid_ends = end_matches[end_matches >= start + start_len]
            if len(valid_ends) > 0:
                matches.append((start, valid_ends[0]))  # 只取最近的
        
        return matches


class MyDataset_eval(Dataset):
    """
    YOLO格式目标检测数据集类
    """
    
    def __init__(self, annotation_path: str, processor_path: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
        """
        Args:
            annotation_path (str): 标注文件路径
        """

        self.annotation_path = annotation_path
        self.processor = self.load_processor(processor_path)
        self.eos_token_ids = [self.processor.tokenizer.eos_token_id, self.processor.tokenizer.pad_token_id]
        self.image_root = os.path.join(os.path.dirname(annotation_path), "JPEGImages")

        with open(self.annotation_path, 'r', encoding='utf-8') as f:
            self.data_samples = [json.loads(line) for line in f]
    

    def load_processor(self, processor_path: str):
        """
        加载processor
        """
        processor = AutoProcessor.from_pretrained(processor_path, use_fast=False, padding_side="left")
        
        #########################################################################################
        '''给processor中的tokenizer添加特殊标记'''
        #########################################################################################
        special_tokens = [
            '<|tool_call_start|>',
            '<|tool_call_end|>',
            '<|tool_result_start|>', 
            '<|tool_result_end|>',
            '<|bbox_start|>', 
            '<|bbox_end|>'
        ]
        
        default_special_tokens = processor.tokenizer.additional_special_tokens
        all_special_tokens = default_special_tokens + special_tokens

        # 过滤掉已经存在的特殊标记
        existing_tokens = set(processor.tokenizer.get_vocab().keys())
        new_special_tokens = [token for token in all_special_tokens if token not in existing_tokens]
        
        if new_special_tokens:
            # 添加特殊标记到tokenizer
            num_added = processor.tokenizer.add_special_tokens({
                'additional_special_tokens': new_special_tokens + default_special_tokens
            })


        #########################################################################################
        '''因为使用了新的special tokens，因此需要使用新的chat_template，并进行测试'''
        #########################################################################################
        new_chat_template = """{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% set object_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system
        You are a helpful assistant.<|im_end|>
        {% endif %}<|im_start|>{{ message['role'] }}
        {% if message['content'] is string %}{{ message['content'] }}<|im_end|>
        {% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif content['type'] == 'bbox' %}{% for box in content['bbox'] %}{% set object_count.value = object_count.value + 1 %}Obj {{ object_count.value }}: <|bbox_start|>{{ box }}<|bbox_end|>{% endfor %}{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>
        {% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
        {% endif %}"""

        processor.chat_template = new_chat_template

        return processor
    
    
    
    def __len__(self) -> int:
        return len(self.data_samples)
    
    def __getitem__(self, idx: int):
        """
        获取单个样本
        Args:
            idx: 样本索引
        Returns:
            包含图像和标签的字典
        """
        data_sample = self.data_samples[idx]
        
        image_path = data_sample['filename']

        # (x min, y min, x max, y max)
        positions = data_sample['positions']
        obj_name = data_sample['name']
        width, height = data_sample['size']

        frac_positions = []
        for position in positions:
            x_min, y_min, x_max, y_max = position
            # frac_position = f'(x_min={x_min/width:.4f}, y_min={y_min/height:.4f}, x_max={x_max/width:.4f}, y_max={y_max/height:.4f})'
            frac_position = f'(x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max})'
            frac_positions.append(frac_position)
        
        # 构造message
        message = [
            {
                'role': 'system',
                'content': '你是一个很有帮助的小助手'
            },
            {
                'role': 'user',
                'content': [
                    {'image_url': os.path.join(self.image_root ,image_path)},
                    {'type': 'text', 'text': f'这张图片中有多少"{obj_name}"？它们的具体的位置在哪里？'}
                ]
            }
        ]

        return {
            'image_path': os.path.join(self.image_root ,image_path),
            'obj_name': obj_name,
            'positions': frac_positions,
            'positions_original': positions,
            'image_size': (width, height),
            'message': message
        }
    
    def collate_fn(self, batch):
        """
        自定义批处理函数
        Args:
            batch: 批次数据列表
        Returns:
            批处理后的数据
        """
        image_paths = [item['image_path'] for item in batch]
        obj_names = [item['obj_name'] for item in batch]
        positions = [item['positions'] for item in batch]
        positions_original = [item['positions_original'] for item in batch]
        image_sizes = [item['image_size'] for item in batch]
        messages = [item['message'] for item in batch]

        texts = self.processor.apply_chat_template(
            conversation=messages,
            add_generation_prompt=True,
            add_vision_id=False,
            tokenize=False
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        labels = self.get_labels(inputs['input_ids'])
        
        return {
            'image_paths': image_paths,
            'obj_names': obj_names,
            'positions': positions,
            'positions_original': positions_original,
            'image_sizes': image_sizes,
            'messages': messages,
            'texts': texts,
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'pixel_values': inputs['pixel_values'],
            'image_grid_thw': inputs['image_grid_thw'],
            'labels': labels
        }


    def get_labels(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        根据 input_ids 生成 labels，用于计算损失。
        此函数会将用户输入部分（包括 system prompt）的标签设置为 -100，将padding部分也设置为-100
        只保留 assistant 回答部分的标签。


        第一步：找到 “<|im_start|>assistant\n” 的位置
        """
        labels = input_ids.clone().numpy()
        assistant_start = self.processor.tokenizer.encode("<|im_start|>assistant\n")
        assistant_end = self.processor.tokenizer.encode("<|im_end|>\n")
        
        # 遍历批处理中的每个样本
        for i, sample_input_ids in enumerate(labels):

            # 找到所有可能的 assistant 分隔符位置
            matches = self.find_pattern_indices(
                sample_input_ids, 
                assistant_start, 
                assistant_end
            )
            for (start, end) in matches:
                labels[i, start:end+len(assistant_end)] = -100

        labels2 = input_ids.clone().numpy()
        labels2[labels != -100] = -100

        return torch.tensor(labels2)



    def find_pattern_indices(self, a, pattern_start, pattern_end):
        """
        优化后的模式匹配函数，使用向量化操作提高性能
        
        参数:
        a -- NumPy一维数组
        pattern_start -- 序列开头的固定部分(前三个数字)
        pattern_end -- 序列结尾的固定部分(最后两个数字)
        
        返回:
        列表，包含所有匹配序列的(起始索引, 结束索引)元组
        """
        n = len(a)
        start_len = len(pattern_start)
        end_len = len(pattern_end)
        
        # 使用滑动窗口和向量化操作快速找到所有可能的起始位置
        # 创建所有可能的start_len长度的窗口
        start_windows = np.lib.stride_tricks.sliding_window_view(a, start_len)
        # 找到匹配pattern_start的位置
        start_matches = np.where(np.all(start_windows == pattern_start, axis=1))[0]
        
        # 同样方法找到所有可能的结束位置
        end_windows = np.lib.stride_tricks.sliding_window_view(a, end_len)
        end_matches = np.where(np.all(end_windows == pattern_end, axis=1))[0]
        end_matches += end_len - 1  # 转换为结束索引
        
        matches = []
        for start in start_matches:
            # 找到第一个在 start+start_len 之后的 end_matches
            valid_ends = end_matches[end_matches >= start + start_len]
            if len(valid_ends) > 0:
                matches.append((start, valid_ends[0]))  # 只取最近的
        
        return matches



if __name__ == "__main__":

    # #########################################################################################
    # '''加载processor'''
    # #########################################################################################
    # model_name_or_path = "Qwen/Qwen2.5-VL-3B-Instruct"
    # processor = AutoProcessor.from_pretrained(model_name_or_path, use_fast=False)


    # #########################################################################################
    # '''给processor中的tokenizer添加特殊标记'''
    # #########################################################################################
    # special_tokens = [
    #     '<|tool_call_start|>',
    #     '<|tool_call_end|>',
    #     '<|tool_result_start|>', 
    #     '<|tool_result_end|>',
    #     '<|bbox_start|>', 
    #     '<|bbox_end|>'
    # ]
    
    # # 默认的特殊标记
    # default_special_tokens = processor.tokenizer.additional_special_tokens
    # all_special_tokens = default_special_tokens + special_tokens
    
    # # 过滤掉已经存在的特殊标记
    # existing_tokens = set(processor.tokenizer.get_vocab().keys())
    # new_special_tokens = [token for token in all_special_tokens if token not in existing_tokens]
    
    # if new_special_tokens:
    #     print(f"添加新的特殊标记: {new_special_tokens}")

    #     # 添加特殊标记到tokenizer
    #     num_added = processor.tokenizer.add_special_tokens({
    #         'additional_special_tokens': new_special_tokens + default_special_tokens
    #     })
        
    #     print(f"成功添加了 {num_added} 个特殊标记")
    #     print(f"词表大小从 {len(processor.tokenizer.get_vocab()) - num_added} 增加到 {len(processor.tokenizer.get_vocab())}")
    # else:
    #     print("所有特殊标记都已存在，无需添加")


    # #########################################################################################
    # '''因为使用了新的special tokens，因此需要使用新的chat_template，并进行测试'''
    # #########################################################################################
    # new_chat_template = """{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% set object_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system
    # You are a helpful assistant.<|im_end|>
    # {% endif %}<|im_start|>{{ message['role'] }}
    # {% if message['content'] is string %}{{ message['content'] }}<|im_end|>
    # {% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif content['type'] == 'bbox' %}{% for box in content['bbox'] %}{% set object_count.value = object_count.value + 1 %}Obj {{ object_count.value }}: <|bbox_start|>{{ box }}<|bbox_end|>{% endfor %}{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>
    # {% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
    # {% endif %}"""

    # processor.chat_template = new_chat_template
    # print("--- 已成功将自定义模板应用到 Processor ---")

    
    # messages = [
    #     {
    #         'role': 'system',
    #         'content': "你是一个很有帮助的小助手"
    #     },
    #     {
    #         'role': 'user',
    #         'content': [
    #             {'type': 'text', 'text': '这张图片中有麻雀吗？'},
    #             {'image_url': '...url...'},
    #         ]
    #     },
    #     {
    #         'role': 'assistant',
    #         'content': '有麻雀，一共有2只。'
    #     },
    #     {
    #         'role': 'user',
    #         'content': '麻雀都在什么位置？'
    #     },
    #     {
    #         'role': 'assistant',
    #         'content': [
    #             {'type': 'text', 'text': '图中两只麻雀的位置如下：'},
    #             {'type': 'bbox', 'bbox': ['(x_min=0.12, y_min=0.13, x_max=0.22, y_max=0.23)', '(x_min=0.32, y_min=0.33, x_max=0.66, y_max=0.67)']}
    #         ]
    #     },
    #     {
    #         'role': 'user',
    #         'content': '图中有兔子吗？'
    #     }
    # ]


    # # 使用 apply_chat_template 进行测试
    # #    - add_generation_prompt=True 和 add_vision_id=True 会作为变量传入模板
    # #    - tokenize=False 是关键，它让方法返回格式化后的字符串，而不是token ID列表
    # formatted_text = processor.apply_chat_template(
    #     conversation=messages,
    #     add_generation_prompt=True,
    #     add_vision_id=True,
    #     tokenize=False
    # )

    # print("\n--- 模板格式化后的最终输出 ---")
    # print(formatted_text)


    # expected_output = """<|im_start|>system
    # 你是一个很有帮助的小助手<|im_end|>
    # <|im_start|>user
    # 这张图片中有麻雀吗？Picture 1: <|vision_start|><|image_pad|><|vision_end|><|im_end|>
    # <|im_start|>assistant
    # 有麻雀，一共有2只。<|im_end|>
    # <|im_start|>user
    # 麻雀都在什么位置？<|im_end|>
    # <|im_start|>assistant
    # 图中两只麻雀的位置如下：Obj 1: <|bbox_start|>(x_min=0.12, y_min=0.13, x_max=0.22, y_max=0.23)<|bbox_end|>Obj 2: <|bbox_start|>(x_min=0.32, y_min=0.33, x_max=0.66, y_max=0.67)<|bbox_end|><|im_end|>
    # <|im_start|>user
    # 图中有兔子吗？<|im_end|>
    # <|im_start|>assistant
    # """

    # cleaned_formatted = "".join(formatted_text.split())
    # cleaned_expected = "".join(expected_output.split())

    # assert cleaned_formatted == cleaned_expected, "输出与预期不符！"
    # print("\n--- 验证成功：输出与预期完全一致！ ---")


    # #########################################################################################
    # '''测试batch处理能力'''
    # #########################################################################################
    # messages1 = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {
    #                 "type": "image",
    #                 "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
    #             },
    #             {"type": "text", "text": "Describe this image."},
    #         ],
    #     }
    # ]

    # messages2 = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "text", "text": "dhs0o sjdfao 3ejl  zjoidfawj daj sjdoai jdjifj ak"},
    #             {
    #                 "type": "image",
    #                 "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
    #             },
    #             {"type": "text", "text": "Describe this image."},
    #         ],
    #     },
    #     {
    #         "role": "assistant",
    #         "content": "lkdjoi jdsfoia  sdikoai "
    #     }
    # ]

    # messages = [messages1, messages2]
    # text = processor.apply_chat_template(
    #     messages, tokenize=False, add_generation_prompt=True
    # )

    # image_inputs, video_inputs = process_vision_info(messages)
    # inputs = processor(
    #     text=text,
    #     images=image_inputs,
    #     videos=video_inputs,
    #     padding=True,
    #     return_tensors="pt",
    # )
    # inputs = inputs.to("cuda")


    #########################################################################################
    '''测试完整的dataset和dataloader处理能力'''
    #########################################################################################

    # train_dataset = MyDataset(annotation_path="/slurm/home/yrd/kanlab/zhangchenfeng/program/practice_model/und_sft/detection_dataset/train.jsonl")

    # batch_size_for_ddp = 4

    # train_dataloader = DataLoader(train_dataset, 
    #                               batch_size=batch_size_for_ddp,
    #                               sampler=None, 
    #                               collate_fn=train_dataset.collate_fn,
    #                               shuffle=True, 
    #                               num_workers=2)
    
    
    # print(torch.cuda.is_available())
    # for batch in train_dataloader:
    #     print(batch)
    #     breakpoint()


    #########################################################################################
    '''测试inference的dataset和dataloader处理能力'''
    #########################################################################################
    
    annotation_path = "/slurm/home/yrd/kanlab/zhangchenfeng/program/practice_model/und_sft/detection_dataset/test.jsonl"
    processor_path = "Qwen/Qwen2.5-VL-3B-Instruct"
    eval_batch_size = 32
    num_workers = 2


    val_dataset = MyDataset_eval(annotation_path = annotation_path, processor_path = processor_path)
    val_dataloader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, collate_fn=val_dataset.collate_fn)

    for batch in val_dataloader:
        print(batch)
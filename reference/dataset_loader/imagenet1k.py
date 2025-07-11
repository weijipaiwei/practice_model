import json
import os
import random
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from transformers import Gemma3ImageProcessor

class Imagenet1k(Dataset):
    def __init__(self, root_dir, vision_encoder_image_size=896, vae_image_size=256, category_map_file=None):
        """
        Args:
            root_dir (string): imagenet的根目录
            vision_encoder_image_size: 视觉编码器输入图像的大小
            vae_image_size: VAE输入图像的大小

        """
        self.root_dir = root_dir
        self.vision_encoder_transform = self.get_image_transform(vision_encoder_image_size, random_flip=False)
        self.vae_transform = self.get_image_transform(vae_image_size, random_flip=False)
        self.split_name = [f for f in os.listdir(root_dir)]
        self.vision_encoder_image_size = vision_encoder_image_size
        self.vae_image_size = vae_image_size
        
        with open(category_map_file, 'r') as f:
            category_map_original = json.load(f)
        
        '''
        "WNID": {
            "ILSVRC2012_ID": 
            "WNID": 
            "words": 
            "gloss": 
            "num_train_images": 
        }
        '''
        self.category_map = {}
        for k, v in category_map_original.items():
            self.category_map[v["WNID"]] = v
        
        '''
        (img_path, caption, class_id)
        '''
        self.dataset = []
        for split in self.split_name:
            if "train" in split:
                path = os.path.join(root_dir, split)
                img_names = os.listdir(path)
                img_path = [os.path.join(path, name) for name in img_names]
                img_num = len(img_names)
                for i in range(img_num):
                    category = img_names[i].split("_")[0]
                    words = self.category_map[category]["words"]
                    caption = self.get_augmented_caption(words)

                    class_id = int(self.category_map[category]["ILSVRC2012_ID"])
                    self.dataset.append((img_path[i], caption, class_id))

        self.test_captions = []
        for i in range(8):
            test = random.choice(self.dataset)
            self.test_captions.append(test[1])
 
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        '''
        (img_path, caption, class_id)
        '''
        img_path, caption, class_id = self.dataset[idx]
        # 读取图片
        image = Image.open(img_path).convert("RGB")

        image_for_vision_encoder = self.vision_encoder_transform(image)

        sample = {'image_for_vision_encoder': image_for_vision_encoder, 'caption': caption, 'class_id': class_id}

        return sample
    
    def collate_fn(self, batch):
        batched = {'images_for_vision_encoder': [], 'captions': [], 'class_ids': []}
        for i in batch:
            batched['images_for_vision_encoder'].append(i['image_for_vision_encoder'])
            batched['captions'].append(i['caption'])
            batched['class_ids'].append(i['class_id'])

        batched['images_for_vision_encoder'] = torch.stack(batched['images_for_vision_encoder'], dim=0)
        return batched
    

    def get_image_transform(self, resolution, random_flip):

        transform = transforms.Compose(
            [
                transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        return transform
    

    def get_augmented_caption(self, words):
        tmp_words = words.split(",")
        words_list = []
        for words in tmp_words:
            if words != "":
                words_list.append(words.strip())

        # 从words随机选择一个词
        primary_object = random.choice(words_list)

        # 准备模板
        templates = [
            "Generate a picture of {primary_object}.",
            "Get a picture of {primary_object}.",
            "Draw a picture of {primary_object}.",
            "Create a picture about {primary_object}.",
            "Please generate a picture of {primary_object}.",
            "Please get a picture of {primary_object}.",
            "Please draw a picture of {primary_object}.",
            "Please create a picture about {primary_object}.",
            "Please generate a picture about {primary_object}.",
        ]
        
        # 随机选择一个模板
        template = random.choice(templates)
        
        # 替换模板中的占位符
        caption = template.format(
            primary_object=primary_object,
        )
        
        return caption
    

    
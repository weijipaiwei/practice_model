# inference pipelien

import hydra
import logging
import os
import torch
import transformers
import diffusers
import argparse
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger
from tqdm import tqdm
from torch.utils.data import DataLoader

from train_flowllm_f8d16.models.EMA import EMAModel
from train_flowllm_f8d16.models.flowllm import Flowllm

logger = get_logger(__name__, log_level="INFO")

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to the config file")
    args = parser.parse_args()
    
    basic_config = OmegaConf.load(args.config)
    if hasattr(basic_config, "special_config_path"):
        for special_config_path in basic_config.special_config_path:
            special_config = OmegaConf.load(special_config_path)
            OmegaConf.resolve(special_config)
            basic_config = OmegaConf.merge(basic_config, special_config)
    
    OmegaConf.resolve(basic_config)
    # 如果命令行还有参数，则也合并进来
    basic_config = OmegaConf.merge(basic_config, OmegaConf.from_cli())

    return basic_config


def main(args):
    #################################### 1. 准备推理的环境 - 开始 ####################################

    accelerator: Accelerator = hydra.utils.instantiate(args.accelerate, _convert_="partial")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(f"Finish instantiating accelerator <{args.accelerate._target_}>.")

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False
    torch.set_float32_matmul_precision('high')

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    
    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mixed_precision == "bf16":
        dtype = torch.bfloat16
    elif args.mixed_precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    #################################### 1. 准备训练的环境 - 结束 ####################################



    #################################### 2. 加载模型和推理数据集 - 开始 ####################################
    if args.model_name == "flowllm":
        tmp_list = os.listdir(args.output_dir)
        ckpt_path = None
        for tmp in tmp_list:
            if "final_checkpoint" in tmp:
                ckpt_path = os.path.join(args.output_dir, tmp)
                break
        if ckpt_path is None:
            raise ValueError("No checkpoint found in the output directory")
        model = Flowllm.from_pretrained(ckpt_path) # TODO

        if args.use_ema:
            ema_path = None
            tmp_list = os.listdir(ckpt_path)
            for tmp in tmp_list:
                if "ema_model" in tmp:
                    ema_path = os.path.join(ckpt_path, tmp)
                    break
            if ema_path is None:
                raise ValueError("No EMA model found in the output directory: {}".format(args.output_dir))
            
            ema_model = EMAModel(model.parameters())
            checkpoint = torch.load(ema_path, map_location="cpu")
            ema_model.load_state_dict(checkpoint)

            for ema_param, model_param in zip(ema_model.shadow_params, model.parameters()):
                # FIXME llm的词表保存到ema中时，只保存了6个，所以这里会报错。。。
                try:
                    model_param.data.copy_(ema_param.data)
                except:
                    logger.info(f"Failed to copy EMA parameters to model parameters for {model_param.shape}")
                    breakpoint()
    else:
        raise ValueError("Model '{}' not implemented".format(args.model_name))

    # 加载推理数据集
    logger.info(f"Loading inference dataset from {args.inference_dataset._target_}")
    inference_dataset = hydra.utils.instantiate(args.inference_dataset)

    # 4. 推理，并且在推理的过程中保存推理结果
    inference_dataloader = DataLoader(inference_dataset, 
                            batch_size=args.inference_batch_size,
                            sampler=None, 
                            collate_fn=inference_dataset.collate_fn,
                            shuffle=False, 
                            num_workers=args.dataloader_num_workers)
    #################################### 2. 加载模型和推理数据集 - 结束 ####################################

    

    #################################### 3. 推理 - 开始 ####################################
    model, inference_dataloader = accelerator.prepare(model, inference_dataloader)
    model.eval()

    with torch.no_grad():
        with torch.autocast(device_type=model.device.type, dtype=dtype):
            for step, batch in enumerate(inference_dataloader):
                images = batch["images_for_vision_encoder"]
                captions = batch["captions"]
                logger.info(f"Start training...")

                result_dict = model.forward_generation(texts=captions, images=images)
                lpd = result_dict.last_predicted_distributions
                # predicted_fm_condition_list = result_dict.backward_kl_loss
                # interpolated_fm_condition_latents = result_dict.forward_kl_loss
                pvi = model.decode_vision(predicted_distributions=lpd)
                print("loss is", result_dict.total_loss)
                print("captions is", captions)

                # 保存tensor
                # torch.save(interpolated_fm_condition_latents, os.path.join(ckpt_path, f"train_interpolated_fm_condition_latents.pt"))
                # torch.save(predicted_fm_condition_list, os.path.join(ckpt_path, f"train_predicted_fm_condition_list.pt"))
                

                # 保存图片
                pvi = (pvi / 2 + 0.5).clamp(0, 1).squeeze()
                pvi = (pvi.permute(0, 2, 3, 1) * 255).to(torch.uint8).cpu().numpy()
                images_list = [np.asarray(img) for img in pvi][:8]
                # 保存图片到本地
                predicted_vae_image_save_path = os.path.join(ckpt_path, f"training_image_ecodec")
                if not os.path.exists(predicted_vae_image_save_path):
                    os.makedirs(predicted_vae_image_save_path)
                for i, img_array in enumerate(images_list):
                    img = Image.fromarray(img_array)
                    img.save(os.path.join(predicted_vae_image_save_path, f"{i}.png"))
                

                gi = result_dict.gt_image
                gi = (gi / 2 + 0.5).clamp(0, 1).squeeze()
                gi = (gi.permute(0, 2, 3, 1) * 255).to(torch.uint8).cpu().numpy()
                images_list = [np.asarray(img) for img in gi][:8]


                ground_truth_image_save_path = os.path.join(ckpt_path, f"training_image_gt")
                if not os.path.exists(ground_truth_image_save_path):
                    os.makedirs(ground_truth_image_save_path)
                for i, img_array in enumerate(images_list):
                    img = Image.fromarray(img_array)
                    img.save(os.path.join(ground_truth_image_save_path, f"{i}.png"))

                logger.info(f"Start inference...")
                # 仅使用caption生成
                pvi = model.generate_cfg(captions)

                # 保存图片
                pvi = (pvi / 2 + 0.5).clamp(0, 1).squeeze()
                pvi = (pvi.permute(0, 2, 3, 1) * 255).to(torch.uint8).cpu().numpy()
                images_list = [np.asarray(img) for img in pvi]

                test_predicted_vae_image_save_path = os.path.join(ckpt_path, f"inference_image_generation")
                if not os.path.exists(test_predicted_vae_image_save_path):
                    os.makedirs(test_predicted_vae_image_save_path)
                for i, img_array in enumerate(images_list):
                    img = Image.fromarray(img_array)
                    img.save(os.path.join(test_predicted_vae_image_save_path, f"{i}.png"))
                
                # 保存tensor
                # torch.save(infer_interpolated_fm_condition_latents, os.path.join(ckpt_path, f"inference_interpolated_fm_condition_latents.pt"))
                # torch.save(infer_predicted_fm_condition_list, os.path.join(ckpt_path, f"inference_predicted_fm_condition_list.pt"))

                break

    # # 使用test captions 进行测试
    # test_captions = inference_dataset.test_captions

    # print(test_captions)

    # logger.info(f"Generating images with {args.model_name}...")
    # with torch.no_grad():
    #     pvi = model.generate_cfg(test_captions)

    # # 保存图片
    # pvi = (pvi / 2 + 0.5).clamp(0, 1).squeeze()
    # pvi = (pvi.permute(0, 2, 3, 1) * 255).to(torch.uint8).cpu().numpy()
    # images_list = [np.asarray(img) for img in pvi]

    # test_predicted_vae_image_save_path = os.path.join(ckpt_path, f"inference_image_generation")
    # if not os.path.exists(test_predicted_vae_image_save_path):
    #     os.makedirs(test_predicted_vae_image_save_path)
    # for i, img_array in enumerate(images_list):
    #     img = Image.fromarray(img_array)
    #     img.save(os.path.join(test_predicted_vae_image_save_path, f"{i}.png"))
    

    #################################### 3. 推理 - 结束 ####################################


if __name__ == "__main__":
    args = get_config()
    main(args)

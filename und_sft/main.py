
import os
import math
import hydra
import argparse
import torch
import copy

import logging
import transformers
import diffusers
import shutil
import numpy as np

from PIL import Image
from omegaconf import OmegaConf
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from diffusers.utils.torch_utils import is_compiled_module


from model import MyModel, EMAModel
from dataset import MyDataset

logger = get_logger(__name__, log_level="INFO")

# Function for unwrapping if model was compiled with `torch.compile`.
def unwrap_model(model, accelerator, restore_config=False):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

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

    #################################### 1. 准备训练的环境 - 开始 ####################################

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


    #################################### 2. 准备数据集 - 开始 ########################################


    logger.info(f"Instantiating dataset <{args.train_dataset._target_}>.")
    train_dataset: MyDataset = hydra.utils.instantiate(args.train_dataset, _convert_="partial") # _convert_="partial"：只对嵌套的 Hydra 配置对象（带有_target_的）进行递归实例化，而普通字典或列表保持不变。

    batch_size_for_ddp = args.train_batch_size // (args.gradient_accumulation_steps * accelerator.num_processes) # 一张GPU上，一个forward过程中的batch size
    
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=batch_size_for_ddp,
                                  sampler=None, 
                                  collate_fn=train_dataset.collate_fn,
                                  shuffle=True, 
                                  num_workers=args.dataloader_num_workers)

    if accelerator.is_main_process:
        # 创建测试集
        val_dataset: MyDataset = hydra.utils.instantiate(args.val_dataset, _convert_="partial")
        val_dataloader = DataLoader(val_dataset, 
                                    batch_size=args.val_batch_size,
                                    sampler=None, 
                                    collate_fn=val_dataset.collate_fn,
                                    shuffle=False, 
                                    num_workers=args.dataloader_num_workers)
    else:
        val_dataset = None
        val_dataloader = None

    #################################### 2. 准备数据集 - 结束 ########################################


    #################################### 3. 准备模型 - 开始 ##########################################


    # TODO 需要完善模型
    logger.info(f"Instantiating model <{args.model._target_}>.")
    model: MyModel = hydra.utils.instantiate(args.model, _convert_="partial").train()


    # 如果使用ema模型，则在此处进行初始化
    if args.use_ema:
        logger.info(f"Instantiating EMA model <{args.ema_model._target_}>.")
        ema_model: EMAModel = hydra.utils.instantiate(args.ema_model, _convert_="partial", parameters=model.parameters())
        ema_model = ema_model.to(accelerator.device)

    #################################### 3. 准备模型 - 结束 ###########################################


    #################################### 4. 准备优化器和学习率调度器 - 开始 #############################

    # 分离需要 weight decay 和不需要 weight decay 的参数
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if any(nd in name for nd in args.no_weight_decay_params):  # NOTE 不需要 weight decay 的参数。后续可以在config中继续添加
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    params = [
        {"params": decay_params, "weight_decay": args.adam_weight_decay},  # 需要 weight decay 的参数
        {"params": no_decay_params, "weight_decay": 0.0},  # 不需要 weight decay 的参数
    ]

    logger.info(f"Instantiating optimizer <{args.optimizer._target_}>.")
    optimizer: AdamW = hydra.utils.instantiate(args.optimizer, _convert_="partial", params=params)

    logger.info(f"Instantiating lr_scheduler <{args.lr_scheduler.name}>.")
    lr_scheduler: LambdaLR = hydra.utils.instantiate(args.lr_scheduler, 
                                                     _convert_="partial", 
                                                     optimizer=optimizer,
                                                     num_warmup_steps=args.lr_warmup_rate * args.max_train_steps * accelerator.num_processes,
                                                     num_training_steps=args.max_train_steps * accelerator.num_processes)

    #################################### 4. 准备优化器和学习率调度器 - 结束 #############################


    #################################### 5. 准备开始训练 - 开始 ########################################

    # 1. 准备需要训练的mllm_with_diffusion 以及需要手动更新参数的ema model
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )


    # 2. 初始化tensorboard
    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name)
    
    # 4. 计算epoch。因为数据集大小前面才知道，因此这里才能计算总的epoch（因为我们直接规定了max_train_steps）
    # args.num_train_epochs = args.max_train_steps // (math.ceil(len(train_dataloader) / (args.gradient_accumulation_steps * batch_size_for_ddp)))
    
    args.num_train_epochs = args.max_train_steps // (math.ceil(len(train_dataloader) / args.train_batch_size))
    # 5. 查看是否需要从checkpoint恢复
    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            if args.use_ema:
                ema_state = torch.load(os.path.join(os.path.join(args.output_dir, path, "ema_state.pth")), map_location="cpu")
                ema_model.load_state_dict(ema_state)
                logger.info("EMA model loaded successfully")

            initial_global_step = global_step
            # first_epoch = global_step // math.ceil(len(train_dataloader) / (args.gradient_accumulation_steps * batch_size_for_ddp))
            first_epoch = global_step // math.ceil(len(train_dataloader) / args.train_batch_size)

    else:
        initial_global_step = 0
    

    # 6. 打印训练的信息
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Num processes (total GPUs used) = {accelerator.num_processes}")
    logger.info(f"  Forward batch size per GPU = {batch_size_for_ddp}")
    logger.info(f"  Total train batch size = {args.train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # 7. 初始化进度条
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    #################################### 5. 准备开始训练 - 结束 ########################################


    #################################### 6. 开始训练 - 开始 ############################################

    for epoch in range(first_epoch, args.num_train_epochs):
        # 做一点每个epoch开始前的调整
        model.train()

         # NOTE 增加记录参数修改位置 1
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            '''
            TODO 此处应有 batch 的详细结构定义
            '''

            with accelerator.accumulate(model):
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                pixel_values = batch["pixel_values"]
                image_grid_thw = batch["image_grid_thw"]
                labels = batch["labels"]

                result_dict = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    return_dict=True,
                    labels=labels,
                )

                loss = result_dict.loss

                # NOTE 增加记录参数修改位置 2
                avg_loss = accelerator.gather(loss.repeat(batch_size_for_ddp)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            if accelerator.sync_gradients:
                # 在每一个train step结束后，做一点小调整
                progress_bar.update(1)
                global_step += 1
                if args.use_ema:
                    ema_model.step(model.parameters())

                # NOTE 增加记录参数修改位置 3
                logs = {"train_loss": train_loss, 
                        "lr": lr_scheduler.get_last_lr()[0]}
                if args.use_ema:
                    logs["ema_decay"] = ema_model.cur_decay_value
                accelerator.log(logs, step=global_step)
                train_loss = 0.0


                if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                    # 保存checkpoint
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")    
                    accelerator.save_state(save_path, safe_serialization=False)
                    logger.info(f"Saved state to {save_path}")
                    if args.use_ema:
                        ema_state = ema_model.state_dict()
                        torch.save(ema_state, os.path.join(save_path, "ema_state.pth"))
                        logger.info(f"Saved EMA state to {save_path}")
                    

                
                if args.validation_steps != -1 and global_step % args.validation_steps == 0 and accelerator.is_main_process:
                    # TODO
                    val_progress_bar = tqdm(
                        range(0, len(val_dataloader)),
                        initial=0,
                        desc="Validating",
                        # Only show the progress bar once on each machine.
                        disable=not accelerator.is_local_main_process,
                    )
                    unwrapped_model = unwrap_model(model, accelerator).eval()
                    with torch.no_grad():
                        total_loss = 0.0
                        for batch in val_dataloader:
                            input_ids = batch["input_ids"].to(accelerator.device)
                            attention_mask = batch["attention_mask"].to(accelerator.device)
                            pixel_values = batch["pixel_values"].to(accelerator.device)
                            image_grid_thw = batch["image_grid_thw"].to(accelerator.device)
                            labels = batch["labels"].to(accelerator.device)

                            result_dict = unwrapped_model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                pixel_values=pixel_values,
                                image_grid_thw=image_grid_thw,
                                return_dict=True,
                                labels=labels,
                            )

                            total_loss += result_dict.loss.item()
                            val_progress_bar.update(1)
                            logs = {"val_step_loss": result_dict.loss.detach().item()}
                            val_progress_bar.set_postfix(**logs)

                        avg_loss = total_loss / len(val_dataloader)
                        logs = {"val_loss": avg_loss}
                        accelerator.log(logs, step=global_step)
                        val_progress_bar.close()


            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            if global_step >= args.max_train_steps:
                break
        accelerator.wait_for_everyone()
        if global_step >= args.max_train_steps:
            break

    #################################### 6. 开始训练 - 结束 ############################################


    #################################### 7. 保存模型并结束所有训练 - 开始 ############################################

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, "final_checkpoint_{}_steps".format(global_step))
        state_dict = accelerator.get_state_dict(model)
        unwrapped_model = unwrap_model(model, accelerator)
        unwrapped_model.save_pretrained(
            save_path,
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=False
        )
        logger.info(f"Saved model final state to {save_path}")

        if args.use_ema:
            # 保存ema模型
            save_path = os.path.join(args.output_dir, "final_checkpoint_{}_steps".format(global_step), "ema_model.pth")
            ema_state = ema_model.state_dict()
            torch.save(ema_state, save_path)
            logger.info(f"Saved EMA model final state to {save_path}")
        
    accelerator.end_training()

    #################################### 7. 保存模型并结束所有训练 - 结束 ############################################

if __name__ == "__main__":
    args = get_config()
    main(args)
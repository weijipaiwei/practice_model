import torch
import math
import numpy as np
from torch import nn
import json

from dataclasses import dataclass
from transformers import PretrainedConfig, PreTrainedModel, Gemma3Config, GemmaTokenizerFast, GemmaTokenizer
import torch.nn.functional as F
from diffusers import AutoencoderKL

from train_flowvae_f8d16.models.flowvae import Flow_vae
from train_flowvae_f8d16.models.modules import DiagonalGaussianDistribution
from train_flowllm_f8d16.models.modules import MultiModalProjector, CustomCausalLM


@dataclass
class Flowllm_result(): # For training
    gt_image: torch.Tensor | None
    text_loss: torch.Tensor | None
    forward_kl_loss: torch.Tensor | None
    backward_kl_loss: torch.Tensor | None
    last_predicted_distributions: torch.Tensor | None
    latent_mse_loss: torch.Tensor | None
    total_loss: torch.Tensor | None



# TODO 最后根据参数修改
class Flowllm_config(PretrainedConfig):
    model_type = "Flowllm"
    def __init__(self,
                 gemma3_config_path =  "/home/zcf/weights/gemma3_4b_pt_language_model/gemma3_config.json",
                 gemma3_language_model_path = "/home/zcf/weights/gemma3_4b_pt_language_model",
                 flow_vae_path = "/home/zcf/Documents/demo_code/diff_gen/flux_vae",
                 multi_modal_projector_path = "/home/zcf/weights/gemma3_4b_pt_language_model/multi_modal_projector.pth",
                 mode = "generation", # "generation" or "understanding"
                 fm_condition_dim = 32,
                 fm_condition_level_size = [1, 2, 4, 8, 16],
                 text_loss_weight = 0.0,
                 forward_kl_loss_weight = 0.1,
                 backward_kl_loss_weight = 1.0,
                 latent_mse_loss_weight = 1.0,
                 **kwargs):
        self.gemma3_config_path = gemma3_config_path
        self.gemma3_language_model_path = gemma3_language_model_path
        self.flow_vae_path = flow_vae_path
        self.multi_modal_projector_path = multi_modal_projector_path
        self.mode = mode
        self.fm_condition_dim = fm_condition_dim
        self.fm_condition_level_size = fm_condition_level_size
        self.text_loss_weight = text_loss_weight
        self.forward_kl_loss_weight = forward_kl_loss_weight
        self.backward_kl_loss_weight = backward_kl_loss_weight
        self.latent_mse_loss_weight = latent_mse_loss_weight
        super().__init__(**kwargs)


class Flowllm(PreTrainedModel):
    config_class = Flowllm_config
    def __init__(self, config: Flowllm_config):
        super().__init__(config)

        self.config = config
        self.inf = 1e16

        # 读取预训练好的flow vae
        self.flow_vae = Flow_vae.from_pretrained(config.flow_vae_path)
        del self.flow_vae.vae.encoder
        del self.flow_vae.vae.quant_conv

        # 使用json读取gemma3的config
        with open(config.gemma3_config_path, 'r') as f:
            gemma3_config = json.load(f)
        self.gemma3_config = Gemma3Config(**gemma3_config)
        self.multi_modal_projector = MultiModalProjector(self.gemma3_config)
        self.multi_modal_projector.load_state_dict(torch.load(config.multi_modal_projector_path))

        self.tokenizer = GemmaTokenizerFast.from_pretrained(config.gemma3_language_model_path, use_fast=False)
        # self.tokenizer = GemmaTokenizer.from_pretrained(config.gemma3_language_model_path, use_fast=False)
        self.add_special_tokens_and_templates()

        self.language_model = CustomCausalLM.from_pretrained(config.gemma3_language_model_path, torch_dtype=torch.bfloat16, attn_implementation='eager')
        text_hidden_size = self.gemma3_config.text_config.hidden_size
        self.segment_position_embeddings = nn.Embedding(len(config.fm_condition_level_size), text_hidden_size)
        segment_position_ids = torch.arange(len(config.fm_condition_level_size)).unsqueeze(1).requires_grad_(False) # [L, 1]
        self.register_buffer('segment_position_ids', segment_position_ids)
        self.distribution_predictor_head = nn.Linear(in_features=text_hidden_size, out_features=config.fm_condition_dim*2, bias=False) # 2560 -> 64
        # self.distribution_predictor_head = self.flow_vae.attention_proj1


        self.vocab_size = self.gemma3_config.text_config.vocab_size
        self.pad_token_id = self.gemma3_config.pad_token_id if self.gemma3_config.pad_token_id is not None else -1

        self.mode = config.mode # "generation" or "understanding"
        self.rng = torch.Generator(device="cuda")


        self.no_grad_modules = ["flow_vae", "multi_modal_projector"]
        self.post_init()
        self.freeze_modules()
        # self.zero_init()

    def zero_init(self):
        '''
        for debug
        '''
        print(f"zero init model parameters")
        for name, param in self.named_parameters():
            nn.init.zeros_(param)
        
        for name, param in self.named_parameters():
            if param.sum() != 0:
                print(name, param.shape)
        print(f"zero init model parameters end")


    def add_special_tokens_and_templates(self):
        r'''
        添加生成图片任务所需要的special tokens
        - `<unused99>` 实际表示 `<start_of_image_generation>`
        - `<unused100>` 实际表示 `<end_of_image_generation>`


        对于图片生成任务：文本进入tokenizer之前的格式：
        {这里都是正常输入的文本}\n\n<start_of_image_generation><image_soft_token><end_of_image_generation>\n\n<eos>
        attention mask: 全部使用因果注意力

        对于图片理解任务：
        \n\n<start_of_image>256*<image_soft_token><end_of_image>\n\n{这里都是正常输入的文本}<eos>
        attention mask: 文本部分使用因果注意力，图片部分使用双向注意力


        text_tokenizer(texts, return_tensors="pt", padding=True, padding_side='left')
        '''
        # 确认未使用的 token 存在
        assert "<unused99>" in self.tokenizer.vocab, "<unused99> 不存在！"
        assert "<unused100>" in self.tokenizer.vocab, "<unused100> 不存在！"

        # 获取它们的 token ID
        self.start_of_image_generation_id = self.tokenizer.convert_tokens_to_ids("<unused99>")
        self.end_of_image_generation_id = self.tokenizer.convert_tokens_to_ids("<unused100>")
    
        # 定义常量（实际仍用 <unused99>，但代码中视为 <start_of_image>）
        self.START_OF_IMAGE_GENERATION = "<unused99>"
        self.END_OF_IMAGE_GENERATION = "<unused100>"

        # 将 <unused99> 和 <unused100> 添加到 special tokens
        self.tokenizer.add_special_tokens({
            "additional_special_tokens": ["<unused99>", "<unused100>"]  
        })

        # 创建图片attention mask模板
        total_length = 0
        _length = self.config.fm_condition_level_size[1:]
        for i in _length:
            total_length += i**2

        image_attn_mask = torch.zeros([total_length, total_length], dtype=torch.float)

        _sum = 0
        _cumsum_length = [0]
        for i in _length:
            _sum += i**2
            _cumsum_length.append(_sum)
        
        for i in range(len(_cumsum_length)-1):
            image_attn_mask[_cumsum_length[i]:_cumsum_length[i+1], _cumsum_length[i+1]:] = -1 * self.inf
        
        self.register_buffer('image_attn_mask', image_attn_mask)


    def freeze_modules(self):
        for module_name in self.no_grad_modules:
            if module_name == "flow_vae":
                for param in self.flow_vae.parameters():
                    param.requires_grad = False
            if module_name == "multi_modal_projector":
                for param in self.multi_modal_projector.parameters():
                    param.requires_grad = False
            if module_name == "language_model":
                for param in self.language_model.parameters():
                    param.requires_grad = False
        return 1

    def forward_generation(self, texts: list[str], images: torch.Tensor):
        '''
        texts: list of strings
        images: [b, 3, 896, 896]
        '''

        # 1. 将texts进行tokenize
        text_input = self.tokenizer(texts, return_tensors="pt", padding=True, padding_side='left').to(self.device)
        text_input_ids = text_input.input_ids
        text_attention_masks = text_input.attention_mask

        # 2. 将images进行编码
        with torch.no_grad():
            semantic_latents_list, semantic_conditions_list = self.flow_vae.get_semantic_level_latents(images)

        # 3. 并行解码
        predicted_text_logits, predicted_fm_condition_posterior_lsit = self.forward_parallel(text_input_ids, 
                                                                                                     text_attention_masks, 
                                                                                                     semantic_latents_list)
        
        # 4. 计算损失函数
        # 4.1 计算text_embeddings的损失函数
        # text_input_ids 进行偏移
        gt_labels = text_input_ids[:, 1:]
        gt_labels = gt_labels.reshape(-1)
        predict_logits = predicted_text_logits[:, :-1, :]
        predict_logits = predict_logits.reshape(-1, predict_logits.size(-1))

        loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        text_loss = loss_fct(predict_logits, gt_labels)


        # 4.2 计算fm_condition_distributions的损失函数
        forward_kl_loss = 0
        backward_kl_loss = 0
        latent_mse_loss = 0
        last_predicted_distributions = None
        for i, predicted_fm_condition_posterior in enumerate(predicted_fm_condition_posterior_lsit):
            gt_sample = semantic_conditions_list[i]
            # h = w = self.config.fm_condition_level_size[i]
            # dim = self.config.fm_condition_dim
            # breakpoint()
            # backward_kl_loss += predicted_fm_condition_posterior.kl(gt_dgd) / (h * w * dim)
            # forward_kl_loss += gt_dgd.kl(predicted_fm_condition_posterior) / (h * w * dim)
            predicted_sample = predicted_fm_condition_posterior.sample()
            latent_mse_loss += F.mse_loss(gt_sample, predicted_sample)
            if i == len(predicted_fm_condition_posterior_lsit) - 1:
                last_predicted_distributions = predicted_fm_condition_posterior

        total_loss = self.config.text_loss_weight * text_loss + \
            self.config.latent_mse_loss_weight * latent_mse_loss + \
            self.config.forward_kl_loss_weight * forward_kl_loss + \
            self.config.backward_kl_loss_weight * backward_kl_loss

        return Flowllm_result(
            gt_image=images,
            text_loss=text_loss,
            forward_kl_loss=forward_kl_loss,
            backward_kl_loss=backward_kl_loss,
            last_predicted_distributions=last_predicted_distributions,
            latent_mse_loss=latent_mse_loss,
            total_loss=total_loss
        )

    def forward_parallel(self, text_input_ids: torch.Tensor, text_attention_masks: torch.Tensor, fm_condition_latents: list[torch.Tensor]):
        '''
        text_input_ids: [B, seq_len] 已经经过了tokenizer，进行left padding，开头已经有了<sos> id了
        text_attention_masks: [B, seq_len]
        fm_condition_latents: list([B, 2560, h_i, w_i]) 从小到大排列，维度可以直接输入llm了
        '''
        device = text_input_ids.device
        batch_size = text_input_ids.size(0)


        # 1. FIXME 先添加位置编码。注意，借用的Qwen2Model已经使用了RoPE。未来消融实验可以做：是否需要多个位置编码，以及怎样使用位置编码效果更好
        # 1.1 text_input_ids 和 [SOI] token 的拼接。注意直接使用Qwen2Model的 '<|im_start|>': 151644 当作 [SOI]
        soig_id = torch.tensor([[self.start_of_image_generation_id]]).to(device) # [1,1] 的 start of image generation token id
        total_text_input_ids = torch.cat([text_input_ids, soig_id.repeat(batch_size, 1)], dim=1)
        total_text_attention_masks = torch.cat([text_attention_masks, torch.ones([batch_size, 1]).to(device)], dim=1)
        text_embeddings = self.language_model.model.embed_tokens(total_text_input_ids) # TODO 
        # text_embeddings = self.text_position_embeddings(text_embeddings) # [B, seq_len + 1, dim]

        # 1.2添加segment position embeddings到所有embedding中
        # 添加到text_embeddings中（此处不要添加到[SOI]）
        # text_embeddings[:, :-1, :] = self.segment_position_embeddings(self.segment_position_ids[0]) + text_embeddings[:, :-1, :]

        # 添加到[SOI]中
        text_embeddings[:, -1, :] = self.segment_position_embeddings(self.segment_position_ids[0]) + text_embeddings[:, -1, :]

        # 添加到image_semantic_latents中
        # image_semantic_latents = self.segment_position_embeddings(self.segment_position_ids[2]) + self.semantic_linear(image_semantic_latents)

        # 添加到image_texture_latents中。不需要对最后一个level的image texture latent进行操作。因为它实际上不会进入LLM
        for i, fm_condition_latent in enumerate(fm_condition_latents[:-1]):
            fm_condition_latents[i] = self.segment_position_embeddings(self.segment_position_ids[i+1]).unsqueeze(2).unsqueeze(3) + fm_condition_latent

        # 1.3对每一个尺寸的image_texture_latents都添加可学习的2d position embeddings。不需要对最后一个level的image texture latent进行操作。因为它实际上不会进入LLM
        # for i, image_texture_latent in enumerate(image_texture_latents[:-1]):
        #     image_texture_latents[i] = self.textural_multi_level_pos_embed_learned[i] + image_texture_latent

        # 2. 构造合适的LLM输入
        # 对每一个尺寸的 image_texture_latents 都进行插值，方便使用 predict next scale 方法计算损失函数。不需要对最后一个level的image texture latent进行操作。因为它实际上不会进入LLM
        interpolated_fm_condition_latents = []
        for i, fm_condition_latent in enumerate(fm_condition_latents[:-1]):
            h = w = self.config.fm_condition_level_size[i+1]
            interpolated_fm_condition_latents.append(F.interpolate(fm_condition_latent, size=(h, w), mode='nearest').reshape(batch_size, -1, h*w).permute(0, 2, 1))
        

        # NOTE 这里保存：train_Interp_decoded_semantic_latents_all：interpolated_fm_condition_latents
        # torch.save(interpolated_fm_condition_latents, "/slurm/home/yrd/kanlab/zhangchenfeng/program/diff_gen/train_flowllm/result/4/final_checkpoint_20000_steps/train_Interp_decoded_semantic_latents_all.pt")
        # breakpoint()

        embeddings_length_info = [len(text_input_ids[0]), 1, *[i.shape[1] for i in interpolated_fm_condition_latents]] # text input ids, [SOIG], interpolated_fm_condition_latents
        accumulated_embeddings_length_info = []
        _sum = 0
        for i in embeddings_length_info:
            _sum += i
            accumulated_embeddings_length_info.append(_sum)

        # TODO 2.1 构造attention mask。这是一个4d attention mask，由于attention中softmax机制，需要忽略的地方数值为 -100000.0，需要注意的地方数值为 0.0
        text_with_soig_length = total_text_attention_masks.sum(dim=1).int().cpu().tolist() 
        text_with_soig_padding_length = (1-total_text_attention_masks).sum(dim=1).int().cpu().tolist()
        text_length = total_text_attention_masks.shape[1]

        # 为batch中每一个样本构造4d attention mask
        whole_attention_mask = torch.ones([batch_size, sum(embeddings_length_info), sum(embeddings_length_info)]).to(device, torch.float) * (-1.0 * self.inf)
        for i in range(batch_size):
            diagonal_mask = (torch.tril(torch.ones([text_with_soig_length[i], text_with_soig_length[i]], dtype=torch.float)) - 1) * self.inf
            whole_attention_mask[i, text_with_soig_padding_length[i]:text_length, text_with_soig_padding_length[i]:text_length] = diagonal_mask
            whole_attention_mask[i, text_length:, text_with_soig_padding_length[i]:text_length] = 0.0
            whole_attention_mask[i, text_length:, text_length:] = self.image_attn_mask

        whole_attention_mask = whole_attention_mask.unsqueeze(1) # [B, 1, L, L]

        # TODO 2.2 拼接所有embeddings
        llm_input_embeddings = torch.cat([text_embeddings, *interpolated_fm_condition_latents], dim=1)
        past_seen_tokens = 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + llm_input_embeddings.shape[1], device=device
        ) 
        # TODO 3. 将所有embeddings输入到llm模型中，得到输出
        llm_output = self.language_model(
            attention_mask=whole_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=llm_input_embeddings,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            cache_position=cache_position,
            logits_to_keep=embeddings_length_info[0],
        )

        # 4. 分离出text_embeddings，image_semantic_latents，image_texture_latents
        # 4.1 分离出text_embeddings
        predicted_text_logits = llm_output.logits # [B, seq_len, D]

        # 4.2 分离出image_semantic_latents(输入[SOI]的位置的embeddings)
        # predicted_image_semantic_latents = llm_output[:, accumulated_embeddings_length_info[0]: accumulated_embeddings_length_info[1], :] # [B, 1, D]

        # 4.3 分离出image_texture_latents
        predicted_fm_condition_latents = llm_output.last_hidden_state[:, accumulated_embeddings_length_info[0]:, :]
        predicted_fm_condition_posterior_list = []

        # NOTE 只在debug时使用，后面删除
        _predicted_fm_condition_list = []

        _fm_condition_level_size = [0] + self.config.fm_condition_level_size # [0, 1, 2, 4, 8, 16, 32]
        s = 0
        for i in range(len(_fm_condition_level_size) - 1):
            s += int(_fm_condition_level_size[i]**2)
            length = int(_fm_condition_level_size[i+1]**2)
            predicted_fm_condition_latent = predicted_fm_condition_latents[:, s:s + length, :]

            predicted_fm_condition_distribution = self.distribution_predictor_head(predicted_fm_condition_latent)
            bs, _, dim = predicted_fm_condition_distribution.shape
            predicted_fm_condition_distribution = predicted_fm_condition_distribution.reshape(bs, int(np.sqrt(length)), int(np.sqrt(length)), dim).permute(0, 3, 1, 2) # [b, c, h, w]
            

            _predicted_fm_condition_list.append(DiagonalGaussianDistribution(predicted_fm_condition_distribution).sample())
            
            predicted_fm_condition_posterior_list.append(DiagonalGaussianDistribution(predicted_fm_condition_distribution))
        # NOTE 这里保存：train_Predicted_semantic_conditions_all：_predicted_fm_condition_list
        # torch.save(_predicted_fm_condition_list, "/slurm/home/yrd/kanlab/zhangchenfeng/program/diff_gen/train_flowllm/result/4/final_checkpoint_20000_steps/train_Predicted_semantic_conditions_all.pt")
        # breakpoint()
        
        return predicted_text_logits, predicted_fm_condition_posterior_list

    
    def forward_understanding(self):
        '''

        '''
        return 1
    
    def forward(self, texts: list[str], images: torch.Tensor):
        if self.mode == "generation":
            return self.forward_generation(texts, images)
        elif self.mode == "understanding":
            return self.forward_understanding()
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
    

    def encode_vision(self):
        '''

        '''
        return 1

    
    def decode_vision(self, predicted_distributions: DiagonalGaussianDistribution):
        '''
        predicted_distributions: DGD
        '''
        predicted_vae_image = self.flow_vae.decode_from_distributions(predicted_distributions)
        return predicted_vae_image
    
    def encode_text(self):
        '''

        '''
        return 1
    
    def decode_text(self):
        '''

        '''
        return 1
    
    def create_4d_causal_attention_mask(self, attention_mask, dtype=torch.float32):
        '''
        attention_mask: [batch_size, seq_length], 1 for tokens, 0 for padding
        '''
        # 1. 扩展 attention_mask 的维度从 [batch_size, seq_length] 到 [batch_size, 1, 1, seq_length]
        # 这样可以在头部维度和查询序列长度维度上进行广播
        batch_size, seq_length = attention_mask.shape
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        # 2. 创建因果掩码 (causal mask)
        # 创建一个下三角矩阵，对角线及以下为1，其余为0
        seq_ids = torch.arange(seq_length, device=attention_mask.device)
        causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1)
        causal_mask = causal_mask <= seq_ids[None, :, None]
        causal_mask = causal_mask.unsqueeze(1)  # [batch_size, 1, seq_length, seq_length]
        
        # 3. 结合 attention_mask 和 causal_mask
        # 将 extended_attention_mask 与 causal_mask 相乘
        # 注意：这里假设 attention_mask 中 1 表示需要关注的位置，0 表示需要忽略的位置
        combined_attention_mask = extended_attention_mask * causal_mask
        
        # 4. 转换为注意力分数掩码
        # 将 0 转换为 -inf，将 1 保持不变（或转换为 0）
        # 这样在 softmax 之前添加到注意力分数中时，-inf 会导致相应位置的注意力权重为 0
        combined_attention_mask = combined_attention_mask.to(dtype=dtype)
        combined_attention_mask = (1.0 - combined_attention_mask) * -self.inf
        
        return combined_attention_mask  # [batch_size, 1, seq_length, seq_length]
    

    def expand_attention_mask(self, attention_mask_4d, expand_size, pad_length):
        '''
        attention_mask_4d: [batch_size, 1, seq_length, seq_length]
        expand_size: int, 扩展的size
        pad_length: [b, ] 在左边填充的pad的个数
        '''
        device = attention_mask_4d.device
        dtype = attention_mask_4d.dtype
        b, _, s, s = attention_mask_4d.shape
        attention_mask_4d_expanded = torch.ones([b, 1, s + expand_size, s + expand_size]).to(device, dtype) * -self.inf
        attention_mask_4d_expanded[:, :, :s, :s] = attention_mask_4d

        for i in range(b):
            attention_mask_4d_expanded[i, :, s:, pad_length[i]:] = 0.

        

        return attention_mask_4d_expanded

    def generate_cfg(self, texts: list[str]):
        device = self.device
        batch_size = len(texts)

        # 1. 将texts进行tokenize
        text_input = self.tokenizer(texts, return_tensors="pt", padding=True, padding_side='left').to(device)
        text_input_ids = text_input.input_ids
        text_attention_masks = text_input.attention_mask

        # 首先拼接上sogi token id
        soig_id = torch.tensor([[self.start_of_image_generation_id]]).to(device) # [1,1] 的 start of image generation token id
        total_text_input_ids = torch.cat([text_input_ids, soig_id.repeat(batch_size, 1)], dim=1)
        total_text_attention_masks = torch.cat([text_attention_masks, torch.ones([batch_size, 1]).to(device)], dim=1)
        pad_length = (total_text_attention_masks.shape[1] - total_text_attention_masks.sum(1)).int()
        text_embeddings = self.language_model.model.embed_tokens(total_text_input_ids)

        # 添加位置编码到soig中
        text_embeddings[:, -1, :] = self.segment_position_embeddings(self.segment_position_ids[0]) + text_embeddings[:, -1, :]

        # NOTE 这两个临时变量只在debug时使用，后面删除
        # infer_Interp_decoded_semantic_latents_all = []
        # infer_Predicted_semantic_conditions_all = []


        with torch.no_grad():
            # TODO 将total_text_attention_masks转为4d attention mask
            total_text_attention_masks_4d = self.create_4d_causal_attention_mask(total_text_attention_masks)
            past_seen_tokens = 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + text_embeddings.shape[1], device=device
            ) 
            # 初始推理
            outputs = self.language_model.generate_conditions(
                attention_mask=total_text_attention_masks_4d,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=text_embeddings,
                use_cache=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
                cache_position=cache_position
            )

            # 获取模型输出的logits，第一次forward生成尺寸为[B, 1, 2560]的conditions
            predicted_conditions_embeds = outputs.last_hidden_state[:, -1:, :]

            predicted_conditions_distribution = self.distribution_predictor_head(predicted_conditions_embeds)
            bs, L, dim = predicted_conditions_distribution.shape
            predicted_conditions_distribution = predicted_conditions_distribution.reshape(bs, int(np.sqrt(L)), int(np.sqrt(L)), dim).permute(0, 3, 1, 2) # [b, c, h, w]
            predicted_conditions_dgd = DiagonalGaussianDistribution(predicted_conditions_distribution)
            next_predicted_embeds = self.flow_vae.condition_codec.decode_from_dgd(predicted_conditions_dgd)

            # infer_Predicted_semantic_conditions_all.append(predicted_conditions_dgd.sample())

            # TODO 对 current_predicted_embeds 进行插值，并且加入位置编码
            h = w = self.config.fm_condition_level_size[1]
            next_predicted_embeds = F.interpolate(next_predicted_embeds, size=(h, w), mode='nearest')
            next_predicted_embeds += self.segment_position_embeddings(self.segment_position_ids[1]).unsqueeze(2).unsqueeze(3)
            next_predicted_embeds = next_predicted_embeds.clone().reshape(batch_size, -1, h*w).permute(0, 2, 1) # [B, s, d]

            # infer_Interp_decoded_semantic_latents_all.append(current_predicted_embeds.clone())

            # 拼接：
            current_predicted_embeds = torch.cat([text_embeddings, next_predicted_embeds], dim=1)
            
            current_attention_mask = total_text_attention_masks_4d.clone() # [b, 1, seq_len, seq_len]
            # TODO 扩展attention mask: [b, 1, seq_len, seq_len] -> [b, 1, seq_len + 4, seq_len + 4]
            current_attention_mask = self.expand_attention_mask(attention_mask_4d=current_attention_mask, 
                                                                expand_size=self.config.fm_condition_level_size[1]**2,
                                                                pad_length=pad_length)
            # breakpoint()

            for i in range(len(self.config.fm_condition_level_size)-1):
                past_seen_tokens = 0
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + current_predicted_embeds.shape[1], device=device
                ) 
                outputs = self.language_model.generate_conditions(
                    attention_mask = current_attention_mask,
                    position_ids=None,
                    past_key_values=None,
                    inputs_embeds= current_predicted_embeds,
                    use_cache=None,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                    cache_position=cache_position
                )

                # 获取新的预测tokens
                length = self.config.fm_condition_level_size[i+1]**2
                predicted_conditions_embeds = outputs.last_hidden_state[:, -length:, :] # [b, s, d]
                # 计算新的current_predicted_embeds
                predicted_conditions_distribution = self.distribution_predictor_head(predicted_conditions_embeds)
                bs, L, dim = predicted_conditions_distribution.shape
                predicted_conditions_distribution = predicted_conditions_distribution.reshape(bs, int(np.sqrt(L)), int(np.sqrt(L)), dim).permute(0, 3, 1, 2) # [b, c, h, w]
                predicted_conditions_dgd = DiagonalGaussianDistribution(predicted_conditions_distribution)
                # infer_Predicted_semantic_conditions_all.append(predicted_conditions_dgd.sample())

                if i!=len(self.config.fm_condition_level_size)-2:
                    next_predicted_embeds = self.flow_vae.condition_codec.decode_from_dgd(predicted_conditions_dgd)
                    # TODO 对 current_predicted_embeds 进行插值，并且加入位置编码
                    h = w = self.config.fm_condition_level_size[i+2]
                    next_predicted_embeds = F.interpolate(next_predicted_embeds, size=(h, w), mode='nearest')
                    next_predicted_embeds += self.segment_position_embeddings(self.segment_position_ids[i+2]).unsqueeze(2).unsqueeze(3)
                    next_predicted_embeds = next_predicted_embeds.reshape(batch_size, -1, h*w).permute(0, 2, 1) # [B, s, d]

                    # infer_Interp_decoded_semantic_latents_all.append(current_predicted_embeds.clone())
                    # 拼接：
                    current_predicted_embeds = torch.cat([current_predicted_embeds, next_predicted_embeds], dim=1)
                    # TODO
                    current_attention_mask = self.expand_attention_mask(attention_mask_4d=current_attention_mask, 
                                                                    expand_size=self.config.fm_condition_level_size[i+2]**2,
                                                                    pad_length=pad_length)
   
                else:
                    break
            

            # NOTE 这里保存：infer_Interp_decoded_semantic_latents_all：infer_Interp_decoded_semantic_latents_all 和 infer_Predicted_semantic_conditions_all：infer_Predicted_semantic_conditions_all
            # torch.save(infer_Interp_decoded_semantic_latents_all, "/slurm/home/yrd/kanlab/zhangchenfeng/program/diff_gen/train_flowllm/result/4/final_checkpoint_20000_steps/infer_Interp_decoded_semantic_latents_all.pt")
            # torch.save(infer_Predicted_semantic_conditions_all, "/slurm/home/yrd/kanlab/zhangchenfeng/program/diff_gen/train_flowllm/result/4/final_checkpoint_20000_steps/infer_Predicted_semantic_conditions_all.pt")
            # breakpoint()

            with torch.autocast(device_type = device.type, dtype=predicted_conditions_embeds.dtype):
                # TODO  此时还没有实现cfg，后面再补充cfg生成
                predicted_vae_image = self.decode_vision(predicted_conditions_dgd)

        return predicted_vae_image


    def generate_cfg_with_cache(self, texts: list[str]):
        '''
        texts: list of strings
        TODO 此时还没有实现cfg生成，后面再补充cfg生成
        TODO kv cache存在问题，需要从头实现。因此这个方法暂时弃用
        '''
        device = self.device
        batch_size = len(texts)

        # 1. 将texts进行tokenize
        text_input = self.tokenizer(texts, return_tensors="pt", padding=True, padding_side='left').to(device)
        text_input_ids = text_input.input_ids
        text_attention_masks = text_input.attention_mask

        # 首先拼接上sogi token id
        soig_id = torch.tensor([[self.start_of_image_generation_id]]).to(device) # [1,1] 的 start of image generation token id
        total_text_input_ids = torch.cat([text_input_ids, soig_id.repeat(batch_size, 1)], dim=1)
        total_text_attention_masks = torch.cat([text_attention_masks, torch.ones([batch_size, 1]).to(device)], dim=1)
        pad_length = (total_text_attention_masks.shape[1] - total_text_attention_masks.sum(1)).int()
        text_embeddings = self.language_model.model.embed_tokens(total_text_input_ids)

        # 添加位置编码到soig中
        text_embeddings[:, -1, :] = self.segment_position_embeddings(self.segment_position_ids[0]) + text_embeddings[:, -1, :]

        with torch.no_grad():
            # TODO 将total_text_attention_masks转为4d attention mask
            total_text_attention_masks_4d = self.create_4d_causal_attention_mask(total_text_attention_masks)

            # 初始推理
            outputs = self.language_model.generate_conditions(
                attention_mask=total_text_attention_masks_4d,
                inputs_embeds=text_embeddings,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True
            )

            # 获取KV缓存
            past_key_values = outputs.past_key_values
            # 获取模型输出的logits，第一次forward生成尺寸为[B, 1, 2560]的conditions
            predicted_conditions_embeds = outputs.last_hidden_state[:, -1:, :]

            # 跟踪已处理的token数量
            past_seen_tokens = text_embeddings.shape[1]

            # TODO 对 current_predicted_embeds 进行插值，并且加入位置编码
            current_predicted_embeds = predicted_conditions_embeds.clone().unsqueeze(1).permute(0, 3, 1, 2) # [B, 2560, 1, 1]
            current_predicted_embeds += self.segment_position_embeddings(self.segment_position_ids[1]).unsqueeze(2).unsqueeze(3)
            h = w = self.config.fm_condition_level_size[1]
            current_predicted_embeds = F.interpolate(current_predicted_embeds, size=(h, w), mode='nearest').reshape(batch_size, -1, h*w).permute(0, 2, 1) # [B, 4, 2560]
            
            
            current_attention_mask = total_text_attention_masks_4d.clone() # [b, 1, seq_len, seq_len]
            # TODO 扩展attention mask: [b, 1, seq_len, seq_len] -> [b, 1, seq_len + 4, seq_len + 4]
            current_attention_mask = self.expand_attention_mask(attention_mask_4d=current_attention_mask, 
                                                                expand_size=self.config.fm_condition_level_size[1]**2,
                                                                pad_length=pad_length)
            
            # 注意：因为使用了kv cache，因此这里得到的4d attention mask只要保留新得到的token的部分就好了
            current_attention_mask = current_attention_mask[:, :, -self.config.fm_condition_level_size[1]**2:, :]


            for i in range(len(self.config.fm_condition_level_size)-1):

                # 创建position_ids和cache_position
                position_ids = torch.arange(
                    past_seen_tokens,
                    past_seen_tokens + current_predicted_embeds.shape[1],
                    device=device
                )

                cache_position = position_ids.clone()

                # 使用预测token更新KV缓存
                outputs = self.language_model.generate_conditions(
                    attention_mask = current_attention_mask,
                    inputs_embeds=current_predicted_embeds,
                    position_ids=position_ids.unsqueeze(0),
                    past_key_values=past_key_values,
                    use_cache=True,
                    cache_position=cache_position,
                )


                # 更新kv缓存
                past_key_values = outputs.past_key_values

                # 获取新的预测tokens
                length = self.config.fm_condition_level_size[i+1]**2
                predicted_conditions_embeds = outputs.last_hidden_state[:, -length:, :] # [b, s, d]

                # 更新已处理的token数量
                past_seen_tokens += current_predicted_embeds.shape[1]

                if i!=len(self.config.fm_condition_level_size)-2:
                    # 计算新的current_predicted_embeds和current_attention_mask
                    b, s, d = predicted_conditions_embeds.shape
                    current_predicted_embeds = predicted_conditions_embeds.reshape(b, int(np.sqrt(s)), int(np.sqrt(s)), d).permute(0, 3, 1, 2) # [B, c, h, w]
                    current_predicted_embeds += self.segment_position_embeddings(self.segment_position_ids[i+2]).unsqueeze(2).unsqueeze(3)
                    h = w = self.config.fm_condition_level_size[i+2]
                    current_predicted_embeds = F.interpolate(current_predicted_embeds, size=(h, w), mode='nearest').reshape(batch_size, -1, h*w).permute(0, 2, 1) # [B, s, d]
                    
                    
                    # TODO
                    current_attention_mask = self.expand_attention_mask(attention_mask_4d=current_attention_mask, 
                                                                    expand_size=self.config.fm_condition_level_size[i+2]**2,
                                                                    pad_length=pad_length)
                    # 注意：因为使用了kv cache，因此这里得到的4d attention mask只要保留新得到的token的部分就好了
                    current_attention_mask = current_attention_mask[:, :, -length:, :]
                
                else:
                    break

            
            pred_dgd = self.distribution_predictor_head(predicted_conditions_embeds) # [B, c, h, w]
            # TODO  此时还没有实现cfg，后面再补充cfg生成
            predicted_vae_image = self.decode_vision(pred_dgd)

        return predicted_vae_image


if __name__ == "__main__":
    a = 1
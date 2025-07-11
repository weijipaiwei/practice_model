import torch
import copy

from dataclasses import dataclass
from transformers import Qwen2VLModel, PretrainedConfig, PreTrainedModel, Qwen2_5_VLForConditionalGeneration, GenerationMixin
from typing import Any, Dict, Iterable, Optional, Union
from torch.nn import CrossEntropyLoss
from transformers.utils import ModelOutput

class EMAModel:
    """
    Exponential Moving Average of models weights
    """

    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        decay: float = 0.9999,
        min_decay: float = 0.0,
        update_after_step: int = 0,
        use_ema_warmup: bool = False,
        inv_gamma: Union[float, int] = 1.0,
        power: Union[float, int] = 2 / 3,
        model_cls: Optional[Any] = None,
        model_config: Dict[str, Any] = None,
        **kwargs,
    ):
        """
        Args:
            parameters (Iterable[torch.nn.Parameter]): The parameters to track.
            decay (float): The decay factor for the exponential moving average.
            min_decay (float): The minimum decay factor for the exponential moving average.
            update_after_step (int): The number of steps to wait before starting to update the EMA weights.
            use_ema_warmup (bool): Whether to use EMA warmup.
            inv_gamma (float):
                Inverse multiplicative factor of EMA warmup. Default: 1. Only used if `use_ema_warmup` is True.
            power (float): Exponential factor of EMA warmup. Default: 2/3. Only used if `use_ema_warmup` is True.
            device (Optional[Union[str, torch.device]]): The device to store the EMA weights on. If None, the EMA
                        weights will be stored on CPU.

        @crowsonkb's notes on EMA Warmup:
            If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are good values for models you plan
            to train for a million or more steps (reaches decay factor 0.999 at 31.6K steps, 0.9999 at 1M steps),
            gamma=1, power=3/4 for models you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999
            at 215.4k steps).
        """
        parameters = list(parameters)
        self.shadow_params = [p.clone().detach() for p in parameters]
        self.temp_stored_params = None

        self.decay = decay
        self.min_decay = min_decay
        self.update_after_step = update_after_step
        self.use_ema_warmup = use_ema_warmup
        self.inv_gamma = inv_gamma
        self.power = power
        self.optimization_step = 0
        self.cur_decay_value = None  # set in `step()`

        self.model_cls = model_cls
        self.model_config = model_config

    def get_decay(self, optimization_step: int) -> float:
        """
        Compute the decay factor for the exponential moving average.
        """
        step = max(0, optimization_step - self.update_after_step - 1)

        if step <= 0:
            return 0.0

        if self.use_ema_warmup:
            cur_decay_value = 1 - (1 + step / self.inv_gamma) ** -self.power
        else:
            cur_decay_value = (1 + step) / (10 + step)

        cur_decay_value = min(cur_decay_value, self.decay)
        # make sure decay is not smaller than min_decay
        cur_decay_value = max(cur_decay_value, self.min_decay)
        return cur_decay_value

    @torch.no_grad()
    def step(self, parameters: Iterable[torch.nn.Parameter]):
        parameters = list(parameters)

        self.optimization_step += 1

        # Compute the decay factor for the exponential moving average.
        decay = self.get_decay(self.optimization_step)
        self.cur_decay_value = decay
        one_minus_decay = 1 - decay
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                s_param.sub_(one_minus_decay * (s_param - param))
            else:
                s_param.copy_(param)

    def to(self, device=None, dtype=None) -> None:
        r"""Move internal buffers of the ExponentialMovingAverage to `device`.

        Args:
            device: like `device` argument to `torch.Tensor.to`
        """
        # .to() on the tensors handles None correctly
        self.shadow_params = [
            p.to(device=device, dtype=dtype) if p.is_floating_point() else p.to(device=device)
            for p in self.shadow_params
        ]
        return self

    def state_dict(self) -> dict:
        r"""
        Returns the state of the ExponentialMovingAverage as a dict. This method is used by accelerate during
        checkpointing to save the ema state dict.
        """
        return {
            "decay": self.decay,
            "min_decay": self.min_decay,
            "optimization_step": self.optimization_step,
            "update_after_step": self.update_after_step,
            "use_ema_warmup": self.use_ema_warmup,
            "inv_gamma": self.inv_gamma,
            "power": self.power,
            "shadow_params": self.shadow_params,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        r"""
        Args:
        Loads the ExponentialMovingAverage state. This method is used by accelerate during checkpointing to save the
        ema state dict.
            state_dict (dict): EMA state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        state_dict = copy.deepcopy(state_dict)

        self.decay = state_dict.get("decay", self.decay)
        if self.decay < 0.0 or self.decay > 1.0:
            raise ValueError("Decay must be between 0 and 1")

        self.min_decay = state_dict.get("min_decay", self.min_decay)
        if not isinstance(self.min_decay, float):
            raise ValueError("Invalid min_decay")

        self.optimization_step = state_dict.get("optimization_step", self.optimization_step)
        if not isinstance(self.optimization_step, int):
            raise ValueError("Invalid optimization_step")

        self.update_after_step = state_dict.get("update_after_step", self.update_after_step)
        if not isinstance(self.update_after_step, int):
            raise ValueError("Invalid update_after_step")

        self.use_ema_warmup = state_dict.get("use_ema_warmup", self.use_ema_warmup)
        if not isinstance(self.use_ema_warmup, bool):
            raise ValueError("Invalid use_ema_warmup")

        self.inv_gamma = state_dict.get("inv_gamma", self.inv_gamma)
        if not isinstance(self.inv_gamma, (float, int)):
            raise ValueError("Invalid inv_gamma")

        self.power = state_dict.get("power", self.power)
        if not isinstance(self.power, (float, int)):
            raise ValueError("Invalid power")

        shadow_params = state_dict.get("shadow_params", None)
        for model_param, ema_param in zip(self.shadow_params, shadow_params):
                model_param.data = ema_param.data.to(model_param)


@dataclass
class MyModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None


# TODO 最后根据参数修改
class MyModelConfig(PretrainedConfig):
    model_type = "MyModel"
    def __init__(self,
                 base_model_name_or_path: str = "Qwen/Qwen2.5-VL-3B-Instruct",
                 torch_dtype: str = "bfloat16",
                 attn_implementation: str = "flash_attention_2",
                 **kwargs):
        super().__init__(**kwargs)
        self.base_model_name_or_path = base_model_name_or_path
        self.torch_dtype = torch_dtype
        self.attn_implementation = attn_implementation



class MyModel(PreTrainedModel, GenerationMixin):
    config_class = MyModelConfig
    
    def __init__(self, config: MyModelConfig):
        super().__init__(config)

        self.config = config
        
        # 加载VLM
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.base_model_name_or_path,
            torch_dtype=config.torch_dtype,
            attn_implementation=config.attn_implementation,
            device_map=None  # 在分布式训练中不能使用device_map="auto"
        )
        
    def forward(self,
                input_ids,
                attention_mask=None,
                pixel_values=None,
                image_grid_thw=None,
                return_dict=True,
                labels=None,
                **kwargs):
        """
        前向传播函数
        
        Args:
            input_ids: 输入ids
            attention_mask: 注意力掩码
            pixel_values: 图像输入
            image_grid_thw: 图像网格
            return_dict: 是否返回字典
            labels: 标签（用于训练）
        """
        # 使用基础模型进行前向传播
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            return_dict=return_dict,
            **kwargs
        )

        loss = None
        if labels is not None:
            logits = outputs.logits
            logits = logits.float()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        # 返回结果字典，包含损失和其他信息
        return MyModelOutput(
            loss=loss,
            logits=outputs.logits,
        )
    
    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, pixel_values=None, image_grid_thw=None, **kwargs):
        """
        为生成准备输入
        """
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            **kwargs
        }
    
    def get_output_embeddings(self):
        """
        获取输出嵌入层
        """
        return self.model.get_output_embeddings()
    
    def set_output_embeddings(self, new_embeddings):
        """
        设置输出嵌入层
        """
        return self.model.set_output_embeddings(new_embeddings)
    
    def get_encoder(self):
        """
        获取编码器
        """
        return self.model.get_encoder()
    
    def get_decoder(self):
        """
        获取解码器
        """
        return self.model.get_decoder()
    
    def _reorder_cache(self, past, beam_idx):
        """
        重新排序缓存
        """
        return self.model._reorder_cache(past, beam_idx)



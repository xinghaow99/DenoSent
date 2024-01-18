from transformers import PretrainedConfig
from typing import Optional

class DenoSentConfig(PretrainedConfig):
    def __init__(self,
                encoder_name_or_path:Optional[str]=None,
                hidden_size:Optional[int]=768,
                max_length:Optional[int]=32,
                decoder_num_heads:Optional[int]=1,
                decoder_num_layers:Optional[int]=16,
                decoder_noise_dropout:Optional[float]=0.825,
                pooler:Optional[str]='mask',
                do_contrastive:Optional[bool]=False,
                do_generative:Optional[bool]=False,
                prompt_format:Optional[str]='[X] means [MASK]',
                contrastive_weight:Optional[float]=1.0,
                generative_weight:Optional[float]=1.0,
                contrastive_temp: Optional[float]=0.05,
                **kwargs):
        super().__init__(**kwargs)
        self.encoder_name_or_path = encoder_name_or_path
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.decoder_num_heads = decoder_num_heads
        self.decoder_num_layers = decoder_num_layers
        self.decoder_noise_dropout = decoder_noise_dropout
        self.pooler = pooler
        self.do_contrastive = do_contrastive
        self.do_generative = do_generative
        self.prompt_format = prompt_format
        self.contrastive_weight = contrastive_weight
        self.generative_weight = generative_weight
        self.contrastive_temp = contrastive_temp
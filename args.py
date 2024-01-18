from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union, Dict

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default='bert-base-uncased'
    )
    max_length: Optional[int] = field(
        default=32
    )
    pooler: Optional[str] = field(
        default='cls'
    )
    prompt_format: Optional[str] = field(
        default='"[X]" means [MASK].'
    )
    decoder_num_layers: Optional[int] = field(
        default=16
    )
    decoder_num_heads: Optional[int] = field(
        default=1
    )
    decoder_target_dropout: Optional[float] = field(
        default=0.825
    )

    do_contrastive: Optional[bool] = field(
        default=False
    )
    do_generative: Optional[bool] = field(
        default=False
    )
    contrastive_temp: Optional[float] = field(
        default=0.05
    )
    contrastive_weight: Optional[float] = field(
        default=1.0
    )
    generative_weight: Optional[float] = field(
        default=1.0
    )


@dataclass
class DatasetArguments:
    train_dataset: Optional[str] = field(
        # Singhoo/stssickr, princeton-nlp/datasets-for-simcse, bookcorpus
        default='Singhoo/denosent_data',
        metadata={
            'help': 'Can be princeton-nlp/datasets-for-simcse, wiki1m-aug, wiki1m-aug-cleaned, Singhoo/wiki1m_translated, Singhoo/stssickr, bookcorpus.'
        }
    )
    split: Optional[str] = field(
        default='train'
    )
    use_auth_token: Optional[bool] = field(
        default=False
    )
    group: Optional[str] = field(
        default=None
    )

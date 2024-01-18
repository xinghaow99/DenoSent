from transformers import AutoTokenizer, BertForMaskedLM
from transformers.models.bert.modeling_bert import BertForMaskedLM
from transformers.modeling_outputs import TokenClassifierOutput
from transformers import PreTrainedModel
import torch
from torch import nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer

from typing import Optional

import wandb
import numpy as np

class DenoSentModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.pooler = config.pooler
        self.sent_embedding_projector = nn.Linear(config.hidden_size, config.hidden_size)
        self.decoder = TransformerDecoder(TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.decoder_num_heads, batch_first=True, dropout=0.1), num_layers=config.decoder_num_layers)
        self.decoder_noise_dropout = nn.Dropout(config.decoder_noise_dropout)
        self.sim = nn.CosineSimilarity(dim=-1)
        self.init_weights()
        self.tokenizer = AutoTokenizer.from_pretrained(config.encoder_name_or_path)
        self.encoder = BertForMaskedLM.from_pretrained(config.encoder_name_or_path)
        self.prediction_head = self.encoder.cls
        self.encoder = self.encoder.bert
        self.post_init()

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def encode(self, sentences, batch_size=32, **kwargs):
        """ Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """
        self.eval()
        all_embeddings = []
        length_sorted_idx = np.argsort([len(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
        if self.config.pooler == 'mask':
            prompt_length = len(self.tokenizer(self.config.prompt_format, add_special_tokens=False)['input_ids'])
            sentences_sorted = self.tokenizer.batch_decode(self.tokenizer(sentences_sorted, padding=True, truncation=True, max_length=self.config.max_length, return_tensors='pt').input_ids, skip_special_tokens=True)
            sentences_sorted = [self.config.prompt_format.replace('[X]', s).replace('[MASK]', self.tokenizer.mask_token) for s in sentences_sorted]
        for start_index in range(0, len(sentences), batch_size):
            sentences_batch = sentences_sorted[start_index:start_index+batch_size]
            inputs = self.tokenizer(sentences_batch, padding='max_length', truncation=True, return_tensors="pt", max_length=self.config.max_length+prompt_length)
            inputs = {k: v.to(self.device) for k,v in inputs.items()}
            with torch.no_grad():
                encoder_outputs = self.encoder(**inputs, output_hidden_states=True, output_attentions=True, return_dict=True)
                last_hidden_state = encoder_outputs.last_hidden_state
                if self.config.pooler == 'cls':
                    embeddings = last_hidden_state[:, 0, :]
                elif self.config.pooler == 'mean':
                    embeddings = (last_hidden_state * inputs['attention_mask'].unsqueeze(-1)).sum(1) / inputs['attention_mask'].sum(-1).unsqueeze(-1)
                elif self.pooler == 'mask':
                    embeddings = last_hidden_state[inputs['input_ids'] == self.tokenizer.mask_token_id]
                else:
                    raise NotImplementedError()
            all_embeddings.extend(embeddings.cpu().numpy())
        all_embeddings = torch.tensor(np.array([all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]))
        return all_embeddings

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            positive_input_ids: Optional[torch.LongTensor] = None,
            positive_attention_mask: Optional[torch.LongTensor] = None,
            negative_input_ids: Optional[torch.LongTensor] = None,
            negative_attention_mask: Optional[torch.LongTensor] = None,
            global_step: Optional[int] = None,
            max_steps: Optional[int] = None,
    ):
        batch_size = input_ids.size(0)
        if negative_input_ids is not None:
            encoder_input_ids = torch.cat([input_ids, positive_input_ids, negative_input_ids], dim=0).to(self.device)
            encoder_attention_mask = torch.cat([attention_mask, positive_attention_mask, negative_attention_mask], dim=0).to(self.device)
        elif positive_input_ids is not None:
            encoder_input_ids = torch.cat([input_ids, positive_input_ids], dim=0).to(self.device)
            encoder_attention_mask = torch.cat([attention_mask, positive_attention_mask], dim=0).to(self.device)
        elif self.config.do_contrastive:
            encoder_input_ids = torch.cat([input_ids, input_ids], dim=0).to(self.device)
            encoder_attention_mask = torch.cat([attention_mask, attention_mask], dim=0).to(self.device)
        elif self.config.do_generative and not self.config.do_contrastive:
            encoder_input_ids = input_ids.to(self.device)
            encoder_attention_mask = attention_mask.to(self.device)
        else:
            raise NotImplementedError()
        encoder_outputs = self.encoder(input_ids=encoder_input_ids, attention_mask=encoder_attention_mask, return_dict=True, output_hidden_states=True, output_attentions=True)
        if self.pooler == 'cls':
            sent_embedding = encoder_outputs.last_hidden_state[:, 0, :]
        elif self.pooler == 'mean':
            sent_embedding = ((encoder_outputs.last_hidden_state * encoder_attention_mask.unsqueeze(-1)).sum(1) / encoder_attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler == 'mask':
            sent_embedding = encoder_outputs.last_hidden_state[encoder_input_ids == self.tokenizer.mask_token_id]
        else:
            raise NotImplementedError()
        sent_embedding = sent_embedding.unsqueeze(1)
        sent_embedding = self.sent_embedding_projector(sent_embedding)

        if self.config.do_generative:
            if positive_input_ids is not None:
                tgt = encoder_outputs.hidden_states[0][batch_size:2*batch_size].detach()
                tgt_key_padding_mask = (positive_input_ids == self.tokenizer.pad_token_id)
                labels = positive_input_ids
            else:
                tgt = encoder_outputs.hidden_states[0][:batch_size].detach()
                tgt_key_padding_mask = (input_ids == self.tokenizer.pad_token_id)
                labels = input_ids
            tgt = self.decoder_noise_dropout(tgt)
            decoder_outputs = self.decoder(tgt=tgt, memory=sent_embedding[:batch_size], tgt_mask=None, tgt_key_padding_mask=tgt_key_padding_mask)
            logits = self.prediction_head(decoder_outputs)
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            generative_loss = loss_fct(logits.view(-1, self.encoder.config.vocab_size), labels.view(-1))
            wandb.log({'train/generative_loss': generative_loss})

        if self.config.do_contrastive:
            positive_sim = self.sim(sent_embedding[:batch_size], sent_embedding[batch_size:2*batch_size].transpose(0, 1))
            cos_sim = positive_sim
            if negative_attention_mask is not None:
                negative_sim = self.sim(sent_embedding[:batch_size], sent_embedding[2*batch_size:].transpose(0, 1))
                cos_sim = torch.cat([positive_sim, negative_sim], dim=1)
            cos_sim = cos_sim / self.config.contrastive_temp
            contrastive_labels = torch.arange(batch_size, dtype=torch.long, device=self.device)
            contrastive_loss = nn.CrossEntropyLoss()(cos_sim, contrastive_labels)
            wandb.log({'train/contrastive_loss': contrastive_loss.item()})
            logits = None
        loss = 0
        if self.config.do_contrastive:
            loss += self.config.contrastive_weight * contrastive_loss
        if self.config.do_generative:
            loss += self.config.generative_weight * generative_loss
        wandb.log({'train/loss': loss})
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

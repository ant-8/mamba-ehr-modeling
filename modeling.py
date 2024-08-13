from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import Mamba2PreTrainedModel, Mamba2Model

class Time2Vec(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Time2Vec, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.w0 = nn.Parameter(torch.randn(input_dim))
        self.b0 = nn.Parameter(torch.randn(input_dim))

        self.w = nn.Parameter(torch.randn(input_dim, output_dim - 1))
        self.b = nn.Parameter(torch.randn(input_dim, output_dim - 1))

    def forward(self, x):
        v = self.w0 * x + self.b0
        periodic_terms = torch.sin(self.w * x.unsqueeze(-1) + self.b)
        v = torch.cat([v.unsqueeze(-1), periodic_terms], dim=-1)
        return v

class TemporalEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim, proj_dim):
        super(TemporalEmbedding, self).__init__()
        self.time2vec = Time2Vec(input_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, proj_dim)

    def forward(self, x):
        time_embedding = self.time2vec(x)
        projected_embedding = self.proj(time_embedding)
        return projected_embedding

class Mamba2CausalLMOutput():
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None

    def __init__(self, loss, logits):
        self.loss = loss
        self.logits = logits

class Mamba2ForEHRModeling(Mamba2PreTrainedModel):
    _tied_weights_keys = []

    def __init__(self, config):
        super().__init__(config)
        self.backbone = Mamba2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        self.segment_embeddings = nn.Embedding(2, config.hidden_size)

        #self.age_embeddings = nn.Embedding(73, config.hidden_size) # 18 - 91
        self.age_embeddings = TemporalEmbedding(input_dim=1, embed_dim=32, proj_dim=config.hidden_size)
        self.time_embeddings = TemporalEmbedding(input_dim=1, embed_dim=32, proj_dim=config.hidden_size)
        self.visit_order_embeddings = TemporalEmbedding(input_dim=1, embed_dim=32, proj_dim=config.hidden_size)
        
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.backbone.set_input_embeddings(new_embeddings)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        age_ids: Optional[torch.LongTensor] = None,
        time_ids: Optional[torch.LongTensor] = None,
        segment_ids: Optional[torch.LongTensor] = None,
        visit_order_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, Mamba2CausalLMOutput]:
        assert input_ids.shape == age_ids.shape and \
            age_ids.shape == time_ids.shape and \
            time_ids.shape == segment_ids.shape and \
            segment_ids.shape == visit_order_ids.shape
        
        input_embeds = self.backbone.embeddings(input_ids)

        zero_vector = torch.zeros(self.config.hidden_size, device=input_embeds.device)

        adjusted_age_ids = age_ids - 18
        age_embeds = self.age_embeddings(torch.clamp(adjusted_age_ids, min=0))
        mask = (adjusted_age_ids < 0).unsqueeze(-1).expand_as(age_embeds)
        age_embeds = torch.where(mask, zero_vector, age_embeds)
        input_embeds += age_embeds

        adjusted_segment_ids = segment_ids - 1
        seg_embeds = self.segment_embeddings(torch.clamp(adjusted_segment_ids, min=0))
        mask = (adjusted_segment_ids < 0).unsqueeze(-1).expand_as(seg_embeds)
        seg_embeds = torch.where(mask, zero_vector, seg_embeds)
        input_embeds += seg_embeds

        time_embeds = self.time_embeddings(torch.clamp(time_ids, min=0))
        mask = (time_ids < 0).unsqueeze(-1).expand_as(time_embeds)
        time_embeds = torch.where(mask, zero_vector, time_embeds)
        input_embeds += time_embeds

        visit_order_embeds = self.visit_order_embeddings(torch.clamp(visit_order_ids, min=0))
        mask = (visit_order_ids < 0).unsqueeze(-1).expand_as(visit_order_embeds)
        visit_order_embeds = torch.where(mask, zero_vector, visit_order_embeds)
        input_embeds += visit_order_embeds

        mamba2_outputs = self.backbone(
            input_ids=None,
            inputs_embeds=input_embeds,
            output_hidden_states=output_hidden_states,
            attention_mask=attention_mask,
        )
        hidden_states = mamba2_outputs[0].to(self.lm_head.weight.dtype)

        logits = self.lm_head(hidden_states).float()

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return Mamba2CausalLMOutput(
            loss=loss,
            logits=logits
        )
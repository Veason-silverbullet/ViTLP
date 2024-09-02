# The Transformer code is adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/bart/modeling_bart.py
import copy
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.activations import ACT2FN
from transformers.file_utils import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from .configuration_ViTLP import ViTLPConfig
from .utils_ViTLP import load_pretrained_vit, load_pretrained_bart
from typing import Optional, Tuple


################################ Utility Functions ################################
def _make_causal_mask(input_ids: torch.Tensor, dtype: torch.dtype, device: torch.device):
    bsz, tgt_len = input_ids.size()
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min), device=device, dtype=dtype)
    mask_cond = torch.arange(mask.size(-1), device=device, dtype=torch.int16)
    mask.masked_fill_(mask_cond <= mask_cond.view(mask.size(-1), 1), 0)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len)


def calculate_IOU(prediction_bbox, label_bbox):
    prediction_bbox[:, 0] = torch.minimum(prediction_bbox[:, 0], prediction_bbox[:, 2])
    prediction_bbox[:, 1] = torch.minimum(prediction_bbox[:, 1], prediction_bbox[:, 3])
    x_left = torch.maximum(prediction_bbox[:, 0], label_bbox[:, 0])
    y_top = torch.maximum(prediction_bbox[:, 1], label_bbox[:, 1])
    x_right = torch.minimum(prediction_bbox[:, 2], label_bbox[:, 2])
    y_bottom = torch.minimum(prediction_bbox[:, 3], label_bbox[:, 3])

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    prediction_bbox_area = (prediction_bbox[:, 2] - prediction_bbox[:, 0]) * (prediction_bbox[:, 3] - prediction_bbox[:, 1])
    label_bbox_area = (label_bbox[:, 2] - label_bbox[:, 0]) * (label_bbox[:, 3] - label_bbox[:, 1])

    iou = intersection_area / (prediction_bbox_area + label_bbox_area - intersection_area)
    iou.masked_fill_((x_right < x_left) | (y_bottom < y_top), 0)
    return iou.mean()


################################ VIT patch embedding ################################
class ViTPatchEmbeddings(nn.Module):
    def __init__(self, patch_size=32, num_channels=3, embed_dim=1024):
        super().__init__()
        self.projection = nn.Conv2d(num_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size))

    def forward(self, pixel_values):
        x = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return x


################################ Super Architecture ################################
class VitlpPretrainedModel(PreTrainedModel):
    config_class = ViTLPConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_unexpected = [r'encoder\.version', r'decoder\.version']

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (VitlpDecoder, VitlpEncoder)):
            module.gradient_checkpointing = value


class ViTLPPreTrainedModel(PreTrainedModel):
    config_class = ViTLPConfig
    base_model_prefix = 'vitlp'
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r'position_ids']

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, ViTLPModel):
            module.gradient_checkpointing = value


################################ Pretraining Architecture ################################
class VitlpAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads}).')
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # if key_value_states are provided this layer is used as a cross-attention layer for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()
        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k, v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first 'if' case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third 'elif' case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)
        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(f'Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}')
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(f'Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}')
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)
        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(f'`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}')
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, past_key_value


# This image encoder strictly follows VIT (https://github.com/huggingface/transformers/blob/v4.20.1/src/transformers/models/vit/modeling_vit.py)
class VitlpEncoderLayer(nn.Module):
    def __init__(self, config: ViTLPConfig):
        super().__init__()
        self.embed_dim = config.encoder_hidden_size
        self.self_attn = VitlpAttention(embed_dim=self.embed_dim, num_heads=config.encoder_attention_heads, dropout=config.attention_dropout)
        self.intermediate = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.output = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.layernorm_before = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.activation_fn = ACT2FN[config.activation_function]
        self.dropout = config.dropout

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor):
        # 1. ViTAttention
        residual = hidden_states
        hidden_states, _ = self.self_attn(
            hidden_states=self.layernorm_before(hidden_states),
            attention_mask=attention_mask
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layernorm_after(hidden_states)

        # 2. ViTIntermediate
        hidden_states = self.activation_fn(self.intermediate(hidden_states))

        # 3. ViTOutput
        hidden_states = self.output(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16 and (torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        outputs = (hidden_states,)
        return outputs


class VitlpDecoderLayer(nn.Module):
    def __init__(self, config: ViTLPConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        # self.activation_dropout = config.activation_dropout

        self.self_attn = VitlpAttention(embed_dim=self.embed_dim, num_heads=config.decoder_attention_heads, dropout=config.attention_dropout, is_decoder=True)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.encoder_attn = VitlpAttention(embed_dim=self.embed_dim, num_heads=config.decoder_attention_heads, dropout=config.attention_dropout, is_decoder=True)
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache = False
    ):
        residual = hidden_states
        # Self-Attention Block
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        residual = hidden_states
        # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
        cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
        hidden_states, cross_attn_present_key_value = self.encoder_attn(
            hidden_states=hidden_states,
            key_value_states=encoder_hidden_states,
            past_key_value=cross_attn_past_key_value,
            attention_mask=encoder_attention_mask
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # add cross-attn to positions 3,4 of present_key_value tuple
        present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        outputs = (hidden_states,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


class VitlpEncoder(VitlpPretrainedModel):
    def __init__(self, config: ViTLPConfig):
        super().__init__(config)
        self.patch_size = config.patch_size
        self.image_height = config.image_height
        self.image_width = config.image_width
        assert self.image_height % self.patch_size == 0 and self.image_width % self.patch_size == 0 and (self.image_height // self.patch_size) * (self.image_width // self.patch_size) == config.patch_num

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.encoder_hidden_size))                                   # VIT CLS token embeddings
        self.patch_embeddings = ViTPatchEmbeddings(config.patch_size, config.num_channels, config.encoder_hidden_size) # Image patch embeddings
        self.position_embeddings = nn.Parameter(torch.zeros(1, config.patch_num + 1, config.encoder_hidden_size))
        self.layers = nn.ModuleList([VitlpEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm = nn.LayerNorm(config.encoder_hidden_size, eps=config.layer_norm_eps)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropout_ = nn.Dropout(config.hidden_dropout_prob / 2)
        self.gradient_checkpointing = config.gradient_checkpointing
        self.post_init()

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        npatch = embeddings.shape[1] - 1
        N = self.position_embeddings.shape[1] - 1
        if npatch == N:
            return self.position_embeddings
        class_pos_embed = self.position_embeddings[:, 0]
        patch_pos_embed = self.position_embeddings[:, 1:]
        dim = embeddings.shape[-1]
        assert self.image_height % self.patch_size == 0 and self.image_width % self.patch_size == 0 and height % self.patch_size == 0 and width % self.patch_size == 0
        h0 = self.image_height // self.patch_size
        w0 = self.image_width // self.patch_size
        h1 = height // self.patch_size
        w1 = width // self.patch_size
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, h0, w0, dim).permute(0, 3, 1, 2),
            scale_factor=(h1 / h0, w1 / w0),
            mode='bicubic',
            align_corners=False
        )
        assert h1 == patch_pos_embed.shape[-2] and w1 == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def _calc_img_embeddings(self, pixel_values):
        batch_size = pixel_values.size(0)
        cls_embeddings = self.cls_token.expand(batch_size, -1, -1)
        patch_embeddings = self.patch_embeddings(pixel_values)
        embeddings = torch.cat([cls_embeddings, patch_embeddings], dim=1)
        embeddings = embeddings + self.interpolate_pos_encoding(embeddings, pixel_values.size(2), pixel_values.size(3))
        embeddings = self.dropout(embeddings)
        return embeddings

    def forward(self, image: torch.Tensor):
        hidden_states = self._calc_img_embeddings(pixel_values=image)

        for encoder_layer in self.layers:
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                layer_outputs = torch.utils.checkpoint.checkpoint(create_custom_forward(encoder_layer), hidden_states, None)
            else:
                layer_outputs = encoder_layer(hidden_states, None)
            hidden_states = layer_outputs[0]

        hidden_states = hidden_states[:, 1:, :]
        hidden_states = self.layernorm(hidden_states)
        hidden_states = self.dropout_(hidden_states)
        return ModelOutput(last_hidden_state=hidden_states, attentions=None)


class ViTLPLMDecoder(VitlpPretrainedModel):
    def __init__(self, config, word_embeddings, embed_positions, layernorm_embedding, layers, lm_head, bbox_input_embeddings_x, bbox_input_embeddings_y, bbox_decoder_embedding, bbox_output_embeddings, bbox_decoder_start_embedding, bbox_decoder, bbox_head):
        super().__init__(config)
        self.word_embeddings = word_embeddings
        self.embed_positions = embed_positions
        self.layernorm_embedding = layernorm_embedding
        self.layers = layers
        self.lm_head = lm_head
        self.bbox_input_embeddings_x = bbox_input_embeddings_x
        self.bbox_input_embeddings_y = bbox_input_embeddings_y
        self.bbox_decoder_embedding = bbox_decoder_embedding
        self.bbox_output_embeddings = bbox_output_embeddings
        self.bbox_decoder_start_embedding = bbox_decoder_start_embedding
        self.bbox_decoder = bbox_decoder
        self.bbox_head = bbox_head

        self.config.is_encoder_decoder = True
        self.prepare_inputs_for_generation = self.prepare_inputs_for_inference
        self.dropout = config.dropout
        self.register_buffer('position_ids', torch.arange(config.seq_length, dtype=torch.int32).expand((1, -1)))
        self.hidden_size = config.hidden_size
        self.LOCATE_ID = 50265 # hard-code confirmation
        self.gradient_checkpointing = config.gradient_checkpointing

    def prepare_inputs_for_inference(self, input_ids: torch.Tensor, past=None, use_cache: bool = None, **model_kwargs):
        if past is not None:
            input_ids = input_ids[:, -1:]
        output = {
            'encoder_outputs': model_kwargs['encoder_outputs'].last_hidden_state,
            'decoder_input_ids': input_ids,
            'past_key_values': past,
            'use_cache': use_cache
        }
        return output

    @staticmethod
    def legal_mask_check(decoder_attention_mask):
        decoder_attention_mask = decoder_attention_mask.squeeze(dim=1)
        batch_size = decoder_attention_mask.size(0)
        seq_len = decoder_attention_mask.size(1)
        decoder_attention_mask = decoder_attention_mask.cpu().tolist()
        for i in range(batch_size):
            for j in range(seq_len):
                for k in range(seq_len):
                    if k <= j:
                        assert decoder_attention_mask[i][j][k] == 0, 'illegal decoder_attention_mask : ' + str(float(decoder_attention_mask[i][j][k]))
                    else:
                        assert decoder_attention_mask[i][j][k] < -5e4, 'illegal decoder_attention_mask : ' + str(float(decoder_attention_mask[i][j][k]))
        print('The decoder_attention_mask is legal')
        exit()

    def forward_(self, encoder_outputs, decoder_input_ids, decoder_input_bboxes, past_key_values=None, use_cache=False):
        seq_length = decoder_input_ids.size(1)
        if past_key_values is None:
            decoder_position_ids = self.position_ids[:, :seq_length].expand(decoder_input_ids.size())
        else:
            past_key_values_length = past_key_values[0][0].shape[2]
            decoder_position_ids = self.position_ids[:, past_key_values_length: past_key_values_length + 1].expand(decoder_input_ids.size())
        word_embeddings = self.word_embeddings(decoder_input_ids)
        bbox_input_embeddings_x1 = self.bbox_input_embeddings_x(decoder_input_bboxes.select(dim=2, index=0)) # [batch_size, seq_length, hidden_size // 4]
        bbox_input_embeddings_y1 = self.bbox_input_embeddings_y(decoder_input_bboxes.select(dim=2, index=1)) # [batch_size, seq_length, hidden_size // 4]
        bbox_input_embeddings_x2 = self.bbox_input_embeddings_x(decoder_input_bboxes.select(dim=2, index=2)) # [batch_size, seq_length, hidden_size // 4]
        bbox_input_embeddings_y2 = self.bbox_input_embeddings_y(decoder_input_bboxes.select(dim=2, index=3)) # [batch_size, seq_length, hidden_size // 4]
        bbox_input_embeddings = torch.cat([bbox_input_embeddings_x1, bbox_input_embeddings_y1, bbox_input_embeddings_x2, bbox_input_embeddings_y2], dim=2) # [batch_size, seq_length, hidden_size]
        decoder_embeds = torch.where((decoder_input_ids == self.LOCATE_ID).unsqueeze(dim=2), bbox_input_embeddings + self.bbox_decoder_embedding, word_embeddings) + self.embed_positions(decoder_position_ids)
        hidden_states = self.layernorm_embedding(decoder_embeds)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        if past_key_values is None:
            decoder_attention_mask = _make_causal_mask(decoder_input_ids, encoder_outputs.dtype, encoder_outputs.device)
            # self.legal_mask_check(decoder_attention_mask)
        else:
            decoder_attention_mask = None

        next_decoder_cache = () if use_cache else None
        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            if self.gradient_checkpointing and self.training and idx < 5: # pre-training, we only have 24G memory per GPU, save it (and me)!
            # if self.gradient_checkpointing and self.training: # fine-tuning
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                layer_outputs = torch.utils.checkpoint.checkpoint(create_custom_forward(decoder_layer), hidden_states, decoder_attention_mask, encoder_outputs, None, past_key_value, use_cache)
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=decoder_attention_mask,
                    encoder_hidden_states=encoder_outputs,
                    encoder_attention_mask=None,
                    past_key_value=past_key_value,
                    use_cache=use_cache
                )
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[1],)
        if use_cache:
            return hidden_states, next_decoder_cache
        return hidden_states

    def forward_one_step(self, encoder_outputs, hidden_states):
        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=None,
                encoder_hidden_states=encoder_outputs,
                encoder_attention_mask=None,
                past_key_value=None
            )
            hidden_states = layer_outputs[0]
        return hidden_states.squeeze(dim=1)

    # encoder_outputs      : [batch_size, hidden_size]
    # decoder_input_ids    : [batch_size, max_length]
    # decoder_input_bboxes : [batch_size, max_length, 4]
    # bbox_input_ids       : [batch_size, max_length, 3]
    def forward(self, encoder_outputs, decoder_input_ids, decoder_input_bboxes, bbox_input_ids, past_key_values=None, return_dict=False, use_cache=None, output_attentions=False, output_hidden_states=False):
        batch_seq_length = decoder_input_ids.numel()
        hidden_states = self.forward_(encoder_outputs, decoder_input_ids, decoder_input_bboxes, past_key_values)
        lm_logits = self.lm_head(hidden_states)
        bbox_output_embeddings = self.bbox_output_embeddings(bbox_input_ids).view([batch_seq_length, 3, self.hidden_size])
        bbox_logits = []
        for i in range(4):
            if i == 0:
                h = self.bbox_decoder(self.bbox_decoder_start_embedding.expand(batch_seq_length, -1), hidden_states.view([batch_seq_length, self.hidden_size]), i) # pre-training
                # h = self.bbox_decoder(self.bbox_decoder_start_embedding.expand(batch_seq_length, -1), hidden_states.detach().view([batch_seq_length, self.hidden_size]), i) # fine-tuning
            else:
                h = self.bbox_decoder(bbox_output_embeddings[:, i - 1, :], h, i)
            bbox_logits.append(self.bbox_head[i % 2](h))
        bbox_logits = torch.stack(bbox_logits, dim=1)
        return lm_logits, bbox_logits

    # encoder_outputs   : [batch_size, hidden_size]
    # decoder_input_ids : [batch_size, max_length]
    def __forward__(self, encoder_outputs, decoder_input_ids, decoder_input_bboxes, past_key_values=None, return_dict=False, use_cache=None, output_attentions=False, output_hidden_states=False):
        hidden_states = self.forward_(encoder_outputs, decoder_input_ids, decoder_input_bboxes)
        lm_logits = self.lm_head(hidden_states)
        return lm_logits


class _RNNCell_(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hh = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=True), nn.Linear(hidden_size, hidden_size, bias=True)])
        self.gelu = nn.GELU()

    def forward(self, x, h, step):
        return self.gelu(x + self.hh[step % 2](h))


class VitlpDecoder(VitlpPretrainedModel):
    def __init__(self, config: ViTLPConfig):
        super().__init__(config)
        self.bin_size = config.bin_size
        # based LM parameters
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.embed_positions = nn.Embedding(config.seq_length, config.hidden_size)
        self.layernorm_embedding = nn.LayerNorm(config.hidden_size)
        self.layers = nn.ModuleList([VitlpDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bbox_decoder_embedding = nn.Parameter(torch.zeros(1, config.hidden_size))
        # bbox parameters
        self.bbox_input_embeddings_x = nn.Embedding(self.bin_size + 1, config.hidden_size // 4, padding_idx=self.bin_size)
        self.bbox_input_embeddings_y = nn.Embedding(self.bin_size + 1, config.hidden_size // 4, padding_idx=self.bin_size)
        self.bbox_output_embeddings = nn.Embedding(self.bin_size + 1, config.hidden_size, padding_idx=self.bin_size)
        self.bbox_decoder_start_embedding = nn.Parameter(torch.zeros(1, config.hidden_size))
        self.bbox_decoder = _RNNCell_(hidden_size=config.hidden_size)
        self.bbox_head = nn.ModuleList([nn.Linear(config.hidden_size, self.bin_size, bias=False), nn.Linear(config.hidden_size, self.bin_size, bias=False)])
        # ViTLP decoder
        self.lm_decoder = ViTLPLMDecoder(config, self.word_embeddings, self.embed_positions, self.layernorm_embedding, self.layers, self.lm_head, self.bbox_input_embeddings_x, self.bbox_input_embeddings_y, self.bbox_decoder_embedding, self.bbox_output_embeddings, self.bbox_decoder_start_embedding, self.bbox_decoder, self.bbox_head)
        self.post_init()
        for i in range(2):
            nn.init.eye_(self.bbox_decoder.hh[i].weight)
            self.bbox_decoder.hh[i].bias.data.zero_()

    def extend_word_embedding(self):
        assert id(self.word_embeddings.weight) == id(self.lm_head.weight)
        token_embedding = torch.zeros([1, self.word_embeddings.weight.size(1)], device=self.word_embeddings.weight.device)
        token_embedding.normal_(mean=0.0, std=0.02)
        self.word_embeddings.weight = nn.Parameter(torch.cat([self.word_embeddings.weight.data, token_embedding], dim=0))
        self.lm_head.weight = self.word_embeddings.weight
        assert id(self.word_embeddings.weight) == id(self.lm_head.weight)
        self.word_embeddings.num_embeddings += 1


class ViTLPModel(ViTLPPreTrainedModel):
    def __init__(self, config):
        super().__init__(copy.deepcopy(config))
        self.encoder = VitlpEncoder(copy.deepcopy(config))
        self.decoder = VitlpDecoder(copy.deepcopy(config))

    def get_input_embeddings(self):
        return self.decoder.word_embeddings

    def set_input_embeddings(self, value):
        self.decoder.word_embeddings = value


class ViTLPForPreTraining(ViTLPPreTrainedModel):
    def __init__(self, config):
        super().__init__(copy.deepcopy(config))
        self.vitlp = ViTLPModel(copy.deepcopy(config))
        self.loss_fct = CrossEntropyLoss()
        self.post_init()
        loaded_parameter_num = 0
        loaded_parameters = set()
        if config.load_bart:
            self.vitlp, loaded_bart_parameters = load_pretrained_bart(self.vitlp, config.pretrained_bart_path, config.decoder_layers)
            loaded_parameter_num += len(loaded_bart_parameters)
            loaded_parameters |= loaded_bart_parameters
            # Since decoder input embeddings and output embeddings are shared in ViTLP, we do not need to load lm_head
            # See _tie_or_clone_weights in transformers/modeling_utils.py
        if config.load_vit:
            self.vitlp, loaded_vit_parameters = load_pretrained_vit(self.vitlp, config.pretrained_vit_path, config.encoder_layers, config.image_height // config.patch_size, config.image_width // config.patch_size)
            loaded_parameter_num += len(loaded_vit_parameters)
            loaded_parameters |= loaded_vit_parameters
        if config.load_bart or config.load_vit:
            vitlp_parameter_num = len(list(p for p in self.vitlp.parameters()))
            print('Total parameter groups in ViTLP vs. loaded parameter groups in backbone models : %d vs. %d' % (vitlp_parameter_num, loaded_parameter_num))
            if vitlp_parameter_num > loaded_parameter_num:
                for name, p in self.vitlp.named_parameters():
                    if name not in loaded_parameters:
                        print(name + ' is not loaded')
        # assert (self.vitlp.decoder.lm_head.weight - self.vitlp.decoder.word_embeddings.weight).abs().max() < 1e-8, str((self.vitlp.decoder.lm_head.weight - self.vitlp.decoder.word_embeddings.weight).abs().max())
        self.bin_size = config.bin_size
        self.LOCATE_ID = 50265 # hard-code confirmation
        self.BBOX_TOKEN_TYPE = 2

    def get_input_embeddings(self):
        return self.vitlp.decoder.word_embeddings

    def get_output_embeddings(self):
        return self.vitlp.decoder.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.vitlp.decoder.lm_head = new_embeddings

    # image                : [batch_size, 3, image_height, image_width]
    # decoder_input_ids    : [batch_size, max_length]
    # decoder_input_bboxes : [batch_size, max_length, 4]
    # labels               : [batch_size, max_length]
    # bboxes               : [batch_size, max_length, 4]
    def forward(self, image, decoder_input_ids, decoder_input_bboxes, labels, bboxes):
        encoder_outputs = self.vitlp.encoder(image).last_hidden_state
        bbox_input_ids = bboxes[:, :, :3].to(torch.int32)
        bbox_input_ids[bbox_input_ids == -100] = self.bin_size
        lm_logits, bbox_logits = self.vitlp.decoder.lm_decoder(encoder_outputs, decoder_input_ids, decoder_input_bboxes, bbox_input_ids)

        lm_loss = self.loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
        locate_loss = self.loss_fct(bbox_logits.view(-1, self.bin_size), bboxes.view(-1))
        return lm_loss, locate_loss

    def validate(self, image, decoder_input_ids, decoder_input_bboxes, labels, bboxes, token_types):
        bbox_condition = token_types.view(-1) == self.BBOX_TOKEN_TYPE
        label_bbox = (torch.masked_select(bboxes.view(-1, 4), bbox_condition.unsqueeze(dim=1))).view(-1, 4).float()
        encoder_outputs = self.vitlp.encoder(image).last_hidden_state
        bbox_input_ids = bboxes[:, :, :3]
        bbox_input_ids[bbox_input_ids == -100] = self.bin_size
        lm_logits, bbox_logits = self.vitlp.decoder.lm_decoder(encoder_outputs, decoder_input_ids, decoder_input_bboxes, bbox_input_ids)

        # 1. PPL
        lm_logits = lm_logits.view(-1, self.config.vocab_size)
        ppl = self.loss_fct(lm_logits, labels.view(-1)).item()
        # 2. locate slot accuracy
        box_num = bbox_condition.float().sum()
        locate_slot_num = (bbox_condition & (torch.argmax(lm_logits, dim=1) == self.LOCATE_ID)).float().sum()
        locate_slot_acc = (locate_slot_num / box_num).item()
        # 3. IOU
        bbox_prediction = torch.argmax(bbox_logits, dim=2)
        prediction_bbox = (torch.masked_select(bbox_prediction, bbox_condition.unsqueeze(dim=1))).view(-1, 4).float()
        iou = calculate_IOU(prediction_bbox, label_bbox).item()
        return ppl, locate_slot_acc, iou


class ViTLPForTokenClassification(ViTLPPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.vitlp = ViTLPModel(config)
        self.hidden_size = config.hidden_size
        self.num_labels = config.num_labels
        if config.fc_type == 0:
            self.fc = nn.Linear(config.hidden_size * 2, config.num_labels)
        elif config.fc_type == 1:
            self.fc = nn.Sequential(nn.Linear(config.hidden_size * 2, config.hidden_size), nn.Tanh(), nn.Linear(config.hidden_size, config.num_labels))
        else:
            raise Exception('FC classification head error : ' + str(config.fc_type))
        self.post_init()
        delattr(self.vitlp.decoder, 'bbox_output_embeddings')
        delattr(self.vitlp.decoder, 'bbox_decoder_start_embedding')
        delattr(self.vitlp.decoder, 'bbox_decoder')
        delattr(self.vitlp.decoder, 'bbox_head')
        delattr(self.vitlp.decoder.lm_decoder, 'bbox_output_embeddings')
        delattr(self.vitlp.decoder.lm_decoder, 'bbox_decoder_start_embedding')
        delattr(self.vitlp.decoder.lm_decoder, 'bbox_decoder')
        delattr(self.vitlp.decoder.lm_decoder, 'bbox_head')

    def get_input_embeddings(self):
        return self.vitlp.decoder.word_embeddings

    def get_output_embeddings(self):
        return self.vitlp.decoder.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.vitlp.decoder.lm_head = new_embeddings

    # image                    : [batch_size, 3, image_height, image_width]
    # decoder_input_ids        : [batch_size, max_length]
    # decoder_input_bboxes     : [batch_size, max_length, 4]
    # decoder_input_bbox_masks : [batch_size, max_length]
    # decoder_input_word_masks : [batch_size, max_length]
    def forward(self, image, decoder_input_ids, decoder_input_bboxes, decoder_input_bbox_masks, decoder_input_word_masks):
        encoder_outputs = self.vitlp.encoder(image).last_hidden_state
        hidden_states = self.vitlp.decoder.lm_decoder.forward_(encoder_outputs, decoder_input_ids, decoder_input_bboxes)
        bbox_hidden_states = torch.masked_select(hidden_states, decoder_input_bbox_masks.unsqueeze(dim=2)).view([-1, self.hidden_size])
        word_hidden_states = torch.masked_select(hidden_states, decoder_input_word_masks.unsqueeze(dim=2)).view([-1, self.hidden_size])
        logits = self.fc(torch.cat([bbox_hidden_states, word_hidden_states], dim=1))
        return logits


class ViTLPForDocumentClassification(ViTLPPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.vitlp = ViTLPModel(config)
        self.num_labels = config.num_labels
        self.dropout = config.dropout
        self.image_prompt_embedding = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.fc = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()
        delattr(self.vitlp.decoder, 'word_embeddings')
        delattr(self.vitlp.decoder, 'embed_positions')
        delattr(self.vitlp.decoder, 'layernorm_embedding')
        delattr(self.vitlp.decoder, 'lm_head')
        delattr(self.vitlp.decoder, 'bbox_input_embeddings_x')
        delattr(self.vitlp.decoder, 'bbox_input_embeddings_y')
        delattr(self.vitlp.decoder, 'bbox_output_embeddings')
        delattr(self.vitlp.decoder, 'bbox_decoder_start_embedding')
        delattr(self.vitlp.decoder, 'bbox_decoder')
        delattr(self.vitlp.decoder, 'bbox_head')
        delattr(self.vitlp.decoder.lm_decoder, 'word_embeddings')
        delattr(self.vitlp.decoder.lm_decoder, 'embed_positions')
        delattr(self.vitlp.decoder.lm_decoder, 'layernorm_embedding')
        delattr(self.vitlp.decoder.lm_decoder, 'lm_head')
        delattr(self.vitlp.decoder.lm_decoder, 'bbox_input_embeddings_x')
        delattr(self.vitlp.decoder.lm_decoder, 'bbox_input_embeddings_y')
        delattr(self.vitlp.decoder.lm_decoder, 'bbox_output_embeddings')
        delattr(self.vitlp.decoder.lm_decoder, 'bbox_decoder_start_embedding')
        delattr(self.vitlp.decoder.lm_decoder, 'bbox_decoder')
        delattr(self.vitlp.decoder.lm_decoder, 'bbox_head')

    # image : [batch_size, 3, image_height, image_width]
    def forward(self, image):
        encoder_outputs = self.vitlp.encoder(image).last_hidden_state
        input_prompt_embeddings = nn.functional.dropout(self.image_prompt_embedding.expand(image.size(0), -1, -1), p=self.dropout, training=self.training)
        hidden_states = self.vitlp.decoder.lm_decoder.forward_one_step(encoder_outputs, input_prompt_embeddings)
        logits = self.fc(hidden_states)
        return logits


class ViTLPForDocVQA(ViTLPPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.vitlp = ViTLPModel(config)
        self.loss_fct = CrossEntropyLoss()
        self.bin_size = config.bin_size
        self.LOCATE_ID = 50265 # hard-code confirmation

    def get_input_embeddings(self):
        return self.vitlp.decoder.word_embeddings

    def get_output_embeddings(self):
        return self.vitlp.decoder.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.vitlp.decoder.lm_head = new_embeddings

    # image                : [batch_size, 3, image_height, image_width]
    # decoder_input_ids    : [batch_size, max_length]
    # decoder_input_bboxes : [batch_size, max_length, 4]
    # labels               : [batch_size, max_length]
    # bboxes               : [batch_size, max_length, 4]
    def forward(self, image, decoder_input_ids, decoder_input_bboxes, labels, bboxes):
        encoder_outputs = self.vitlp.encoder(image).last_hidden_state
        bbox_input_ids = bboxes[:, :, :3].to(torch.int32)
        bbox_input_ids[bbox_input_ids == -100] = self.bin_size
        lm_logits, bbox_logits = self.vitlp.decoder.lm_decoder(encoder_outputs, decoder_input_ids, decoder_input_bboxes, bbox_input_ids)

        lm_loss = self.loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
        locate_loss = self.loss_fct(bbox_logits.view(-1, self.bin_size), bboxes.view(-1))
        return lm_loss, locate_loss

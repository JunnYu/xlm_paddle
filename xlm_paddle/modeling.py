# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2019-present, Facebook, Inc and the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddlenlp.transformers import PretrainedModel, register_base_model
from paddlenlp.transformers.albert.modeling import ACT2FN
from xlm_paddle.adaptive import AdaptiveLogSoftmaxWithLoss

__all__ = [
    "XLMModel",
    "XLMWithLMHeadModel",
    "XLMForSequenceClassification",
]

INF = 1e4


class SinusoidalPositionalEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__(num_embeddings, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out):
        n_pos, dim = out.shape
        out.stop_gradient = True
        position_ids = paddle.arange(0, n_pos, dtype=out.dtype).unsqueeze(1)
        indices = paddle.arange(0, dim // 2, dtype=out.dtype).unsqueeze(0)
        indices = 10000.0**(-2 * indices / dim)
        embeddings = paddle.matmul(position_ids, indices)
        out[:, 0::2] = paddle.sin(embeddings)
        out[:, 1::2] = paddle.cos(embeddings)
        return out

    @paddle.no_grad()
    def forward(self, position_ids):
        return super().forward(position_ids)


def get_masks(seqlen, lengths, causal, padding_mask=None):
    """
    Generate hidden states mask, and optionally an attention mask.
    """
    alen = paddle.arange(0, seqlen, dtype="int64")
    if padding_mask is not None:
        mask = padding_mask
    else:
        # assert lengths.max().item() <= seqlen
        mask = alen < lengths[:, None]

    # attention mask is the same as mask, or triangular inferior attention (causal)
    bs = paddle.shape(lengths)[0]
    if causal:
        attn_mask = (paddle.tile(alen[None, None, :],
                                 (bs, seqlen, 1)) <= alen[None, :, None])
    else:
        attn_mask = mask

    # sanity check
    # assert mask.shape == [bs, seqlen]
    # assert causal is False or attn_mask.shape == [bs, seqlen, seqlen]

    return mask, attn_mask


class MultiHeadAttention(nn.Layer):

    NEW_ID = itertools.count()

    def __init__(self, n_heads, dim, attention_probs_dropout_prob):
        super().__init__()
        self.layer_id = next(MultiHeadAttention.NEW_ID)
        self.dim = dim
        self.n_heads = n_heads
        assert self.dim % self.n_heads == 0
        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)
        self.out_lin = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(attention_probs_dropout_prob)
        self.dim_per_head = self.dim // self.n_heads

    def shape(self, x):
        """projection"""
        return x.reshape([0, 0, self.n_heads, self.dim_per_head]).transpose(
            [0, 2, 1, 3])

    def unshape(self, x):
        """compute context"""
        return x.transpose([0, 2, 1, 3]).reshape(
            [0, 0, self.n_heads * self.dim_per_head])

    def forward(self,
                input,
                mask,
                kv=None,
                cache=None,
                output_attentions=False):
        """
        Self-attention (if kv is None) or attention over source sentence (provided by kv).
        """
        # Input is (bs, qlen, dim)
        # Mask is (bs, klen) (non-causal) or (bs, klen, klen)
        bs, qlen, dim = paddle.shape(input)
        if kv is None:
            klen = qlen if cache is None else cache["seqlen"] + qlen
        else:
            klen = paddle.shape(kv)[1]

        mask_reshape = (bs, 1, qlen, klen) if mask.ndim == 3 else (bs, 1, 1,
                                                                   klen)

        q = self.shape(self.q_lin(input))  # (bs, n_heads, qlen, dim_per_head)
        if kv is None:
            k = self.shape(
                self.k_lin(input))  # (bs, n_heads, qlen, dim_per_head)
            v = self.shape(
                self.v_lin(input))  # (bs, n_heads, qlen, dim_per_head)
        elif cache is None or self.layer_id not in cache:
            k = v = kv
            k = self.shape(self.k_lin(k))  # (bs, n_heads, qlen, dim_per_head)
            v = self.shape(self.v_lin(v))  # (bs, n_heads, qlen, dim_per_head)

        if cache is not None:
            if self.layer_id in cache:
                if kv is None:
                    k_, v_ = cache[self.layer_id]
                    k = paddle.concat(
                        [k_, k], axis=2)  # (bs, n_heads, klen, dim_per_head)
                    v = paddle.concat(
                        [v_, v], axis=2)  # (bs, n_heads, klen, dim_per_head)
                else:
                    k, v = cache[self.layer_id]
            cache[self.layer_id] = (k, v)

        q = q / math.sqrt(
            self.dim_per_head)  # (bs, n_heads, qlen, dim_per_head)

        scores = paddle.matmul(
            q, k, transpose_y=True)  # (bs, n_heads, qlen, klen)

        mask = mask.reshape(mask_reshape)  # (bs, n_heads, qlen, klen)

        scores = scores + (mask.astype(scores.dtype) - 1) * INF

        weights = F.softmax(scores, axis=-1)  # (bs, n_heads, qlen, klen)
        weights = self.dropout(weights)  # (bs, n_heads, qlen, klen)

        context = paddle.matmul(weights,
                                v)  # (bs, n_heads, qlen, dim_per_head)
        context = self.unshape(context)  # (bs, qlen, dim)

        outputs = (self.out_lin(context), )
        if output_attentions:
            outputs = outputs + (weights, )
        return outputs


class TransformerFFN(nn.Layer):
    def __init__(self, in_dim, dim_hidden, out_dim, hidden_act, dropout_prob):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, dim_hidden)
        self.lin2 = nn.Linear(dim_hidden, out_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.act = ACT2FN[hidden_act]

    def forward(self, x):
        x = self.lin1(x)
        x = self.act(x)
        x = self.lin2(x)
        x = self.dropout(x)
        return x


class XLMPretrainedModel(PretrainedModel):
    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "xlm-mlm-tlm-xnli15-1024": {
            "is_encoder": True,
            "causal": False,
            "n_langs": 15,
            "use_lang_embeddings": True,
            "vocab_size": 95000,
            "eos_token_id": 1,
            "pad_token_id": 2,
            "hidden_size": 1024,
            "num_attention_heads": 8,
            "num_hidden_layers": 12,
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "use_sinusoidal_embeddings": False,
            "layer_norm_eps": 1e-12,
            "hidden_act": "gelu",
            "embed_init_std": 2048**-0.5,
            "init_std": 0.02,
            "use_asm": False,
            "asm_cutoffs": 1,
            "asm_div_value": 1,
            "mask_token_id": 0,
            "lang_id": 4,
            "lang2id": {
                "ar": 0,
                "bg": 1,
                "de": 2,
                "el": 3,
                "en": 4,
                "es": 5,
                "fr": 6,
                "hi": 7,
                "ru": 8,
                "sw": 9,
                "th": 10,
                "tr": 11,
                "ur": 12,
                "vi": 13,
                "zh": 14,
            }
        },
        "xlm-mlm-tlm-xnli15-1024-fintuned-on-xnli": {
            "is_encoder": True,
            "causal": False,
            "n_langs": 15,
            "use_lang_embeddings": True,
            "vocab_size": 95000,
            "eos_token_id": 1,
            "pad_token_id": 2,
            "hidden_size": 1024,
            "num_attention_heads": 8,
            "num_hidden_layers": 12,
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "use_sinusoidal_embeddings": False,
            "layer_norm_eps": 1e-12,
            "hidden_act": "gelu",
            "embed_init_std": 2048**-0.5,
            "init_std": 0.02,
            "use_asm": False,
            "asm_cutoffs": 1,
            "asm_div_value": 1,
            "mask_token_id": 0,
            "lang_id": 4,
            "lang2id": {
                "ar": 0,
                "bg": 1,
                "de": 2,
                "el": 3,
                "en": 4,
                "es": 5,
                "fr": 6,
                "hi": 7,
                "ru": 8,
                "sw": 9,
                "th": 10,
                "tr": 11,
                "ur": 12,
                "vi": 13,
                "zh": 14,
            }
        },
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "xlm-mlm-tlm-xnli15-1024":
            "https://huggingface.co/junnyu/xlm-mlm-tlm-xnli15-1024-paddle/resolve/main/model_state.pdparams",
            "xlm-mlm-tlm-xnli15-1024-fintuned-on-xnli":
            "https://huggingface.co/junnyu/xlm-mlm-tlm-xnli15-1024-paddle-fintuned-on-xnli/resolve/main/model_state.pdparams",
        }
    }
    base_model_prefix = "xlm"

    def init_weights(self):
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, layer):
        """Initialization hook"""
        if isinstance(layer, nn.Embedding):
            new_weight = paddle.normal(
                mean=0.0,
                std=self.embed_init_std if hasattr(self, "embed_init_std") else
                self.xlm.config["embed_init_std"],
                shape=layer.weight.shape, )
            if layer._padding_idx is not None:
                new_weight[layer._padding_idx] = paddle.zeros_like(new_weight[
                    layer._padding_idx])
            layer.weight.set_value(new_weight)
        elif isinstance(layer, nn.Linear):
            layer.weight.set_value(
                paddle.normal(
                    mean=0.0,
                    std=self.init_std if hasattr(self, "init_std") else
                    self.xlm.config["init_std"],
                    shape=layer.weight.shape, ))
            if layer.bias is not None:
                layer.bias.set_value(paddle.zeros_like(layer.bias))
        elif isinstance(layer, nn.LayerNorm):
            layer.bias.set_value(paddle.zeros_like(layer.bias))
            layer.weight.set_value(paddle.full_like(layer.weight, 1.0))


@register_base_model
class XLMModel(XLMPretrainedModel):
    def __init__(
            self,
            is_encoder=True,
            causal=False,
            n_langs=15,
            use_lang_embeddings=True,
            vocab_size=95000,
            hidden_size=1024,
            num_attention_heads=8,
            num_hidden_layers=12,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            use_sinusoidal_embeddings=False,
            layer_norm_eps=1e-12,
            hidden_act="gelu",
            embed_init_std=2048**-0.5,
            init_std=0.02,
            eos_token_id=1,
            pad_token_id=2,
            mask_token_id=0,
            lang_id=4,
            use_asm=False,
            lang2id={
                "ar": 0,
                "bg": 1,
                "de": 2,
                "el": 3,
                "en": 4,
                "es": 5,
                "fr": 6,
                "hi": 7,
                "ru": 8,
                "sw": 9,
                "th": 10,
                "tr": 11,
                "ur": 12,
                "vi": 13,
                "zh": 14,
            },
            asm_cutoffs=[8000, 20000],  # Adaptive softmax cutoffs
            asm_div_value=4,  # Adaptive softmax cluster sizes ratio
    ):
        super().__init__()
        self.causal = causal
        self.num_hidden_layers = num_hidden_layers
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.hidden_size = hidden_size
        self.embed_init_std = embed_init_std
        self.init_std = init_std
        self.use_lang_embeddings = use_lang_embeddings
        self.n_langs = n_langs
        if not is_encoder:
            raise NotImplementedError(
                "Currently XLM can only be used as an encoder")
        assert (
            hidden_size % num_attention_heads == 0
        ), "xlm model's hidden_size must be a multiple of num_attention_heads"

        # embeddings
        if use_sinusoidal_embeddings:
            self.position_embeddings = SinusoidalPositionalEmbedding(
                max_position_embeddings, hidden_size)
        else:
            self.position_embeddings = nn.Embedding(max_position_embeddings,
                                                    hidden_size)
        if n_langs > 1 and use_lang_embeddings:
            self.lang_embeddings = nn.Embedding(n_langs, hidden_size)
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.layer_norm_emb = nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)

        self.attentions = nn.LayerList()
        self.layer_norm1 = nn.LayerList()
        self.ffns = nn.LayerList()
        self.layer_norm2 = nn.LayerList()
        self.dropout = nn.Dropout(hidden_dropout_prob)

        for _ in range(self.num_hidden_layers):
            self.attentions.append(
                MultiHeadAttention(num_attention_heads, hidden_size,
                                   attention_probs_dropout_prob))
            self.layer_norm1.append(
                nn.LayerNorm(
                    hidden_size, epsilon=layer_norm_eps))

            self.ffns.append(
                TransformerFFN(
                    hidden_size,
                    hidden_size * 4,
                    hidden_size,
                    hidden_act,
                    hidden_dropout_prob, ))
            self.layer_norm2.append(
                nn.LayerNorm(
                    hidden_size, epsilon=layer_norm_eps))

        self.register_buffer(
            "position_ids",
            paddle.arange(0, max_position_embeddings).reshape((1, -1)),
            persistable=False, )
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            langs=None,
            position_ids=None,
            lengths=None,
            cache=None,
            output_attentions=False,
            output_hidden_states=False, ):
        bs, seqlen = paddle.shape(input_ids)

        if lengths is None:
            if input_ids is not None:
                lengths = (input_ids != self.pad_token_id).sum(
                    axis=1).astype("int64")
            else:
                lengths = paddle.to_tensor([seqlen] * bs, dtype="int64")

        # check inputs
        # assert lengths.shape[0] == bs
        # assert lengths.max().item() <= seqlen
        # generate masks
        mask, attn_mask = get_masks(
            seqlen, lengths, self.causal, padding_mask=attention_mask)

        # position_ids
        if position_ids is None:
            position_ids = self.position_ids[:, :seqlen]
        # assert position_ids.shape == [bs, seqlen]  # (seqlen, bs)

        # langs
        # if langs is not None:
        # assert langs.shape == [bs, seqlen]  # (seqlen, bs)

        # do not recompute cached elements
        if cache is not None and input_ids is not None:
            _seqlen = seqlen - cache["seqlen"]
            input_ids = input_ids[:, -_seqlen:]
            position_ids = position_ids[:, -_seqlen:]
            if langs is not None:
                langs = langs[:, -_seqlen:]
            mask = mask[:, -_seqlen:]
            attn_mask = attn_mask[:, -_seqlen:]

        # embeddings
        tensor = self.embeddings(input_ids) + self.position_embeddings(
            position_ids)
        if langs is not None and self.use_lang_embeddings and self.n_langs > 1:
            tensor = tensor + self.lang_embeddings(langs)

        tensor = self.layer_norm_emb(tensor)
        tensor = self.dropout(tensor)
        tensor = tensor * mask.unsqueeze(-1).astype(tensor.dtype)

        # transformer layers
        hidden_states = () if output_hidden_states else None
        attentions = () if output_attentions else None
        for i in range(self.num_hidden_layers):
            if output_hidden_states:
                hidden_states = hidden_states + (tensor, )
            # self attention
            attn_outputs = self.attentions[i](
                tensor,
                attn_mask,
                cache=cache,
                output_attentions=output_attentions, )
            attn = attn_outputs[0]
            if output_attentions:
                attentions = attentions + (attn_outputs[1], )
            attn = self.dropout(attn)
            tensor = tensor + attn
            tensor = self.layer_norm1[i](tensor)
            # FFN
            tensor = tensor + self.ffns[i](tensor)
            tensor = self.layer_norm2[i](tensor)
            tensor = tensor * mask.unsqueeze(-1).astype(tensor.dtype)

        # Add last hidden state
        if output_hidden_states:
            hidden_states = hidden_states + (tensor, )

        # update cache length
        if cache is not None:
            cache["seqlen"] += paddle.shape(tensor)[1]

        return tuple(v for v in [tensor, hidden_states, attentions]
                     if v is not None)


class XLMPredLayer(nn.Layer):
    """
    Prediction layer (cross_entropy or adaptive_softmax).
    """

    def __init__(
            self,
            use_asm,
            asm_cutoffs,
            asm_div_value,
            vocab_size,
            hidden_size,
            embedding_weights=None, ):
        super().__init__()
        self.use_asm = use_asm
        self.vocab_size = vocab_size
        if use_asm:
            self.proj = AdaptiveLogSoftmaxWithLoss(
                in_features=hidden_size,
                n_classes=vocab_size,
                cutoffs=asm_cutoffs,
                div_value=asm_div_value,
                head_bias=True,  # default is False
            )
        else:
            if embedding_weights is None:
                self.proj = nn.Linear(hidden_size, vocab_size)
            else:
                self.bias = self.create_parameter(
                    shape=[vocab_size], is_bias=True)
                self.proj = (
                    lambda x: paddle.matmul(x, embedding_weights, transpose_y=True)
                    + self.bias
                )

    def forward(self, x, y=None):
        """Compute the loss, and optionally the scores."""
        outputs = ()
        if self.use_asm:
            scores = self.proj.log_prob(x)
            outputs = (scores, ) + outputs
            if y is not None:
                _, loss = self.proj(x, y)
                outputs = (loss, ) + outputs
        else:
            scores = self.proj(x)
            outputs = (scores, ) + outputs
            if y is not None:
                loss = F.cross_entropy(
                    scores.reshape([-1, self.vocab_size]),
                    y.flatten(),
                    reduction="mean")
                outputs = (loss, ) + outputs
        return outputs


class XLMWithLMHeadModel(XLMPretrainedModel):
    def __init__(self, xlm):
        super().__init__()
        self.xlm = xlm
        self.pred_layer = XLMPredLayer(
            xlm.config["use_asm"],
            xlm.config["asm_cutoffs"],
            xlm.config["asm_div_value"],
            xlm.config["vocab_size"],
            xlm.config["hidden_size"],
            embedding_weights=self.xlm.embeddings.weight, )
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            langs=None,
            position_ids=None,
            lengths=None,
            cache=None,
            labels=None,
            output_attentions=False,
            output_hidden_states=False, ):
        xlm_outputs = self.xlm(
            input_ids,
            attention_mask=attention_mask,
            langs=langs,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states, )

        output = xlm_outputs[0]
        outputs = self.pred_layer(output, labels)
        return outputs + xlm_outputs[1:]

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        mask_token_id = self.xlm.config.mask_token_id
        lang_id = self.config.lang_id
        effective_batch_size = paddle.shape(input_ids)[0]
        mask_token = paddle.full(
            (effective_batch_size, 1), mask_token_id, dtype="int64")
        input_ids = paddle.concat([input_ids, mask_token], axis=1)
        if lang_id is not None:
            langs = paddle.full_like(input_ids, lang_id)
        else:
            langs = None
        return {"input_ids": input_ids, "langs": langs}


class XLMForSequenceClassification(XLMPretrainedModel):
    def __init__(self, xlm, num_classes=2, dropout=None):
        super().__init__()
        self.num_classes = num_classes
        self.xlm = xlm
        dropout_prob = (dropout if dropout is not None else
                        self.xlm.config["hidden_dropout_prob"])
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(self.xlm.config["hidden_size"],
                                    num_classes)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            langs=None,
            position_ids=None,
            lengths=None,
            output_attentions=False,
            output_hidden_states=False,
            labels=None, ):
        sequence_output = self.xlm(
            input_ids,
            attention_mask,
            langs,
            position_ids,
            lengths,
            None,
            output_attentions,
            output_hidden_states, )[0]
        logits = self.classifier(sequence_output[:, 0])

        if labels is not None:
            loss = F.cross_entropy(logits, labels.flatten())
            return (loss, logits)
        else:
            return logits

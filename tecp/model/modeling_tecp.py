import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
from torch.nn import CrossEntropyLoss
import torchvision
import numpy as np
import os
import sys
import math
from transformers import RobertaModel,BertModel,AlbertModel,ElectraModel,ViTModel,SwinModel,DeiTModel,ConvNextModel
from torchvision.models import resnet50

import copy
from typing import Optional, List
from torch import nn, Tensor
torch.cuda.empty_cache()


class ImageModel(nn.Module):
    def __init__(self):
        super(ImageModel, self).__init__()
        self.resnet = resnet50(pretrained=True)

    def forward(self, x, aux_imgs=None):
        # full image prompt
        prompt_guids = self.get_resnet_prompt(x)  # 4x[bsz, 256, 7, 7]

        # aux_imgs: bsz x 3(nums) x 3 x 224 x 224
        if aux_imgs is not None:
            aux_prompt_guids = []  # goal: 3 x (4 x [bsz, 256, 7, 7])
            aux_imgs = aux_imgs.permute([1, 0, 2, 3, 4])  # 3(nums) x bsz x 3 x 224 x 224
            for i in range(len(aux_imgs)):
                aux_prompt_guid = self.get_resnet_prompt(aux_imgs[i])  # 4 x [bsz, 256, 7, 7]
                aux_prompt_guids.append(aux_prompt_guid)
            return prompt_guids, aux_prompt_guids
        return prompt_guids, None

    def get_resnet_prompt(self, x):
        """generate image prompt

        Args:
            x ([torch.tenspr]): bsz x 3 x 224 x 224

        Returns:
            prompt_guids ([List[torch.tensor]]): 4 x List[bsz x 256 x 7 x 7]
        """
        # image: bsz x 3 x 224 x 224
        prompt_guids = []
        for name, layer in self.resnet.named_children():
            if name == 'fc' or name == 'avgpool':  continue
            x = layer(x)  # (bsz, 256, 56, 56)
            if 'layer' in name:
                bsz, channel, ft, _ = x.size()
                kernel = ft // 2
                prompt_kv = nn.AvgPool2d(kernel_size=(kernel, kernel), stride=kernel)(x)  # (bsz, 256, 7, 7)
                prompt_guids.append(prompt_kv)  # conv2: (bsz, 256, 7, 7)
        return prompt_guids

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense1 = nn.Linear(config.hidden_size, 3072)
        if isinstance("gelu", str) or (sys.version_info[0] == 2 and isinstance("gelu", unicode)):
            self.intermediate_act_fn = ACT2FN["gelu"]
        else:
            self.intermediate_act_fn = "gelu"
        self.dense2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states, attention_output):
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + attention_output)
        return hidden_states

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = MultiHeadAttention(8,d_model,d_model,d_model)
        self.multihead_attn = MultiHeadAttention(8,d_model,d_model,d_model)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(self.with_pos_embed(tgt, query_pos),
                                   self.with_pos_embed(memory, pos),
                                   memory
                                   )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class TECPModel(nn.Module):
    def __init__(self,config1,config2,text_num_labels,alpha,beta,use_prompt=True,text_model_name="roberta",image_model_name='vit'):
        super().__init__()
        if text_model_name == 'roberta':
            self.roberta = RobertaModel(config1,add_pooling_layer=False)
        if image_model_name == 'vit':
            self.vit = ViTModel(config2)
        self.alpha = alpha
        self.beta = beta
        self.text_model_name=text_model_name
        self.image_model_name=image_model_name
        self.config1 = config1
        self.config2 = config2
        self.text_num_labels = text_num_labels
        self.image_text_cross = MultiHeadAttention(8,config1.hidden_size,config1.hidden_size,config1.hidden_size)
        self.dropout = nn.Dropout(config1.hidden_dropout_prob)
        self.loss_fct = CrossEntropyLoss()
        self.classifier1 = nn.Linear(config1.hidden_size, self.text_num_labels)
        self.classifier0= nn.Linear(config1.hidden_size,self.text_num_labels)
        self.CRF = CRF(self.text_num_labels,batch_first=True)

        self.intermediate = BertIntermediate(config1)

        decoder_layer = TransformerDecoderLayer(config1.hidden_size, config1.num_attention_heads, 2048,
                                                config1.hidden_dropout_prob, config1.hidden_act, False)
        # decoder_norm = nn.LayerNorm(config1.hidden_size)
        self.cap_decoder = decoder_layer

        self.cap_loss = torch.nn.CrossEntropyLoss()
        self.mlp = MLP(config1.hidden_size, config1.hidden_size, config1.vocab_size, 3)


    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                pixel_values=None,
                anp_labels=None,
                region_feats=None,
                cap_input_ids=None,
                cap_attention_mask=None,
                aux_pixel_values=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                image_labels=None,
                head_mask=None,
                cross_labels=None,
                return_dict=None):

        return_dict = return_dict if return_dict is not None else self.config1.use_return_dict


        text_outputs = self.roberta(input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids,
                                    position_ids=position_ids,
                                    head_mask=head_mask,
                                    inputs_embeds=inputs_embeds,
                                    output_attentions=output_attentions,
                                    output_hidden_states=output_hidden_states,
                                    return_dict=return_dict)
        cap_outputs = self.roberta(cap_input_ids,
                                    attention_mask=cap_attention_mask,
                                    token_type_ids=token_type_ids,
                                    position_ids=position_ids,
                                    head_mask=head_mask,
                                    inputs_embeds=inputs_embeds,
                                    output_attentions=output_attentions,
                                    output_hidden_states=output_hidden_states,
                                    return_dict=return_dict)

        image_outputs = self.vit(pixel_values,head_mask=head_mask)

        text_last_hidden_states = text_outputs["last_hidden_state"]
        image_last_hidden_states = image_outputs["last_hidden_state"]

        target_input = cap_outputs["last_hidden_state"]
        cap_out = self.cap_decoder(target_input,image_last_hidden_states)
        cap_logist = self.mlp(cap_out.permute(1, 0, 2))

        cap_loss = self.cap_loss(cap_logist.permute(1, 2, 0), cap_input_ids[:, :])

        # 初始 cross_crf_loss
        image_text_cross_attention, _ = self.image_text_cross(text_last_hidden_states, image_last_hidden_states,
                                                              image_last_hidden_states)


        image_text_cross_attention = self.intermediate(image_text_cross_attention, image_text_cross_attention)

        cross_logits = self.classifier0(image_text_cross_attention)# 仅是图文交互后的文本特征
        mask = (labels != -100)
        mask[:,0] = 1
        # print(cross_logits.shape, cross_labels.shape)

        cross_crf_loss =  -self.CRF(cross_logits,cross_labels,mask=mask) / 10


        # text_loss，文本在进行属性标注的输出向量
        sequence_output1 = self.dropout(text_last_hidden_states)
        text_token_logits = self.classifier1(sequence_output1)

        # getTextLoss: CrossEntropy
        text_loss = self.loss_fct(text_token_logits.view(-1, self.text_num_labels), labels.view(-1))


        loss =  cross_crf_loss + self.beta * cap_loss + self.alpha * text_loss #描述作为新增的子任务

        # end train
        return {"loss":loss,
            "logits":text_token_logits,
            "cross_logits": cross_logits,
                }





class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, dropout2=False, attn_type='softmax'):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        if dropout2:
            # self.dropout2 = nn.Dropout(dropout2)
            self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_type=attn_type,
                                                       dropout=dropout2)
        else:
            self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_type=attn_type,
                                                       dropout=dropout)

        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(d_model)

        if n_head > 1:
            self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
            nn.init.xavier_normal_(self.fc.weight)

    def forward(self, q, k, v, attn_mask=None, dec_self=False):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        if hasattr(self, 'dropout2'):
            q = self.dropout2(q)

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        if attn_mask is not None:
            attn_mask = attn_mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..

        output, attn = self.attention(q, k, v, attn_mask=attn_mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        if hasattr(self, 'fc'):
            output = self.fc(output)

        if hasattr(self, 'dropout'):
            output = self.dropout(output)

        if dec_self:
            output = self.layer_norm(output + residual)
        else:
            output = self.layer_norm(output + residual)

        return output, attn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, dropout=0.1, attn_type='softmax'):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
        if attn_type == 'softmax':
            self.attn_type = nn.Softmax(dim=2)
            # self.softmax = BottleSoftmax()
        else:
            self.attn_type = nn.Sigmoid()

    def forward(self, q, k, v, attn_mask=None, stop_sig=False):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if attn_mask is not None:
            # attn = attn.masked_fill(attn_mask, -np.inf)
            attn = attn.masked_fill(attn_mask, -1e6)

        if stop_sig:
            print('**')
            stop()

        attn = self.attn_type(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


def position_encoding_init(n_position, d_pos_vec):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_pos_vec) for j in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)


def distant_cross_entropy(logits, positions, mask=None):
    '''
    :param logits: [N, L]
    :param positions: [N, L]
    :param mask: [N]
    '''
    log_softmax = nn.LogSoftmax(dim=-1)
    log_probs = log_softmax(logits)
    if mask is not None:
        loss = -1 * torch.mean(torch.sum(positions.to(dtype=log_probs.dtype) * log_probs, dim=-1) /
                               (torch.sum(positions.to(dtype=log_probs.dtype), dim=-1) + mask.to(dtype=log_probs.dtype)))
    else:
        loss = -1 * torch.mean(torch.sum(positions.to(dtype=log_probs.dtype) * log_probs, dim=-1) /
                               torch.sum(positions.to(dtype=log_probs.dtype), dim=-1))
    return loss



def cost_matrix_cosine(x, y, eps=1e-5):
    """ Compute cosine distnace across every pairs of x, y (batched)
    [B, L_x, D] [B, L_y, D] -> [B, Lx, Ly]"""
    assert x.dim() == y.dim()
    assert x.size(0) == y.size(0)
    assert x.size(2) == y.size(2)
    x_norm = F.normalize(x, p=2, dim=-1, eps=eps)
    y_norm = F.normalize(y, p=2, dim=-1, eps=eps)
    cosine_sim = x_norm.matmul(y_norm.transpose(1, 2))
    cosine_dist = 1 - cosine_sim
    return cosine_dist


def trace(x):
    """ compute trace of input tensor (batched) """
    b, m, n = x.size()
    assert m == n
    mask = torch.eye(n, dtype=torch.bool, device=x.device
                     ).unsqueeze(0).expand_as(x)
    trace = x.masked_select(mask).contiguous().view(
        b, n).sum(dim=-1, keepdim=False)
    return trace


@torch.no_grad()
def ipot(C, x_len, x_pad, y_len, y_pad, joint_pad, beta, iteration, k):
    """ [B, M, N], [B], [B, M], [B], [B, N], [B, M, N]"""
    b, m, n = C.size()
    sigma = torch.ones(b, m, dtype=C.dtype, device=C.device
                       ) / x_len.unsqueeze(1)
    T = torch.ones(b, n, m, dtype=C.dtype, device=C.device)
    A = torch.exp(-C.transpose(1, 2)/beta)

    # mask padded positions
    sigma.masked_fill_(x_pad, 0)
    joint_pad = joint_pad.transpose(1, 2)
    T.masked_fill_(joint_pad, 0)
    A.masked_fill_(joint_pad, 0)

    # broadcastable lengths
    x_len = x_len.unsqueeze(1).unsqueeze(2)
    y_len = y_len.unsqueeze(1).unsqueeze(2)

    # mask to zero out padding in delta and sigma
    x_mask = (x_pad.to(C.dtype) * 1e4).unsqueeze(1)
    y_mask = (y_pad.to(C.dtype) * 1e4).unsqueeze(1)

    for _ in range(iteration):
        Q = A * T  # bs * n * m
        sigma = sigma.view(b, m, 1)
        for _ in range(k):
            delta = 1 / (y_len * Q.matmul(sigma).view(b, 1, n) + y_mask)
            sigma = 1 / (x_len * delta.matmul(Q) + x_mask)
        T = delta.view(b, n, 1) * Q * sigma
    T.masked_fill_(joint_pad, 0)
    return T


def optimal_transport_dist(txt_emb, img_emb, txt_pad, img_pad,
                           beta=0.5, iteration=50, k=1):
    """ [B, M, D], [B, N, D], [B, M], [B, N]"""
    cost = cost_matrix_cosine(txt_emb, img_emb)
    # mask the padded inputs
    joint_pad = txt_pad.unsqueeze(-1) | img_pad.unsqueeze(-2)
    cost.masked_fill_(joint_pad, 0)

    txt_len = (txt_pad.size(1) - txt_pad.sum(dim=1, keepdim=False)
               ).to(dtype=cost.dtype)
    img_len = (img_pad.size(1) - img_pad.sum(dim=1, keepdim=False)
               ).to(dtype=cost.dtype)

    T = ipot(cost.detach(), txt_len, txt_pad, img_len, img_pad, joint_pad,
             beta, iteration, k)
    distance = trace(cost.matmul(T.detach()))
    return distance






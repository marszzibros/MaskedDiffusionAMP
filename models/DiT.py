import math
import typing

import flash_attn
import flash_attn.layers.rotary
from flash_attn.bert_padding import pad_input, unpad_input
import huggingface_hub

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

def bias_dropout_add_scale(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
    training: bool) -> torch.Tensor:
  if bias is not None:
    out = scale * F.dropout(x + bias, p=prob, training=training)
  else:
    out = scale * F.dropout(x, p=prob, training=training)
  if residual is not None:
    out = residual + out
  return out

def get_bias_dropout_add_scale(training):
  def _bias_dropout_add(x, bias, scale, residual, prob):
    return bias_dropout_add_scale(
      x, bias, scale, residual, prob, training)

  return _bias_dropout_add

# function overload
def modulate(x: torch.Tensor,
             shift: torch.Tensor,
             scale: torch.Tensor) -> torch.Tensor:
  return x * (1 + scale) + shift

@torch.jit.script
def bias_dropout_add_scale_fused_train(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
  return bias_dropout_add_scale(
    x, bias, scale, residual, prob, True)

@torch.jit.script
def bias_dropout_add_scale_fused_inference(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
  return bias_dropout_add_scale(
    x, bias, scale, residual, prob, False)

@torch.jit.script
def modulate_fused(x: torch.Tensor,
                   shift: torch.Tensor,
                   scale: torch.Tensor) -> torch.Tensor:
  return modulate(x, shift, scale)

class Rotary(torch.nn.Module):
  def __init__(self, dim, base=10_000):
    super().__init__()
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    self.register_buffer('inv_freq', inv_freq)
    self.seq_len_cached = None
    self.cos_cached = None
    self.sin_cached = None

  def forward(self, x, seq_dim=1):
    seq_len = x.shape[seq_dim]
    if seq_len != self.seq_len_cached:
      self.seq_len_cached = seq_len
      t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
      freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
      emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
      # dims are: batch, seq_len, qkv, head, dim
      self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1,1,3,1,1)
      self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1,1,3,1,1)
      # This makes the transformation on v an identity.
      self.cos_cached[:,:,2,:,:].fill_(1.)
      self.sin_cached[:,:,2,:,:].fill_(0.)

    return self.cos_cached, self.sin_cached

def rotate_half(x):
  x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
  return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(qkv, cos, sin):
  cos = cos[0,:,0,0,:cos.shape[-1]//2]
  sin = sin[0,:,0,0,:sin.shape[-1]//2]
  return flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv, cos, sin)

# function overload
def modulate(x, shift, scale):
  return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNorm(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.weight = nn.Parameter(torch.ones([dim]))
    self.dim = dim
  def forward(self, x):
    with torch.amp.autocast('cuda', enabled=False):
      x = F.layer_norm(x.float(), [self.dim])
    return x * self.weight[None,None,:].to(x.device)

def residual_linear(x, W, x_skip, residual_scale):
  """x_skip + residual_scale * W @ x"""
  dim_out, dim_in = W.shape[0], W.shape[1]
  return torch.addmm(
    x_skip.view(-1, dim_out),
    x.view(-1, dim_in),
    W.T,
    alpha=residual_scale).view(*x.shape[:-1], dim_out)

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
class TimestepEmbedder(nn.Module):
  """
  Embeds scalar timesteps into vector representations.
  """
  def __init__(self, hidden_size, frequency_embedding_size=256):
    super().__init__()
    self.mlp = nn.Sequential(
      nn.Linear(frequency_embedding_size, hidden_size, bias=True),
      nn.SiLU(),
      nn.Linear(hidden_size, hidden_size, bias=True))
    self.frequency_embedding_size = frequency_embedding_size

  @staticmethod
  def timestep_embedding(t, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    half = dim // 2
    freqs = torch.exp(
      - math.log(max_period)
      * torch.arange(start=0, end=half, dtype=torch.float32)
      / half).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
      embedding = torch.cat(
        [embedding,
         torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

  def forward(self, t):
    t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
    t_emb = self.mlp(t_freq)
    return t_emb

class LabelEmbedder(nn.Module):
  """Embeds class labels into vector representations.
  
  Also handles label dropout for classifier-free guidance.
  """
  def __init__(self, num_classes, cond_size):
    super().__init__()
    self.embedding_table = nn.Embedding(num_classes + 1, cond_size)
    self.num_classes = num_classes

    # TODO think of initializing with 0.02 std deviation like in original DiT paper

  def forward(self, labels):
    embeddings = self.embedding_table(labels)
    return embeddings
    
#################################################################################
#                                 Core Model                                    #
#################################################################################

class Attention(nn.Module):
  def __init__(self, dim, n_heads, dropout, mlp_ratio=4,):
    super().__init__()
    self.n_heads = n_heads

    self.norm = LayerNorm(dim)

    self.attn_q = nn.Linear(dim, dim, bias=False)
    self.attn_k = nn.Linear(dim, dim, bias=False)
    self.attn_v = nn.Linear(dim, dim, bias=False)

    self.attn_out = nn.Linear(dim, dim, bias=False)

    self.norm2 = LayerNorm(dim)
    self.mlp = nn.Sequential(
      nn.Linear(dim, mlp_ratio * dim, bias=True),
      nn.GELU(approximate='tanh'),
      nn.Linear(mlp_ratio * dim, dim, bias=True))

    self.dropout = dropout

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return bias_dropout_add_scale_fused_inference

  def forward(self, x, rotary_cos_sin, seqlens, adaLN_parm):
    x_skip = x
    bias_dropout_scale_fn = self._get_bias_dropout_scale()

    x = modulate_fused(self.norm(x), adaLN_parm[0], adaLN_parm[1])

    # Compute Q, K, V and reshape
    qkv = torch.cat([self.attn_q(x), self.attn_k(x), self.attn_v(x)], dim=-1)
    qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)

    B, L, C, H, D = qkv.shape

    # apply rotary pos embedding
    cos, sin = rotary_cos_sin
    
    qkv = apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))  
    qkv = rearrange(qkv, 'b s c h d -> b s (c h d)')
    
    # create masks for attending only "valid" tokens 
    valid_mask = torch.arange(L, device=qkv.device)[None, :] < seqlens[:, None]  
    valid_indices = valid_mask.flatten().nonzero(as_tuple=False).squeeze(-1)

    qkv = qkv.flatten(0,1)[valid_indices].view(-1, 3, H, D)

    cu_seqlens = torch.cat((
        torch.tensor([0], device=x.device, dtype=torch.int32),
        seqlens.to(dtype=torch.int32)
    )).cumsum(dim=0, dtype=torch.int32)

    x = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
        qkv, cu_seqlens, L, 0., causal=False)
    
    # apply masks
    out_padded = torch.zeros(B * L, H, D, device=x.device, dtype=x.dtype)
    out_padded[valid_indices] = x

    # rearrange it back
    x = rearrange(out_padded.view(B, L, H, D), 'b s h d -> b s (h d)')

    x = bias_dropout_scale_fn(self.attn_out(x),
                              None,
                              adaLN_parm[2],
                              x_skip,
                              self.dropout)
    # mlp operation
    x = bias_dropout_scale_fn(
      self.mlp(modulate_fused(
        self.norm2(x), adaLN_parm[3], adaLN_parm[4])),
      None, adaLN_parm[5], x, self.dropout)
    
    return x

class CrossAttention(nn.Module):
  def __init__(self, dim, n_heads, dropout, mlp_ratio=4,fusion="sigmoid"):
    super().__init__()
    self.n_heads = n_heads
    self.fusion = fusion

    self.norm = LayerNorm(dim)

    self.attn_q = nn.Linear(dim, dim, bias=False)
    self.attn_k = nn.Linear(dim, dim, bias=False)
    self.attn_v = nn.Linear(dim, dim, bias=False)

    self.attn_out = nn.Linear(dim, dim, bias=False)

    self.norm2 = LayerNorm(dim)
    self.mlp = nn.Sequential(
      nn.Linear(dim, mlp_ratio * dim, bias=True),
      nn.GELU(approximate='tanh'),
      nn.Linear(mlp_ratio * dim, dim, bias=True))

    self.dropout = dropout

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return bias_dropout_add_scale_fused_inference
    
  def forward(self, x, s, rotary_cos_sin, seqlens, adaLN_parm):
    x_skip = x
    bias_dropout_scale_fn = self._get_bias_dropout_scale()
    if self.fusion != "without_norm_cross":
      x = self.norm(x)
    x = modulate_fused(x, adaLN_parm[0], adaLN_parm[1])

    # Compute Q, K, V and reshape
    qkv = torch.cat([self.attn_q(x), self.attn_k(s), self.attn_v(s)], dim=-1)
    qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)

    B, L, C, H, D = qkv.shape

    # apply rotary pos embedding
    cos, sin = rotary_cos_sin
    
    qkv = apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))  
    qkv = rearrange(qkv, 'b s c h d -> b s (c h d)')
    
    # create masks for attending only "valid" tokens 
    valid_mask = torch.arange(L, device=qkv.device)[None, :] < seqlens[:, None]  
    valid_indices = valid_mask.flatten().nonzero(as_tuple=False).squeeze(-1)

    qkv = qkv.flatten(0,1)[valid_indices].view(-1, 3, H, D)

    cu_seqlens = torch.cat((
        torch.tensor([0], device=x.device, dtype=torch.int32),
        seqlens.to(dtype=torch.int32)
    )).cumsum(dim=0, dtype=torch.int32)

    x = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
        qkv, cu_seqlens, L, 0., causal=False)
    
    # apply masks
    out_padded = torch.zeros(B * L, H, D, device=x.device, dtype=x.dtype)
    out_padded[valid_indices] = x

    # rearrange it back
    x = rearrange(out_padded.view(B, L, H, D), 'b s h d -> b s (h d)')

    x = bias_dropout_scale_fn(self.attn_out(x),
                              None,
                              adaLN_parm[2],
                              x_skip,
                              self.dropout)
    
    return x


class DDiTBlock(nn.Module):
  def __init__(self, dim, n_heads, cond_dim, dropout=0.1, fusion="sigmoid"):
    super().__init__()
    self.fusion = fusion
    self.attn_seqs = Attention(dim=dim, n_heads=n_heads, dropout=dropout, mlp_ratio=4)
    if self.fusion not in ["attention_seq_only", "without_structure"]:
      self.attn_strc = Attention(dim=dim, n_heads=n_heads, dropout=dropout, mlp_ratio=4)

    

    self.dropout = dropout
    

    if self.fusion in ["sigmoid"]:
      self.fuse_gate = nn.Sequential(
        nn.Linear(2 * dim, 1),  
        nn.Sigmoid()
      )
    else: 
      self.attn_cros = CrossAttention(dim=dim, n_heads=n_heads, dropout=dropout, mlp_ratio=4, fusion=self.fusion)
      self.adaLN_modulation_cross = nn.Linear(cond_dim, 3 * dim, bias=True)
      self.adaLN_modulation_cross.weight.data.zero_()
      self.adaLN_modulation_cross.bias.data.zero_()

    self.adaLN_modulation_seqs = nn.Linear(cond_dim, 6 * dim, bias=True)
    self.adaLN_modulation_strc = nn.Linear(cond_dim, 6 * dim, bias=True)

    self.adaLN_modulation_seqs.weight.data.zero_()
    self.adaLN_modulation_seqs.bias.data.zero_()

    self.adaLN_modulation_strc.weight.data.zero_()
    self.adaLN_modulation_strc.bias.data.zero_()


  def forward(self, x, s, rotary_cos_sin, c, seqlens):

    # (shift_msa, scale_msa, gate_msa, 
    #  shift_mlp, scale_mlp, gate_mlp) 
    adaLN_parm_seqs = self.adaLN_modulation_seqs(c)[:, None].chunk(6, dim=2)
    adaLN_parm_strc = self.adaLN_modulation_strc(c)[:, None].chunk(6, dim=2)
    BS, L, D = x.shape
    seqs = self.attn_seqs(x, rotary_cos_sin, seqlens, adaLN_parm_seqs)  
    if self.fusion not in ["without_structure"]:
      strc = self.attn_strc(s, rotary_cos_sin, seqlens, adaLN_parm_strc)

    if self.fusion in ["sigmoid"]:
      gate = self.fuse_gate(torch.cat([seqs, strc], dim=-1) )
      x = gate * seqs + (1 - gate) * strc
    elif self.fusion in ["crossattention", "without_norm_cross"]:
      adaLN_parm_cross = self.adaLN_modulation_cross(c)[:, None].chunk(3, dim=2)
      x = self.attn_cros(seqs, strc, rotary_cos_sin, seqlens, adaLN_parm_cross)
    elif self.fusion in ["without_structure"]:
      x = seqs
    return x

class EmbeddingLayer(nn.Module):
    def __init__(self, dim, tokens_dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((tokens_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        return self.embedding[x]

class StructureEmbeddingLayer(nn.Module):
    def __init__(self, dim=256, num_categoricals=25):
        super().__init__()
        self.embedding = nn.Embedding(num_categoricals, dim)

    def forward(self, x, seq_lens):
        emb = self.embedding(x)
        
        BS, L, _ = x.shape
        seq_lens = seq_lens

        range_row = torch.arange(L, device=x.device).unsqueeze(0).expand(BS, -1)  # shape: (BS, L)
        mask = range_row < seq_lens.unsqueeze(1) # BS x L

        row_mask = mask.unsqueeze(2).unsqueeze(-1)   # BS x L x 1 x 1
        col_mask = mask.unsqueeze(1).unsqueeze(-1)   # BS x 1 x L x 1

        full_mask = row_mask & col_mask              # BS x L x L x 1
        emb = emb * full_mask.float()

        valid_counts = full_mask.float().sum(dim=2).clamp(min=1e-6)  # (BS, L, 1)
        pooled = emb.sum(dim=2) / valid_counts

        return pooled

class DDitFinalLayer(nn.Module):
  def __init__(self, hidden_size, out_channels, cond_dim, seq_length, num_categoricals):
    super().__init__()
    
    self.hidden_size = hidden_size
    self.seq_length = seq_length
    self.num_categoricals = num_categoricals
    self.norm_final = LayerNorm(hidden_size)

    self.linear_seq = nn.Linear(hidden_size, out_channels)
    self.linear_seq.weight.data.zero_()
    self.linear_seq.bias.data.zero_()

    self.conv_strc = nn.Sequential(
        nn.Conv2d(1, 4, kernel_size=5, padding=2),
        nn.SiLU(),
        nn.Conv2d(4, 8, kernel_size=5, padding=2),
        nn.SiLU(),
        nn.Dropout2d(0.2),
        nn.Conv2d(8, self.num_categoricals, kernel_size=5, padding=2)
    )


    self.linear_strc = nn.Linear(self.num_categoricals, self.num_categoricals)
    self.linear_strc.weight.data.zero_()
    self.linear_strc.bias.data.zero_()

    self.adaLN_modulation = nn.Linear(cond_dim,
                                      2 * hidden_size,
                                      bias=True)
    self.adaLN_modulation.weight.data.zero_()
    self.adaLN_modulation.bias.data.zero_()


  def forward(self, x, c):
    shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
    x = modulate_fused(self.norm_final(x), shift, scale)
    seqs = self.linear_seq(x)

    dp = torch.einsum("bih,bjh->bij", x, x) # [BS, L, L]

    structure = self.conv_strc(dp.unsqueeze(1)).permute(0,2,3,1)  # [BS, 1, L, L]
    structure = self.linear_strc(structure)
    # [BS, 25, L, L]

    return seqs, structure

class DIT(nn.Module, huggingface_hub.PyTorchModelHubMixin):
  def __init__(self, 
               vocab_size,
               seq_length = 66,
               hidden_size = 768,
               cond_dim = 256,
               n_heads = 8,
               n_blocks = 12,
               dropout = 0.2,
               num_categoricals = 25,
               scale_by_sigma = True,
               fusion="sigmoid"):
    super().__init__()

    self.vocab_size = vocab_size
    self.seq_length = seq_length
    self.fusion = fusion
    self.num_categoricals = num_categoricals
    self.seqs_embed = EmbeddingLayer(hidden_size,
                                      vocab_size)
    if self.fusion not in ["without_structure"]:
      self.strc_embed = StructureEmbeddingLayer(hidden_size, num_categoricals)
    self.sigma_map = TimestepEmbedder(cond_dim)
    self.rotary_emb = Rotary(
      hidden_size // n_heads)
    self.condition = nn.Embedding(num_embeddings=4, embedding_dim=cond_dim)

    blocks = []
    for _ in range(n_blocks):
      blocks.append(DDiTBlock(hidden_size,
                              n_heads,
                              cond_dim,
                              dropout=dropout, fusion=fusion))
    self.blocks = nn.ModuleList(blocks)
    
    self.output_layer = DDitFinalLayer(
      hidden_size,
      vocab_size,
      cond_dim,
      seq_length,
      num_categoricals)
    self.scale_by_sigma = scale_by_sigma

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return  bias_dropout_add_scale_fused_inference

  def forward(self, x, sigma, seqlens, s):

    x = self.seqs_embed(x)
    if self.fusion not in ["without_structure"]:
      s = self.strc_embed(s, seqlens)
    c = F.silu(self.sigma_map(sigma))
    # c = self.condition.mean(dim=1)
    rotary_cos_sin = self.rotary_emb(x)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
      for i in range(len(self.blocks)):
        x = self.blocks[i](x, s, rotary_cos_sin, c, seqlens)
      x, s = self.output_layer(x, c)
    return x, s
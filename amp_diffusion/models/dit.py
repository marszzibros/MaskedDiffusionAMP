import math
import typing

# flash-attn is OPTIONAL, so that `import amp_diffusion` works in an
# environment without it. The error is re-raised at model construction if the
# fa2 backend is actually asked for.
#
# On Blackwell (sm_120 -- the RTX PRO 6000), note carefully that the LIBRARY is
# not the problem: flash-attn 2.x has emitted sm_120 kernels since v2.7.3. What
# matters is the CUDA toolkit the WHEEL was built with -- setup.py only emits
# sm_100/sm_120 gencode under toolkit >= 12.8 and silently skips it otherwise.
# The cu126 wheel this repo pinned therefore contains sm_80/sm_90 cubins and an
# empty PTX section, and dies with "no kernel image is available for execution
# on the device" at the first kernel launch. The cu130 build of the very same
# flash-attn version carries sm_120.
#
# This try/except also catches the mismatched-wheel case, which raises
# ImportError on an undefined symbol rather than failing at install (see README).
try:
  import flash_attn
  import flash_attn.layers.rotary
  _FLASH_ATTN_IMPORT_ERROR = None
except ImportError as _e:                                      # pragma: no cover
  flash_attn = None
  _FLASH_ATTN_IMPORT_ERROR = _e
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


def apply_rotary_pos_emb_torch(qkv, cos, sin):
  """Pure-torch equivalent of apply_rotary_pos_emb, for the sdpa backend.

  Matches the flash kernel exactly: full head_dim rotary in the half-split
  (GPT-NeoX, non-interleaved) convention, applied to Q and K only, V untouched.

  Rotary.forward already builds cos/sin as cat((freqs, freqs)) over the whole
  head_dim, which IS the layout rotate_half expects, so the full width is used
  directly here. The flash helper is handed the half-width slice only because
  its kernel re-duplicates it internally -- the two are the same numbers.

  Unlike the flash version this is out-of-place, costing one extra
  (B, S, 3, H, D) allocation per block -- the price of needing no flash-attn.
  """
  c = cos[0, :, 0, 0, :][None, :, None, :]        # (1, S, 1, D)
  s = sin[0, :, 0, 0, :][None, :, None, :]
  q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]      # each (B, S, H, D)
  q = q * c + rotate_half(q) * s
  k = k * c + rotate_half(k) * s
  return torch.stack((q, k, v), dim=2)


ATTN_BACKENDS = ('fa2', 'sdpa', 'auto')

_fa2_probe_cache = {}


def flash_attn_kernel_available():
  """Does the installed flash-attn have kernels for THIS device? Probe, cached.

  This cannot be answered from the compute capability, and it cannot be
  answered from the flash-attn version either. Both were tried and both are
  wrong: support depends on which CUDA toolkit the wheel was compiled with, so
  two wheels reporting the same flash_attn.__version__ differ in which cubins
  they carry. The only honest test is to launch a kernel.

  Importing flash-attn tells you nothing here -- on a device with no matching
  cubin the import succeeds, the model builds, and the failure lands on the
  first attention call inside the training loop. scripts/train.sh already
  guards against that by firing a real kernel up front; this is the same idea,
  reusable from Python.

  Two caveats, both reasons this is only reached from attn_backend='auto' and
  never from an explicit backend:
    * It allocates, so it initialises a CUDA context. Under DDP, model
      construction happens before Lightning assigns each rank its device, so
      every rank would probe cuda:0 and pin a context there. The result is
      cached per compute capability, which keeps the ANSWER right (ranks share a
      GPU model), but on multi-GPU prefer an explicit 'fa2' or 'sdpa'.
    * head_dim 64 is probed, not the model's real head_dim. That is the usual
      supported case; an exotic head_dim could in principle pass the probe and
      still fail for real, which the call site in _attn_varlen_fa2 reports.
  """
  if flash_attn is None or not torch.cuda.is_available():
    return False
  cap = torch.cuda.get_device_capability()
  if cap not in _fa2_probe_cache:
    try:
      qkv = torch.zeros(1, 8, 3, 1, 64, device='cuda', dtype=torch.bfloat16)
      flash_attn.flash_attn_qkvpacked_func(qkv)
      torch.cuda.synchronize()
      _fa2_probe_cache[cap] = True
    except Exception:
      # wrong-arch cubin, ABI mismatch, unsupported head_dim -- all mean
      # "do not use this backend", and none of them should be fatal here.
      _fa2_probe_cache[cap] = False
  return _fa2_probe_cache[cap]


def resolve_attn_backend(backend):
  """Map an attn_backend setting onto a concrete kernel choice.

  An explicit 'fa2' or 'sdpa' is honoured as given and never silently
  downgraded: an H200 run and an RTX PRO 6000 run that quietly used different
  kernels would be very hard to notice afterwards. Only 'auto' probes, and it
  says which way it went.
  """
  if backend not in ATTN_BACKENDS:
    raise ValueError(
      f"attn_backend must be one of {ATTN_BACKENDS}, got {backend!r}")

  if backend == 'auto':
    chosen = 'fa2' if flash_attn_kernel_available() else 'sdpa'
    cap = (torch.cuda.get_device_capability()
           if torch.cuda.is_available() else None)
    print(f"[attn] auto -> {chosen}"
          + (f" (device sm_{cap[0]}{cap[1]})" if cap else " (no CUDA device)"))
    return chosen

  if backend == 'fa2' and flash_attn is None:
    raise ImportError(
      "attn_backend='fa2' needs flash-attn, which failed to import: "
      f"{_FLASH_ATTN_IMPORT_ERROR}. Either install a wheel matching this "
      "torch/CUDA/python triple (see README), or use attn_backend='sdpa', "
      "which needs no flash-attn at all.")
  return backend

# function overload
def modulate(x, shift, scale):
  return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

#################################################################################
#                                     Layers                                    #
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


class VectorEmbedder(nn.Module):
    """
    Embeds vector inputs (Binary or One-Hot) into the condition dimension.
    Includes a learned 'null_embedding' for Classifier-Free Guidance.
    """
    def __init__(self, input_dim, cond_dim):
        super().__init__()

        self.linear = nn.Linear(input_dim, cond_dim)
        self.null_embedding = nn.Parameter(torch.randn(cond_dim))
        
        # Initialize
        nn.init.normal_(self.null_embedding, std=0.02)

    def forward(self, x, drop_mask=None):
        """
        x: (Batch, input_dim) - FloatTensor (Binary or One-Hot)
        drop_mask: (Batch,) - BoolTensor. True means 'Drop Condition' (use Null).
        """
        # 1. Project input features
        emb = self.linear(x)
        
        # 2. Apply CFG Dropout
        if drop_mask is not None:
            # Expand null_embedding to match batch size: (B, cond_dim)
            null_emb_expanded = self.null_embedding.unsqueeze(0).expand(x.size(0), -1)
            
            # Where mask is True, use Null. Where False, use projected input.
            drop_mask = drop_mask.view(-1, 1)
            emb = torch.where(drop_mask, null_emb_expanded, emb)
            
        return emb
    
#################################################################################
#                                Core Model                                     #
#################################################################################

class Attention(nn.Module):
  def __init__(self, dim, n_heads, dropout, mlp_ratio=4, attn_backend='fa2'):
    super().__init__()
    self.n_heads = n_heads
    self.attn_backend = resolve_attn_backend(attn_backend)

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

  def _attn_varlen_fa2(self, qkv, seqlens, device, B, L, H, D):
    """Packed variable-length attention through flash-attn.

    Unchanged from the original implementation, so fa2 runs stay bit-for-bit
    reproducible against every checkpoint trained before the backend switch
    existed. Requires a flash-attn wheel carrying cubins for the running
    device's arch -- see the import note at the top of this file.
    """
    qkv = rearrange(qkv, 'b s c h d -> b s (c h d)')

    # create masks for attending only "valid" tokens
    valid_mask = torch.arange(L, device=qkv.device)[None, :] < seqlens[:, None]
    valid_indices = valid_mask.flatten().nonzero(as_tuple=False).squeeze(-1)

    qkv = qkv.flatten(0,1)[valid_indices].view(-1, 3, H, D)

    cu_seqlens = torch.cat((
        torch.tensor([0], device=device, dtype=torch.int32),
        seqlens.to(dtype=torch.int32)
    )).cumsum(dim=0, dtype=torch.int32)

    try:
      x = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
          qkv, cu_seqlens, L, 0., causal=False)
    except RuntimeError as e:
      # A wrong-arch wheel gets past `import flash_attn` and past model
      # construction, and surfaces only here, at the first launch, as a bare
      # "no kernel image is available for execution on the device". Nothing in
      # that message mentions the wheel, the backend flag, or the fix -- so say
      # it. The try costs nothing when no exception is raised (3.11+).
      cap = (torch.cuda.get_device_capability()
             if torch.cuda.is_available() else None)
      raise RuntimeError(
        f"flash-attn's kernel failed"
        + (f" on sm_{cap[0]}{cap[1]}" if cap else "") + f": {e}\n"
        "If that says 'no kernel image is available for execution on the "
        "device', the installed flash-attn wheel carries no cubin for this GPU. "
        "flash-attn 2.x DOES support sm_120 (Blackwell), but only when built "
        "with CUDA >= 12.8 -- the pinned cu126 wheel is not. Fix by either:\n"
        "  - building an environment whose torch and flash-attn wheels carry this GPU's architecture\n"
        "  - or running with attn_backend='sdpa', which needs no flash-attn."
      ) from e

    # apply masks
    out_padded = torch.zeros(B * L, H, D, device=x.device, dtype=x.dtype)
    out_padded[valid_indices] = x

    # rearrange it back
    return rearrange(out_padded.view(B, L, H, D), 'b s h d -> b s (h d)')

  def _attn_padded_sdpa(self, qkv, seqlens, B, L, H, D):
    """Dense padded attention via SDPA -- the Blackwell-capable path.

    SDPA has no varlen/cu_seqlens API, so attention runs at the padded width
    with a key-padding mask rather than on a packed batch. Same numbers as the
    varlen path at every valid position; it just also computes pad rows and
    throws them away.

    Two traps worth keeping in mind before touching this:

      * The mask is key-side only, and BOOL. An additive -inf mask would leave
        pad *query* rows fully masked, and a softmax over an all--inf row is
        NaN, which then spreads through the residual into every token. Masking
        keys only means even a pad query row still has valid keys to attend to,
        so nothing degenerates.
      * The varlen path leaves pad positions as exact zeros and the residual
        path downstream relies on it, so they are zeroed explicitly here.

    softmax_scale is left default in both paths and agrees: flash-attn and SDPA
    both use 1/sqrt(head_dim).
    """
    valid_mask = torch.arange(L, device=qkv.device)[None, :] < seqlens[:, None]

    # (B, S, H, D) -> (B, H, S, D), the layout SDPA expects
    q, k, v = (t.transpose(1, 2)
               for t in (qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]))

    x = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=valid_mask[:, None, None, :],
        dropout_p=0.,
        is_causal=False)

    x = x.transpose(1, 2)                                    # (B, S, H, D)
    x = x * valid_mask[:, :, None, None].to(x.dtype)
    return rearrange(x, 'b s h d -> b s (h d)')

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

    if self.attn_backend == 'fa2':
      qkv = apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
      x = self._attn_varlen_fa2(qkv, seqlens, x.device, B, L, H, D)
    else:
      qkv = apply_rotary_pos_emb_torch(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
      x = self._attn_padded_sdpa(qkv, seqlens, B, L, H, D)

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

class DDiTBlock(nn.Module):
  def __init__(self, dim, n_heads, cond_dim, dropout=0.1, attn_backend='fa2'):
    super().__init__()

    self.attn_seqs = Attention(dim=dim, n_heads=n_heads, dropout=dropout,
                               mlp_ratio=4, attn_backend=attn_backend)

    self.dropout = dropout


    self.adaLN_modulation_seqs = nn.Linear(cond_dim, 6 * dim, bias=True)

    self.adaLN_modulation_seqs.weight.data.zero_()
    self.adaLN_modulation_seqs.bias.data.zero_()


  def forward(self, x, rotary_cos_sin, c, seqlens):

    # (shift_msa, scale_msa, gate_msa, 
    #  shift_mlp, scale_mlp, gate_mlp) 
    adaLN_parm_seqs = self.adaLN_modulation_seqs(c)[:, None].chunk(6, dim=2)

    BS, L, D = x.shape
    x = self.attn_seqs(x, rotary_cos_sin, seqlens, adaLN_parm_seqs)  
    return x

class EmbeddingLayer(nn.Module):
    def __init__(self, dim, tokens_dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((tokens_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        return self.embedding[x]


class DDitFinalLayer(nn.Module):
  def __init__(self, hidden_size, out_channels, cond_dim, seq_length ):
    super().__init__()
    
    self.hidden_size = hidden_size
    self.seq_length = seq_length

    self.norm_final = LayerNorm(hidden_size)

    self.linear_seq = nn.Linear(hidden_size, out_channels)
    self.linear_seq.weight.data.zero_()
    self.linear_seq.bias.data.zero_()


    self.adaLN_modulation = nn.Linear(cond_dim,
                                      2 * hidden_size,
                                      bias=True)
    self.adaLN_modulation.weight.data.zero_()
    self.adaLN_modulation.bias.data.zero_()


  def forward(self, x, c):
    shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
    x = modulate_fused(self.norm_final(x), shift, scale)
    seqs = self.linear_seq(x)

    # dp = torch.einsum("bih,bjh->bij", x, x) # [BS, L, L]

    return seqs

class DIT(nn.Module, huggingface_hub.PyTorchModelHubMixin):
  def __init__(self, 
               vocab_size,
               seq_length = 66,
               hidden_size = 1536,
               cond_dim = 256,
               n_heads = 12,
               n_blocks = 24,
               dropout = 0.2,
               # Updated Condition Dimensions
               species_dim = 6,  # 6 different species (Binary)
               groups_dim = 5,   # 5 different groups (Binary)
               mic_dim = 10,     # 10 mic values (One-Hot)
               # 'fa2'  packed varlen flash-attn. Needs a wheel built for the
               #        device's arch -- on Blackwell that means a cu130 (or
               #        newer) build, not the cu126 one.
               # 'sdpa' dense padded torch SDPA. Needs no flash-attn at all,
               #        so it runs anywhere torch does, Blackwell included.
               # 'auto' probe a real kernel and pick; prints which it took.
               attn_backend = 'fa2'):
    super().__init__()

    self.vocab_size = vocab_size
    self.seq_length = seq_length
    self.attn_backend = resolve_attn_backend(attn_backend)

    self.seqs_embed = EmbeddingLayer(hidden_size, vocab_size)
    self.sigma_map = TimestepEmbedder(cond_dim)
    
    # --- UPDATED: Use VectorEmbedder ---
    self.species_embedder = VectorEmbedder(species_dim, cond_dim)
    self.groups_embedder  = VectorEmbedder(groups_dim, cond_dim)
    self.mic_embedder     = VectorEmbedder(mic_dim, cond_dim)
    
    self.rotary_emb = Rotary(hidden_size // n_heads)

    blocks = []
    for _ in range(n_blocks):
      blocks.append(DDiTBlock(hidden_size, n_heads, cond_dim, dropout=dropout,
                              attn_backend=self.attn_backend))
    self.blocks = nn.ModuleList(blocks)
    
    self.output_layer = DDitFinalLayer(hidden_size, vocab_size, cond_dim, seq_length)

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return bias_dropout_add_scale_fused_inference

  def forward(self, x, sigma, seqlens, species_vec=None, species_mask=None, groups_vec=None, groups_mask=None, mic_vec=None, mic_mask=None, cond_embedding=None):
    """
    x: Sequence indices
    sigma: Timesteps
    *_vec: Feature vectors (Float)
    *_mask: Dropout masks (Bool, True=Drop)
    """

    x = self.seqs_embed(x)

    # 1. Base Conditioning (Time)
    t_emb = self.sigma_map(sigma)
    
    # 2. Add Biological Conditions
    cond_accum = torch.zeros_like(t_emb)
    
    if species_vec is not None:
        cond_accum = cond_accum + self.species_embedder(species_vec, species_mask)
    
    if groups_vec is not None:
        cond_accum = cond_accum + self.groups_embedder(groups_vec, groups_mask)
        
    if mic_vec is not None:
        cond_accum = cond_accum + self.mic_embedder(mic_vec, mic_mask)
        
    if cond_embedding is not None:
        cond_accum = cond_accum + cond_embedding

    # 3. Combine
    c = F.silu(t_emb + cond_accum)

    rotary_cos_sin = self.rotary_emb(x)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
      for i in range(len(self.blocks)):
        x = self.blocks[i](x, rotary_cos_sin, c, seqlens)
      x = self.output_layer(x, c)
    return x

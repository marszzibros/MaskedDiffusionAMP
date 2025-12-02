
import torch
import torch.nn as nn
from torch import Tensor

from dataclasses import dataclass

import math
import abc


torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

class Noiser(abc.ABC, nn.Module):
  """
  Baseline forward method to get the total + rate of noise at a timestep
  """
  def forward(self, t):
    # Assume time goes from 0 to 1
    return self.total_noise(t), self.rate_noise(t)
  
  @abc.abstractmethod
  def rate_noise(self, t):
    """
    Rate of change of noise ie g(t)
    """
    pass

  @abc.abstractmethod
  def total_noise(self, t):
    """
    Total noise ie \int_0^t g(t) dt + g(0)
    """
    pass

class LogLinearNoise(Noiser):
  """Log Linear noise schedule.
  
  Built such that 1 - 1/e^(n(t)) interpolates between 0 and
  ~1 when t varies from 0 to 1. Total noise is
  -log(1 - (1 - eps) * t), so the sigma will be
  (1 - eps) * t.
  """
  def __init__(self, eps=1e-3):
    super().__init__()
    self.eps = eps
    self.sigma_max = self.total_noise(torch.tensor(1.0))
    self.sigma_min = self.eps + self.total_noise(torch.tensor(0.0))

  def rate_noise(self, t):
    return (1 - self.eps) / (1 - (1 - self.eps) * t)

  def total_noise(self, t):
    return -torch.log1p(-(1 - self.eps) * t)

  def importance_sampling_transformation(self, t):
    f_T = torch.log1p(- torch.exp(- self.sigma_max))
    f_0 = torch.log1p(- torch.exp(- self.sigma_min))
    sigma_t = - torch.log1p(- torch.exp(t * f_T + (1 - t) * f_0))
    t = - torch.expm1(- sigma_t) / (1 - self.eps)
    return t



class Diffusion:

    def __init__ (self, max_length):
        # self.dataset = SwissProtDataset(data_path="data/", max_length=max_length, categorical_bin=categorical_bin)
        self.mask_index = 48
        self.blank_index = 30
        self.neg_infinity = -float("inf")
    def _sample_t(self, n):
        _eps_t = torch.rand(n)

        offset = torch.arange(n) / n
        _eps_t = (_eps_t / n + offset) % 1
        t = (1 - 1e-3) * _eps_t + 1e-3
        return t

    def q_xt(self, x, move_chance):

        move_indices = torch.rand(* x.shape, device=x.device) < move_chance
        xt = torch.where(move_indices, self.mask_index, x)

        return xt
    
    def _subs_parameterization(self, logits, xt):
        # log prob at the mask index = - infinity
        logits[:, :, self.mask_index] += self.neg_infinity
        
        # Normalize the logits such that x.exp() is
        # a probability distribution over vocab_size.
        logits = logits - torch.logsumexp(logits, dim=-1,
                                        keepdim=True)

        # Apply updates directly in the logits matrix.
        # For the logits of the unmasked tokens, set all values
        # to -infinity except for the indices corresponding to
        # the unmasked tokens.
        unmasked_indices = (xt != self.mask_index)

        logits[unmasked_indices] = self.neg_infinity
        logits[unmasked_indices, xt[unmasked_indices]] = 0
        return logits

    def _sample_categorical(self, categorical_probs):
        # Gumbel-max sampling
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(categorical_probs) + 1e-10) + 1e-10)
        return (torch.log(categorical_probs + 1e-10) + gumbel_noise).argmax(dim=-1)


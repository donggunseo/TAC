import torch.nn as nn
import torch

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"



class activation_converter(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=True, eps=1e-6):
        super().__init__()
        self.LSTM = nn.LSTM(input_size = input_size, hidden_size=hidden_size, batch_first=batch_first, dtype = torch.bfloat16)
        self.RMSNorm = RMSNorm(hidden_size=hidden_size, eps=eps)
    def forward(self, inputs, hs):
        out, hs_out = self.LSTM(inputs, hs)
        # out = self.RMSNorm(out)
        return out, hs_out


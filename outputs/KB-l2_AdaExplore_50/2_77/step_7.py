import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        self.batch_norm = nn.BatchNorm3d(out_channels, eps=eps, momentum=momentum)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.eps = eps
        self.momentum = momentum

    def forward(self, x):
        y = self.conv_transpose(x)
        N, C, D, H, W = y.shape
        s = self.scale_factor

        gamma = self.batch_norm.weight
        beta = self.batch_norm.bias

        if self.training:
            # Compute per-channel stats on the unscaled y; scaled stats are derived
            # mu_y (C,), var_y (C,) over (N, D, H, W)
            # Sum and sum-sq via reshape
            y_perm = y.permute(1, 0, 2, 3, 4).contiguous().view(C, -1)  # (C, N*D*H*W)
            M = y_perm.shape[1]
            mu_y = y_perm.mean(dim=1)
            var_y = y_perm.var(dim=1, unbiased=False)

            mu = s * mu_y
            var = (s * s) * var_y

            with torch.no_grad():
                # Standard BN running-stat update uses unbiased variance for running_var
                unbiased_var = var * (M / max(M - 1, 1))
                self.batch_norm.running_mean.mul_(1 - self.momentum).add_(mu.detach(), alpha=self.momentum)
                self.batch_norm.running_var.mul_(1 - self.momentum).add_(unbiased_var.detach(), alpha=self.momentum)

            inv = torch.rsqrt(var + self.eps)
            # nc_mean of y: (N, C)
            nc_mean = y.mean(dim=(2, 3, 4))
            a = s * gamma * inv  # (C,)
            b = beta - mu * gamma * inv  # (C,)
            out = nc_mean * a.unsqueeze(0) + b.unsqueeze(0)
            return out.view(N, C, 1, 1, 1)
        else:
            running_mean = self.batch_norm.running_mean
            running_var = self.batch_norm.running_var
            inv = torch.rsqrt(running_var + self.eps)
            nc_mean = y.mean(dim=(2, 3, 4))
            a = s * gamma * inv
            b = beta - running_mean * gamma * inv
            out = nc_mean * a.unsqueeze(0) + b.unsqueeze(0)
            return out.view(N, C, 1, 1, 1)
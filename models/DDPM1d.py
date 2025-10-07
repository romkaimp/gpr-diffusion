import torch
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(
            self,
            base_dim,  # 128
            hidden_dim,  # 256
            output_dim,  # 512
    ):
        super().__init__()

        # In this example, we assume that the number of embedding dimension is always even.
        # (If not, please pad the result.)
        assert (base_dim % 2 == 0)
        self.timestep_dim = base_dim

        self.hidden1 = nn.Linear(
            base_dim,
            hidden_dim)
        self.hidden2 = nn.Linear(
            hidden_dim,
            output_dim)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, picked_up_timesteps):
        """
        Generate timestep embedding vectors

        Parameters
        ----------
        picked_up_timesteps : torch.tensor((batch_size), dtype=int)
            Randomly picked up timesteps

        Returns
        ----------
        out : torch.tensor((batch_size, output_dim), dtype=float)
            Generated timestep embeddings (vectors) for each timesteps.
        """

        # Generate 1 / 10000^{2i / d_e}
        # shape : (timestep_dim / 2, )
        interval = 1.0 / (10000 ** (torch.arange(0, self.timestep_dim, 2.0).to(self.device) / self.timestep_dim))
        # Generate t / 10000^{2i / d_e}
        # shape : (batch_size, timestep_dim / 2)
        position = picked_up_timesteps.type(torch.get_default_dtype())
        radian = position[:, None] * interval[None, :]
        # Get sin(t / 10000^{2i / d_e}) and unsqueeze
        # shape : (batch_size, timestep_dim / 2, 1)
        sin = torch.sin(radian).unsqueeze(dim=-1)
        # Get cos(t / 10000^{2i / d_e}) and unsqueeze
        # shape : (batch_size, timestep_dim / 2, 1)
        cos = torch.cos(radian).unsqueeze(dim=-1)
        # Get sinusoidal positional encoding
        # shape : (batch_size, timestep_dim)
        pe_tmp = torch.concat((sin, cos), dim=-1)  # shape : (num_timestep, timestep_dim / 2, 2)
        d = pe_tmp.size()[1]
        pe = pe_tmp.view(-1, d * 2)  # shape : (num_timestep, timestep_dim)
        # Apply feedforward
        # shape : (batch_size, timestep_dim * 4)
        out = self.hidden1(pe)
        out = F.silu(out)
        out = self.hidden2(out)

        return out

class CovariateEmbedding(nn.Module):
    def __init__(self, covariate_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(covariate_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, covariates):
        """
        covariates: (batch_size, covariate_dim)
        returns: (batch_size, output_dim)
        """
        x = F.silu(self.fc1(covariates))
        x = F.silu(self.fc2(x))
        return x

class ResnetBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        num_norm_groups, # 32
        timestep_embedding_dim,# 512

    ):
        super().__init__()

        # for normalization
        self.norm1 = nn.GroupNorm(
            num_groups=num_norm_groups,
            num_channels=in_channel,
            eps=1e-06,
        )
        self.norm2 = nn.GroupNorm(
            num_groups=num_norm_groups,
            num_channels=out_channel,
            eps=1e-06,
        )

        # for applying conv
        self.conv1 = nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=3,
            stride=1,
            padding="same",
        )
        self.conv2 = nn.Conv2d(
            out_channel,
            out_channel,
            kernel_size=3,
            stride=1,
            padding="same",
        )

        #if we have covariates
        covariates_embedding_dim = timestep_embedding_dim
        observation_time_embedding_dim = timestep_embedding_dim
        total_embedding_dim = (timestep_embedding_dim +
                               covariates_embedding_dim +
                               observation_time_embedding_dim)

        self.linear_emb = nn.Linear(total_embedding_dim, out_channel)

        # for adding timestep
        #self.linear_pos = nn.Linear(timestep_embedding_dim, out_channel)

        # for residual block
        if in_channel != out_channel:
            self.linear_src = nn.Linear(in_channel, out_channel)
        else:
            self.linear_src = None

    def forward(self, x, t_emb, covariates_emb, observation_time_emb):
        """
        Parameters
        ----------
        x : torch.tensor((batch_size, in_channel, width, height), dtype=float)
            input x
        t_emb : torch.tensor((batch_size, base_channel_dim * 4), dtype=float)
            timestep embeddings
        """

        # Apply conv
        out = self.norm1(x)
        out = F.silu(out)
        out = self.conv1(out)

        #if we have covariates
        embeddings = [t_emb]

        if covariates_emb is not None:
            embeddings.append(covariates_emb)

        if observation_time_emb is not None:
            embeddings.append(observation_time_emb)

        # Concatenate along feature dimension
        combined_emb = torch.cat(embeddings, dim=-1)

        # Add timestep encoding
        pos = F.silu(combined_emb) # иначе просто t_emb
        pos = self.linear_emb(pos)
        pos = pos[:, :, None, None]
        out = out + pos

        # apply dropout + conv
        out = self.norm2(out)
        out = F.silu(out)
        out = F.dropout(out, p=0.1, training=self.training)
        out = self.conv2(out)

        # apply residual
        if self.linear_src is not None:
            x_trans = x.permute(0, 2, 3, 1)       # (N,C,H,W) --> (N,H,W,C)
            x_trans = self.linear_src(x_trans)
            x_trans = x_trans.permute(0, 3, 1, 2) # (N,H,W,C) --> (N,C,H,W)
            out = out + x_trans
        else:
            out = out + x

        return out

class ResnetBlockDiff(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        num_norm_groups, # 32
        timestep_embedding_dim,# 512

    ):
        super().__init__()

        # for normalization
        self.norm1 = nn.GroupNorm(
            num_groups=num_norm_groups,
            num_channels=in_channel,
            eps=1e-06,
        )
        self.norm2 = nn.GroupNorm(
            num_groups=num_norm_groups,
            num_channels=out_channel,
            eps=1e-06,
        )

        # for applying conv
        self.conv1 = nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=3,
            stride=1,
            padding="same",
        )
        self.conv2 = nn.Conv2d(
            out_channel,
            out_channel,
            kernel_size=3,
            stride=1,
            padding="same",
        )

        #if we have covariates
        covariates_embedding_dim = timestep_embedding_dim

        total_embedding_dim = (timestep_embedding_dim +
                               covariates_embedding_dim)

        self.linear_emb = nn.Linear(total_embedding_dim, out_channel)

        # for adding timestep
        #self.linear_pos = nn.Linear(timestep_embedding_dim, out_channel)

        # for residual block
        if in_channel != out_channel:
            self.linear_src = nn.Linear(in_channel, out_channel)
        else:
            self.linear_src = None

    def forward(self, x, t_emb, covariates_emb):
        """
        Parameters
        ----------
        x : torch.tensor((batch_size, in_channel, width, height), dtype=float)
            input x
        t_emb : torch.tensor((batch_size, base_channel_dim * 4), dtype=float)
            timestep embeddings
        """

        # Apply conv
        out = self.norm1(x)
        out = F.silu(out)
        out = self.conv1(out)

        #if we have covariates
        embeddings = [t_emb]

        if covariates_emb is not None:
            embeddings.append(covariates_emb)

        # Concatenate along feature dimension
        combined_emb = torch.cat(embeddings, dim=-1)

        # Add timestep encoding
        pos = F.silu(combined_emb) # иначе просто t_emb
        pos = self.linear_emb(pos)
        pos = pos[:, :, None, None]
        out = out + pos

        # apply dropout + conv
        out = self.norm2(out)
        out = F.silu(out)
        out = F.dropout(out, p=0.1, training=self.training)
        out = self.conv2(out)

        # apply residual
        if self.linear_src is not None:
            x_trans = x.permute(0, 2, 3, 1)       # (N,C,H,W) --> (N,H,W,C)
            x_trans = self.linear_src(x_trans)
            x_trans = x_trans.permute(0, 3, 1, 2) # (N,H,W,C) --> (N,C,H,W)
            out = out + x_trans
        else:
            out = out + x

        return out

#
# For the implementation of multi-head attention,
# see https://github.com/tsmatz/nlp-tutorials/blob/master/09_transformer.ipynb
#
class AttentionBlock(nn.Module):
    def __init__(
        self,
        channel,
        num_norm_groups, # 32
    ):
        super().__init__()

        self.norm = nn.GroupNorm(
            num_groups=num_norm_groups,
            num_channels=channel,
            eps=1e-06,
        )

        self.q_layer = nn.Linear(channel, channel)
        self.k_layer = nn.Linear(channel, channel)
        self.v_layer = nn.Linear(channel, channel)

        self.output_linear = nn.Linear(channel, channel, bias=False)

    def forward(self, x):
        channel = x.size(dim=1)
        height = x.size(dim=2)
        width = x.size(dim=3)

        out = self.norm(x)

        # reshape : (N,C,H,W) --> (N,H*W,C)
        out = out.permute(0, 2, 3, 1)
        out = out.view(-1, height*width, channel)

        # generate query/key/value
        q = self.q_layer(out)
        k = self.k_layer(out)
        v = self.v_layer(out)

        # compute Q K^T
        score = torch.einsum("bic,bjc->bij", q, k)

        # scale the result by 1/sqrt(channel)
        score = score / channel**0.5

        # apply softtmax
        score = F.softmax(score, dim=-1)

        # apply dot product with values
        out = torch.einsum("bij,bjc->bic", score, v)

        # apply final linear
        out = self.output_linear(out)

        # reshape : (N,H*W,C) --> (N,C,H,W)
        out = out.view(-1, height, width, channel)
        out = out.permute(0, 3, 1, 2)

        # apply residual
        out = out + x

        return out

class ResnetAndAttention(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        num_norm_groups, # 32
        timestep_embedding_dim, # 512

    ):
        super().__init__()

        self.resnet = ResnetBlock(
            in_channel,
            out_channel,
            num_norm_groups,
            timestep_embedding_dim,

        )
        self.attention = AttentionBlock(
            out_channel,
            num_norm_groups,
        )

    def forward(self, x, t_emb, covariates_emb, observation_time_emb):
        """
        Parameters
        ----------
        x : torch.tensor((batch_size, in_channel, width, height), dtype=float)
            input x
        t_emb : torch.tensor((batch_size, base_channel_dim * 4), dtype=float)
            timestep embeddings
        """
        out = self.resnet(x, t_emb, covariates_emb, observation_time_emb)
        out = self.attention(out)
        return out

class DownSampleTime(nn.Module):
    def __init__(self, channel):
        super().__init__()
        # уменьшаем H (ось -2), не трогая W
        self.conv = nn.Conv2d(
            channel,
            channel,
            kernel_size=(3, 1),   # фильтр только по H
            stride=(2, 1),        # уменьшаем только H
            padding=(1, 0),       # паддинг по H
        )

    def forward(self, x):
        return self.conv(x)


class UpSampleTime(nn.Module):
    def __init__(self, channel):
        super().__init__()
        # корректирующий conv (сохраняет размер)
        self.conv = nn.Conv2d(
            channel,
            channel,
            kernel_size=(3, 1),
            stride=1,
            padding=(1, 0),
        )

    def forward(self, x):
        # увеличиваем только H (ось -2)
        out = F.interpolate(x, scale_factor=(2, 1), mode="nearest")
        out = self.conv(out)
        return out

class UNet(nn.Module):
    def __init__(
        self,
        source_channel, # 3
        unet_base_channel, # 128
        num_norm_groups, # 32
        covariate_dim, #
        time_dim, #
    ):
        super().__init__()

        self.pos_enc = PositionalEncoding(
            base_dim=unet_base_channel,
            hidden_dim=unet_base_channel*2,
            output_dim=unet_base_channel*4,
        )

        self.covariate_embedding = CovariateEmbedding(
            covariate_dim=covariate_dim,
            output_dim=unet_base_channel*4
        )

        self.observation_times_embedding = CovariateEmbedding(
            covariate_dim=time_dim,
            output_dim=unet_base_channel*4,
        )

        self.down_conv = nn.Conv2d(
            source_channel,
            unet_base_channel,
            kernel_size=3,
            stride=1,
            padding="same",
        )
        self.top_to_down = nn.ModuleList([
            # 1st layer
            ResnetBlock(
                in_channel=unet_base_channel,
                out_channel=unet_base_channel,
                num_norm_groups=num_norm_groups,
                timestep_embedding_dim=unet_base_channel*4,
            ),
            ResnetBlock(
                in_channel=unet_base_channel,
                out_channel=unet_base_channel,
                num_norm_groups=num_norm_groups,
                timestep_embedding_dim=unet_base_channel*4,
            ),
            DownSampleTime(
                channel=unet_base_channel,
            ),
            # 2nd layer
            ResnetAndAttention(
                in_channel=unet_base_channel,
                out_channel=unet_base_channel*2,
                num_norm_groups=num_norm_groups,
                timestep_embedding_dim=unet_base_channel*4,
            ),
            ResnetAndAttention(
                in_channel=unet_base_channel*2,
                out_channel=unet_base_channel*2,
                num_norm_groups=num_norm_groups,
                timestep_embedding_dim=unet_base_channel*4,
            ),
            DownSampleTime(
                channel=unet_base_channel*2,
            ),
            # 3rd layer
            ResnetBlock(
                in_channel=unet_base_channel*2,
                out_channel=unet_base_channel*2,
                num_norm_groups=num_norm_groups,
                timestep_embedding_dim=unet_base_channel*4,
            ),
            ResnetBlock(
                in_channel=unet_base_channel*2,
                out_channel=unet_base_channel*2,
                num_norm_groups=num_norm_groups,
                timestep_embedding_dim=unet_base_channel*4,
            ),
            DownSampleTime(
                channel=unet_base_channel*2,
            ),
            # 4th layer
            ResnetBlock(
                in_channel=unet_base_channel*2,
                out_channel=unet_base_channel*2,
                num_norm_groups=num_norm_groups,
                timestep_embedding_dim=unet_base_channel*4,
            ),
            ResnetBlock(
                in_channel=unet_base_channel*2,
                out_channel=unet_base_channel*2,
                num_norm_groups=num_norm_groups,
                timestep_embedding_dim=unet_base_channel*4,
            ),
        ])
        self.middle = nn.ModuleList([
            ResnetBlock(
                in_channel=unet_base_channel*2,
                out_channel=unet_base_channel*2,
                num_norm_groups=num_norm_groups,
                timestep_embedding_dim=unet_base_channel*4,
            ),
            AttentionBlock(
                channel=unet_base_channel*2,
                num_norm_groups=num_norm_groups,
            ),
            ResnetBlock(
                in_channel=unet_base_channel*2,
                out_channel=unet_base_channel*2,
                num_norm_groups=num_norm_groups,
                timestep_embedding_dim=unet_base_channel*4,
            ),
        ])
        self.bottom_to_up = nn.ModuleList([
            # 1st layer
            ResnetBlock(
                in_channel=unet_base_channel*4,
                out_channel=unet_base_channel*2,
                num_norm_groups=num_norm_groups,
                timestep_embedding_dim=unet_base_channel*4,
            ),
            ResnetBlock(
                in_channel=unet_base_channel*4,
                out_channel=unet_base_channel*2,
                num_norm_groups=num_norm_groups,
                timestep_embedding_dim=unet_base_channel*4,
            ),
            ResnetBlock(
                in_channel=unet_base_channel*4,
                out_channel=unet_base_channel*2,
                num_norm_groups=num_norm_groups,
                timestep_embedding_dim=unet_base_channel*4,
            ),
            UpSampleTime(
                channel=unet_base_channel*2,
            ),
            # 2nd layer
            ResnetBlock(
                in_channel=unet_base_channel*4,
                out_channel=unet_base_channel*2,
                num_norm_groups=num_norm_groups,
                timestep_embedding_dim=unet_base_channel*4,
            ),
            ResnetBlock(
                in_channel=unet_base_channel*4,
                out_channel=unet_base_channel*2,
                num_norm_groups=num_norm_groups,
                timestep_embedding_dim=unet_base_channel*4,
            ),
            ResnetBlock(
                in_channel=unet_base_channel*4,
                out_channel=unet_base_channel*2,
                num_norm_groups=num_norm_groups,
                timestep_embedding_dim=unet_base_channel*4,
            ),
            UpSampleTime(
                channel=unet_base_channel*2,
            ),
            # 3rd layer
            ResnetAndAttention(
                in_channel=unet_base_channel*4,
                out_channel=unet_base_channel*2,
                num_norm_groups=num_norm_groups,
                timestep_embedding_dim=unet_base_channel*4,
            ),
            ResnetAndAttention(
                in_channel=unet_base_channel*4,
                out_channel=unet_base_channel*2,
                num_norm_groups=num_norm_groups,
                timestep_embedding_dim=unet_base_channel*4,
            ),
            ResnetAndAttention(
                in_channel=unet_base_channel*3,
                out_channel=unet_base_channel*2,
                num_norm_groups=num_norm_groups,
                timestep_embedding_dim=unet_base_channel*4,
            ),
            UpSampleTime(
                channel=unet_base_channel*2,
            ),
            # 4th layer
            ResnetBlock(
                in_channel=unet_base_channel*3,
                out_channel=unet_base_channel,
                num_norm_groups=num_norm_groups,
                timestep_embedding_dim=unet_base_channel*4,
            ),
            ResnetBlock(
                in_channel=unet_base_channel*2,
                out_channel=unet_base_channel,
                num_norm_groups=num_norm_groups,
                timestep_embedding_dim=unet_base_channel*4,
            ),
            ResnetBlock(
                in_channel=unet_base_channel*2,
                out_channel=unet_base_channel,
                num_norm_groups=num_norm_groups,
                timestep_embedding_dim=unet_base_channel*4,
            ),
        ])
        self.norm = nn.GroupNorm(
            num_groups=num_norm_groups,
            num_channels=unet_base_channel,
            eps=1e-06,
        )
        self.up_conv = nn.Conv2d(
            unet_base_channel,
            source_channel,
            kernel_size=3,
            stride=1,
            padding="same",
        )

    def forward(self, x, n, z, t):
        """
        Parameters
        ----------
        x : torch.tensor((batch_size, in_channel, width, height), dtype=float)
            Gaussian-noised images
        n : torch.tensor((batch_size), dtype=int)
            timestep
        """
        assert t.size(1)%8 == 0, print("time observation points must be dividable by 8")
        buffer = []

        # generate time embedding
        time_embs = self.pos_enc(n)
        covariates_embs = self.covariate_embedding(z)
        observation_time_embs = self.observation_times_embedding(t)
        #
        # Top-to-down
        #

        # apply down-convolution
        i = 0
        out = self.down_conv(x)
        #print(out.shape, i)
        i += 1
        buffer.append(out)
        # apply top-to-down
        for block in self.top_to_down:
            if isinstance(block, ResnetBlock):
                out = block(out, time_embs, covariates_embs, observation_time_embs)
                #print(out.shape, i)
                i += 1
            elif isinstance(block, ResnetAndAttention):
                out = block(out, time_embs, covariates_embs, observation_time_embs)
                #print(out.shape, i)
                i += 1
            elif isinstance(block, DownSampleTime):
                out = block(out)
                #print(out.shape, i)
                i += 1
            else:
                raise Exception("Unknown block")
            buffer.append(out)

        #
        # Middle
        #
        for block in self.middle:
            if isinstance(block, ResnetBlock):
                out = block(out, time_embs, covariates_embs, observation_time_embs)
            elif isinstance(block, AttentionBlock):
                out = block(out)
            else:
                raise Exception("Unknown block")

        #
        # Bottom-to-up
        #

        # apply bottom-to-up
        for block in self.bottom_to_up:
            if isinstance(block, ResnetBlock):
                buf = buffer.pop()
                # print("buf:", buf.shape)
                # print("out:", out.shape, "Resnet")
                out = torch.cat((out, buf), dim=1)
                out = block(out, time_embs, covariates_embs, observation_time_embs)
            elif isinstance(block, ResnetAndAttention):
                buf = buffer.pop()
                # print("buf:", buf.shape)
                # print("out:", out.shape, "ResnenAndAtt")
                out = torch.cat((out, buf), dim=1)
                out = block(out, time_embs, covariates_embs, observation_time_embs)
            elif isinstance(block, UpSampleTime):
                out = block(out)
                # print("out:", out.shape, "UpSampleTime")
            else:
                raise Exception("Unknown block")
        # apply up-convolution
        out = self.norm(out)
        out = F.silu(out)
        out = self.up_conv(out)

        assert not buffer

        return out


class UNetTiny(nn.Module):
    def __init__(
        self,
        source_channel=3,      # вход (например, 3 признака)
        unet_base_channel=64,  # меньше базовый канал
        num_norm_groups=8,     # меньше групп, чтобы было делимо
        covariate_dim=32,
        time_dim=32,
    ):
        super().__init__()

        # --- Эмбеддинги времени и ковариат ---
        self.pos_enc = PositionalEncoding(
            base_dim=unet_base_channel,
            hidden_dim=unet_base_channel*2,
            output_dim=unet_base_channel*4,
        )
        self.covariate_embedding = nn.Sequential(
            nn.Linear(covariate_dim, unet_base_channel*4),
            nn.ReLU()
        )

        self.observation_times_embedding = CovariateEmbedding(
            covariate_dim=time_dim,
            output_dim=unet_base_channel*4,
        )

        # --- Encoder ---
        self.down_conv = nn.Conv2d(
            source_channel, unet_base_channel, kernel_size=3, stride=1, padding=1
        )
        self.enc1 = ResnetBlock(
            in_channel=unet_base_channel,
            out_channel=unet_base_channel,
            num_norm_groups=num_norm_groups,
            timestep_embedding_dim=unet_base_channel*4,
        )
        self.down1 = DownSampleTime(unet_base_channel)

        self.enc2 = ResnetBlock(
            in_channel=unet_base_channel,
            out_channel=unet_base_channel*2,
            num_norm_groups=num_norm_groups,
            timestep_embedding_dim=unet_base_channel*4,
        )
        self.down2 = DownSampleTime(unet_base_channel*2)

        # --- Bottleneck ---
        self.middle = ResnetBlock(
            in_channel=unet_base_channel*2,
            out_channel=unet_base_channel*2,
            num_norm_groups=num_norm_groups,
            timestep_embedding_dim=unet_base_channel*4,
        )

        # --- Decoder ---
        self.up1 = UpSampleTime(unet_base_channel*2)
        self.dec1 = ResnetBlock(
            in_channel=unet_base_channel*2 + unet_base_channel*2,
            out_channel=unet_base_channel,
            num_norm_groups=num_norm_groups,
            timestep_embedding_dim=unet_base_channel*4,
        )

        self.up2 = UpSampleTime(unet_base_channel)
        self.dec2 = ResnetBlock(
            in_channel=unet_base_channel + unet_base_channel,
            out_channel=unet_base_channel,
            num_norm_groups=num_norm_groups,
            timestep_embedding_dim=unet_base_channel*4,
        )

        # --- Output ---
        self.norm = nn.GroupNorm(num_norm_groups, unet_base_channel)
        self.up_conv = nn.Conv2d(
            unet_base_channel, source_channel, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x, n, z, t):
        # Эмбеддинги
        time_embs = self.pos_enc(n)
        covariates_embs = self.covariate_embedding(z)

        observation_time_embs = self.observation_times_embedding(t)

        # --- Encoder ---
        buf = []
        out = self.down_conv(x)
        out = self.enc1(out, time_embs, covariates_embs, observation_time_embs)
        buf.append(out)
        out = self.down1(out)

        out = self.enc2(out, time_embs, covariates_embs, observation_time_embs)
        buf.append(out)
        out = self.down2(out)

        # --- Bottleneck ---
        out = self.middle(out, time_embs, covariates_embs, observation_time_embs)

        # --- Decoder ---
        out = self.up1(out)
        out = torch.cat([out, buf.pop()], dim=1)
        out = self.dec1(out, time_embs, covariates_embs, observation_time_embs)

        out = self.up2(out)
        out = torch.cat([out, buf.pop()], dim=1)
        out = self.dec2(out, time_embs, covariates_embs, observation_time_embs)

        # --- Output ---
        out = self.norm(out)
        out = F.silu(out)
        out = self.up_conv(out)
        return out

class UNetTinyDiff(nn.Module):
    def __init__(
        self,
        source_channel=3,      # вход (например, 3 признака)
        unet_base_channel=64,  # меньше базовый канал
        num_norm_groups=8,     # меньше групп, чтобы было делимо
        covariate_dim=32,
    ):
        super().__init__()

        # --- Эмбеддинги времени и ковариат ---
        self.pos_enc = PositionalEncoding(
            base_dim=unet_base_channel,
            hidden_dim=unet_base_channel*2,
            output_dim=unet_base_channel*4,
        )
        self.covariate_embedding = nn.Sequential(
            nn.Linear(covariate_dim, unet_base_channel*4),
            nn.ReLU()
        )

        # --- Encoder ---
        self.down_conv = nn.Conv2d(
            source_channel, unet_base_channel, kernel_size=3, stride=1, padding=1
        )
        self.enc1 = ResnetBlockDiff(
            in_channel=unet_base_channel,
            out_channel=unet_base_channel,
            num_norm_groups=num_norm_groups,
            timestep_embedding_dim=unet_base_channel*4,
        )
        self.down1 = DownSampleTime(unet_base_channel)

        self.enc2 = ResnetBlockDiff(
            in_channel=unet_base_channel,
            out_channel=unet_base_channel*2,
            num_norm_groups=num_norm_groups,
            timestep_embedding_dim=unet_base_channel*4,
        )
        self.down2 = DownSampleTime(unet_base_channel*2)

        # --- Bottleneck ---
        self.middle = ResnetBlockDiff(
            in_channel=unet_base_channel*2,
            out_channel=unet_base_channel*2,
            num_norm_groups=num_norm_groups,
            timestep_embedding_dim=unet_base_channel*4,
        )

        # --- Decoder ---
        self.up1 = UpSampleTime(unet_base_channel*2)
        self.dec1 = ResnetBlockDiff(
            in_channel=unet_base_channel*2 + unet_base_channel*2,
            out_channel=unet_base_channel,
            num_norm_groups=num_norm_groups,
            timestep_embedding_dim=unet_base_channel*4,
        )

        self.up2 = UpSampleTime(unet_base_channel)
        self.dec2 = ResnetBlockDiff(
            in_channel=unet_base_channel + unet_base_channel,
            out_channel=unet_base_channel,
            num_norm_groups=num_norm_groups,
            timestep_embedding_dim=unet_base_channel*4,
        )

        # --- Output ---
        self.norm = nn.GroupNorm(num_norm_groups, unet_base_channel)
        self.up_conv = nn.Conv2d(
            unet_base_channel, source_channel, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x, n, z):
        # Эмбеддинги
        time_embs = self.pos_enc(n)
        covariates_embs = self.covariate_embedding(z)

        # --- Encoder ---
        buf = []
        out = self.down_conv(x)
        out = self.enc1(out, time_embs, covariates_embs)
        buf.append(out)
        out = self.down1(out)

        out = self.enc2(out, time_embs, covariates_embs)
        buf.append(out)
        out = self.down2(out)

        # --- Bottleneck ---
        out = self.middle(out, time_embs, covariates_embs)

        # --- Decoder ---
        out = self.up1(out)
        out = torch.cat([out, buf.pop()], dim=1)
        out = self.dec1(out, time_embs, covariates_embs)

        out = self.up2(out)
        out = torch.cat([out, buf.pop()], dim=1)
        out = self.dec2(out, time_embs, covariates_embs)

        # --- Output ---
        out = self.norm(out)
        out = F.silu(out)
        out = self.up_conv(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)

        self.emb_layer = nn.Linear(emb_dim, out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, emb):
        h = self.norm1(self.conv1(x))
        h = F.silu(h)

        # добавляем эмбеддинг (broadcast на все пространство)
        emb_out = self.emb_layer(emb).unsqueeze(-1).unsqueeze(-1)
        h = h + emb_out

        h = self.norm2(self.conv2(h))
        return F.silu(h + self.shortcut(x))


class DiffusionResNet(nn.Module):
    def __init__(self,
                 base_channels=64,
                 covariate_dim=16,
                 time_dim=32,
                 emb_dim=128):
        super().__init__()

        # эмбеддинги
        self.pos_enc = PositionalEncoding(base_dim=128, hidden_dim=256, output_dim=emb_dim)
        self.cov_emb = CovariateEmbedding(covariate_dim=covariate_dim, output_dim=emb_dim)
        self.obs_emb = CovariateEmbedding(covariate_dim=time_dim, output_dim=emb_dim)

        # свертка для входа
        self.conv_in = nn.Conv2d(1, base_channels, kernel_size=3, padding=1)

        # residual-блоки
        self.res1 = ResBlock(base_channels, base_channels, emb_dim*3)
        self.res2 = ResBlock(base_channels, base_channels*2, emb_dim*3)
        self.res3 = ResBlock(base_channels*2, base_channels*2, emb_dim*3)

        # выход
        self.norm = nn.GroupNorm(8, base_channels*2)
        self.conv_out = nn.Conv2d(base_channels*2, 1, kernel_size=3, padding=1)

    def forward(self, x, n, z, obs_times):
        """
        x: [B, 1, time_dim, width]
        n: [B] (diffusion step)
        z: [B, covariate_dim]
        obs_times: [B, time_dim]
        """
        # эмбеддинги
        n_emb = self.pos_enc(n)         # [B, emb_dim]
        z_emb = self.cov_emb(z)         # [B, emb_dim]
        obs_emb = self.obs_emb(obs_times)  # [B, emb_dim]

        emb = torch.cat([n_emb, z_emb, obs_emb], dim=-1)  # [B, 3*emb_dim]

        # свертки
        h = self.conv_in(x)
        h = self.res1(h, emb)
        h = self.res2(h, emb)
        h = self.res3(h, emb)

        h = self.norm(h)
        h = F.silu(h)
        out = self.conv_out(h)

        return out
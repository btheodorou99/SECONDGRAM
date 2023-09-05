import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F

class DownBlock(nn.Module):
    def __init__(self, input_dim, output_dim, embedding_dim):
        super().__init__()
        self.ln = nn.LayerNorm([input_dim])
        self.imgFC = nn.Linear(input_dim, embedding_dim)
        self.embdFC = nn.Linear(embedding_dim, embedding_dim)
        self.imgOut = nn.Linear(embedding_dim, output_dim)
        self.embdOut = nn.Linear(embedding_dim, output_dim)
        self.outFC = nn.Linear(output_dim, output_dim)
        self.act = nn.SiLU()

    def forward(self, img, embd):
        img = self.ln(img)
        img = self.act(self.imgFC(img))
        embd = self.act(self.embdFC(embd))
        out = self.imgOut(img) + self.embdOut(embd)
        out = self.act(self.outFC(out))
        return out

class ResidualBlock(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, embedding_dim)
        self.act = nn.SiLU()

    def forward(self, embd):
        temp = self.act(self.fc1(embd))
        temp = self.act(self.fc2(temp))
        embd = embd + temp
        return embd

class UpBlock(nn.Module):
    def __init__(self, input_dim, output_dim, embedding_dim):
        super().__init__()
        self.ln = nn.LayerNorm([input_dim])
        self.imgFC = nn.Linear(input_dim, embedding_dim)
        self.embdFC = nn.Linear(embedding_dim, embedding_dim)
        self.imgOut = nn.Linear(embedding_dim, output_dim)
        self.embdOut = nn.Linear(embedding_dim, output_dim)
        self.outFC = nn.Linear(output_dim, output_dim)
        self.act = nn.SiLU()

    def forward(self, img, img_skip, embd):
        img = self.ln(img)
        img = self.act(self.imgFC(img))
        embd = self.act(self.embdFC(embd))
        out = self.imgOut(img) + self.embdOut(embd) + img_skip
        out = self.act(self.outFC(out))
        return out

class AutoEncoder(nn.Module):
    def __init__(self, image_dim, embed_dim=256, condition=False, cond_dim=None):
        super().__init__()
        self.image_dim = image_dim
        self.time_embed = nn.Sequential(
            nn.Linear(image_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.condition = condition
        if condition:
            self.cond_embed = nn.Sequential(
               nn.Linear(cond_dim, embed_dim),
               nn.SiLU(),
               nn.Linear(embed_dim, embed_dim),
            )

        self.down1 = DownBlock(image_dim, 512, embed_dim)
        self.down2 = DownBlock(512, 256, embed_dim)
        self.down3 = DownBlock(256, 128, embed_dim)
        self.bot1 = ResidualBlock(128)
        self.bot2 = ResidualBlock(128)
        self.up1 = UpBlock(128, 256, embed_dim)
        self.up2 = UpBlock(256, 512, embed_dim)
        self.up3 = UpBlock(512, image_dim, embed_dim)
        self.out = nn.Linear(image_dim, image_dim)

    def timestep_embedding(self, t, dim, max_period=100):
        inv_freq = 1.0 / (max_period ** (th.arange(0, dim, 2, device=t.device).float() / dim))
        pos_enc_a = th.sin(t.repeat(1, dim // 2 + (dim % 2)) * inv_freq)
        pos_enc_b = th.cos(t.repeat(1, dim // 2 + (dim % 2)) * inv_freq)
        pos_enc = th.cat([pos_enc_a, pos_enc_b], dim=-1)[:, :dim]
        return pos_enc

    def forward(self, x, t, cond=None):
        emb = self.timestep_embedding(t.unsqueeze(-1), self.image_dim)
        emb = self.time_embed(emb)
        if self.condition and cond is not None:
            emb = emb + self.cond_embed(cond)

        x2 = self.down1(x, emb)
        x3 = self.down2(x2, emb)
        x4 = self.down3(x3, emb)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)

        x_out = self.up1(x4, x3, emb)
        x_out = self.up2(x_out, x2, emb)
        x_out = self.up3(x_out, x, emb)
        output = self.out(x_out)
        return output

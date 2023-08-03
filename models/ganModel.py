import numpy as np
import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, image_dim, latent_dim, embed_dim=256, condition=False, cond_dim=None):
        super(Generator, self).__init__()
        self.condition = condition
        if condition:
            self.cond_embed = nn.Sequential(
               nn.Linear(cond_dim, embed_dim),
               nn.SiLU(),
               nn.Linear(embed_dim, embed_dim),
            )

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + embed_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, image_dim),
            nn.Tanh()
        )

    def forward(self, z, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((z, self.cond_embed(labels)), -1)
        img = self.model(gen_input)
        return img


class Discriminator(nn.Module):
    def __init__(self, image_dim, embed_dim=256, condition=False, cond_dim=None):
        super(Discriminator, self).__init__()
        self.condition = condition
        if condition:
            self.cond_embed = nn.Sequential(
               nn.Linear(cond_dim, embed_dim),
               nn.SiLU(),
               nn.Linear(embed_dim, embed_dim),
            )

        # Copied from cgan.py
        self.model = nn.Sequential(
            nn.Linear(image_dim+embed_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img, self.cond_embed(labels)), -1)
        validity = self.model(d_in)
        return validity
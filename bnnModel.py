import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F

class BayesLinear(nn.Module):
    r"""
    Applies Bayesian Linear
    Arguments:
        prior_mu (Float): mean of prior normal distribution.
        prior_sigma (Float): sigma of prior normal distribution.
    .. note:: other arguments are following linear of pytorch 1.2.0.
    https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py
    """
    __constants__ = ['prior_mu', 'prior_sigma', 'bias', 'in_features', 'out_features']

    def __init__(self, prior_mu, prior_sigma, in_features, out_features, bias=True):
        super(BayesLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        self.prior_log_sigma = math.log(prior_sigma)
        
        self.weight_mu = nn.Parameter(th.Tensor(out_features, in_features))
        self.weight_log_sigma = nn.Parameter(th.Tensor(out_features, in_features))
        self.register_buffer('weight_eps', None)
                
        if bias is None or bias is False:
            self.bias = False
        else:
            self.bias = True
            
        if self.bias:
            self.bias_mu = nn.Parameter(th.Tensor(out_features))
            self.bias_log_sigma = nn.Parameter(th.Tensor(out_features))
            self.register_buffer('bias_eps', None)
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_log_sigma', None)
            self.register_buffer('bias_eps', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        # Initialization method of Adv-BNN
        stdv = 1. / math.sqrt(self.weight_mu.size(1))
        self.weight_mu.data.uniform_(-stdv, stdv)
        self.weight_log_sigma.data.fill_(self.prior_log_sigma)
        if self.bias:
            self.bias_mu.data.uniform_(-stdv, stdv)
            self.bias_log_sigma.data.fill_(self.prior_log_sigma)
         
    def freeze(self):
        self.weight_eps = th.randn_like(self.weight_log_sigma)
        if self.bias:
            self.bias_eps = th.randn_like(self.bias_log_sigma)
        
    def unfreeze(self):
        self.weight_eps = None
        if self.bias:
            self.bias_eps = None 
            
    def forward(self, input):
        r"""
        Overriden.
        """
        if self.weight_eps is None:
            weight = self.weight_mu + th.exp(self.weight_log_sigma) * th.randn_like(self.weight_log_sigma)
        else:
            weight = self.weight_mu + th.exp(self.weight_log_sigma) * self.weight_eps
        
        if self.bias:
            if self.bias_eps is None:
                bias = self.bias_mu + th.exp(self.bias_log_sigma) * th.randn_like(self.bias_log_sigma)
            else:
                bias = self.bias_mu + th.exp(self.bias_log_sigma) * self.bias_eps                
        else:
            bias = None

        return F.linear(input, self.mask * weight, bias)

    def extra_repr(self):
        r"""
        Overriden.
        """
        return 'prior_mu={}, prior_sigma={}, in_features={}, out_features={}, bias={}'.format(self.prior_mu, self.prior_sigma, self.in_features, self.out_features, self.bias is not None)

class DownBlock(nn.Module):
    def __init__(self, input_dim, output_dim, embedding_dim):
        super().__init__()
        self.ln = nn.LayerNorm([input_dim])
        self.imgFC = BayesLinear(0, 0.1, input_dim, embedding_dim)
        self.embdFC = BayesLinear(0, 0.1, embedding_dim, embedding_dim)
        self.imgOut = BayesLinear(0, 0.1, embedding_dim, output_dim)
        self.embdOut = BayesLinear(0, 0.1, embedding_dim, output_dim)
        self.outFC = BayesLinear(0, 0.1, output_dim, output_dim)
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
        self.fc1 = BayesLinear(0, 0.1, embedding_dim, embedding_dim)
        self.fc2 = BayesLinear(0, 0.1, embedding_dim, embedding_dim)
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
        self.imgFC = BayesLinear(0, 0.1, input_dim, embedding_dim)
        self.embdFC = BayesLinear(0, 0.1, embedding_dim, embedding_dim)
        self.imgOut = BayesLinear(0, 0.1, embedding_dim, output_dim)
        self.embdOut = BayesLinear(0, 0.1, embedding_dim, output_dim)
        self.outFC = BayesLinear(0, 0.1, output_dim, output_dim)
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
            BayesLinear(0, 0.1, image_dim, embed_dim),
            nn.SiLU(),
            BayesLinear(0, 0.1, embed_dim, embed_dim),
        )
        self.condition = condition
        if condition:
            self.cond_embed = nn.Sequential(
               BayesLinear(0, 0.1, cond_dim, embed_dim),
               nn.SiLU(),
               BayesLinear(0, 0.1, embed_dim, embed_dim),
            )

        self.down1 = DownBlock(image_dim, 256, embed_dim)
        self.down2 = DownBlock(256, 128, embed_dim)
        self.bot1 = ResidualBlock(128)
        self.up1 = UpBlock(128, 256, embed_dim)
        self.up2 = UpBlock(256, image_dim, embed_dim)
        self.out = BayesLinear(0, 0.1, image_dim, image_dim)

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

        x3 = self.bot1(x3)

        x_out = self.up1(x3, x2, emb)
        x_out = self.up2(x_out, x, emb)
        output = self.out(x_out)
        return output
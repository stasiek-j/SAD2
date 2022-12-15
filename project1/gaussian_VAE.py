from typing import Tuple

import torch

from nets import *
from torch import distributions as dist


class EncoderGaussian(nn.Module):
    """
    Encoder for VAE which uses Normal distributions.
    It is paramterized by neural network which should approximate means and variances of these distribution.
    """

    def __init__(self, net: NetEncoder) -> None:
        """
        :param net: Network approximating means and variances of normal distribution encoding given datapoint.
        """
        super(EncoderGaussian, self).__init__()
        self.Network = net
        self.distribution = None
        self.sigma = None
        self.mu = None

    def forward(self, x: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """
        :param x:
        :return:
        """
        mu, logsigma = self.Network.forward(x)
        sigma = torch.exp(logsigma) + 0.001
        # print("sigma.shape: ", sigma.shape)
        # print(torch.min(sigma))
        # sigma = torch.diag_embed(sigma)
        # self.distribution = dist.MultivariateNormal(mu, sigma)
        normal = dist.Normal(mu, sigma)
        self.distribution = dist.Independent(normal, 1)
        self.mu = mu
        self.sigma = sigma
        return mu, sigma

    def log_prob(self, z):
        return self.distribution.log_prob(z)

    def sample(self):
        return self.distribution.rsample()


class DecoderGaussian(nn.Module):
    def __init__(self, net: NetDecoder, hyper_sigma=1.) -> None:
        super(DecoderGaussian, self).__init__()
        self.Network = net
        self.hyper = hyper_sigma
        self.distribution = None

    def forward(self, z) -> torch.Tensor:
        ret = self.Network.forward(z)
        self.distribution = dist.Independent(
            dist.Normal(
                ret,
                torch.ones_like(ret) * self.hyper
            )
            , 1
        )
        return ret

    def log_prob(self, x):
        return self.distribution.log_prob(x)

    def sample(self):
        return self.distribution.rsample()


class VAE(nn.Module):

    def __init__(self, encoder, decoder, prior, beta=1, hyper_sigma=1.) -> None:
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior
        self.hyper_sigma = hyper_sigma
        self.beta = beta

    def loss_function(self, x):
        # print("loss_fn: ", x.shape)
        # print(x)
        self.forward(x)

        reconstruction_loss = self.decoder.log_prob(x).mean()
        kl_loss = dist.kl.kl_divergence(self.encoder.distribution, self.prior).mean()
        # print(kl_loss)
        return -1 * (reconstruction_loss - self.beta * kl_loss), -1 * reconstruction_loss, self.beta* kl_loss

    def forward(self, x):
        self.encoder.forward(x)
        z = self.encoder.sample()
        recons = self.decoder.forward(z)
        return recons

    def sample(self):
        z = self.prior.rsample()
        mu = self.decoder.forward(z)
        return self.decoder.sample()

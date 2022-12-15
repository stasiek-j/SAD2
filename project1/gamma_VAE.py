from torch import nn
from torch import distributions as dist
import torch
from nets import *


class DecoderGamma(nn.Module):
    def __init__(self, net, hyper_sigma=1.) -> None:
        super(DecoderGamma, self).__init__()
        self.Network = net
        self.hyper = hyper_sigma
        self.distribution = None

    def forward(self, z):
        alpha, beta, pi = self.Network.forward(z)
        # print("DecoderGamma.alpha", torch.isinf(torch.exp(alpha)).any())
        alpha = torch.exp(alpha) + 0.0001
        beta = torch.exp(beta) + .0001
        pi = pi.view(1, pi.shape[1], pi.shape[0])
        print("mix: ", torch.concat((pi, 1 - pi), 0).shape)#.(2, 1, 0).shape)
        cat = dist.Categorical(torch.concat((pi, 1 - pi), 0))#-1).permute(2, 1, 0))
        print("mix: ", cat.event_shape, cat.batch_shape)
        gamma = dist.Independent(dist.Gamma(alpha, beta), 1)
        print("comp: ", gamma.event_shape, gamma.batch_shape)

        self.distribution = dist.MixtureSameFamily(cat, gamma)
        # print("logprob of 0: ", self.distribution.log_prob(torch.zeros_like(alpha)))
        return alpha

    def log_prob(self, x):
        return self.distribution.log_prob(x)

    def sample(self):
        return self.distribution.rsample()


class VAE(nn.Module):

    def __init__(self, encoder, decoder, prior, beta=1, hyper_sigma=None) -> None:
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior
        self.beta = beta

    def loss_function(self, x):
        # print("loss_fn: ", x.shape)
        self.forward(x)
        # print("Recons: ", self.decoder.log_prob(x))
        print(x)
        reconstruction_loss = self.decoder.log_prob(x).mean()
        kl_loss = dist.kl.kl_divergence(self.encoder.distribution, self.prior).mean()
        # print(kl_loss)
        return -1 * (reconstruction_loss - self.beta * kl_loss), -1 * reconstruction_loss, self.beta * kl_loss

    def forward(self, x):
        self.encoder.forward(x)
        z = self.encoder.sample()
        recons = self.decoder.forward(z)
        return self.decoder.sample()

    def sample(self):
        z = self.prior.rsample()
        mu = self.decoder.forward(z)
        return self.decoder.sample()

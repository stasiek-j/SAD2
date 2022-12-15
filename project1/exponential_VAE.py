import torch
from torch import nn

from gaussian_VAE import *

#
# class EncoderExponential(EncoderGaussian):
#     def __init__(self, net) -> None:
#         """
#         :param net: Network approximating means and variances of normal distribution encoding given datapoint.
#         """
#         super(EncoderExponential, self).__init__(net)
#         self.Network = net
#         self.distribution = None
#         self.rate = None
#
#     def forward(self, x: torch.Tensor) -> Tuple[float, torch.Tensor]:
#         """
#         :param x:
#         :return:
#         """
#         lograte, _ = self.Network.forward(x)
#         rate = torch.exp(lograte) + 0.001
#         # print("sigma.shape: ", sigma.shape)
#         # print(torch.min(sigma))
#         self.distribution = dist.Exponential(rate)
#         self.rate = rate
#         return rate
#


class DecoderExponential(DecoderGaussian):
    def __init__(self, net, hyper_sigma: float) -> None:
        super(DecoderExponential, self).__init__(net)
        self.Network = net
        self.hyper = hyper_sigma
        self.distribution = None

    def forward(self, z) -> torch.Tensor:
        ret = self.Network.forward(z)

        ret = torch.multiply(ret, ret) + 0.0001

        # print(f"using Exp: {dist.Exponential(ret).batch_shape, dist.Exponential(ret).event_shape}")
        # print(f"using Independent: {dist.Independent(dist.Exponential(ret), 1).batch_shape, dist.Independent(dist.Exponential(ret), 1).event_shape}")

        self.distribution = dist.Independent(dist.Exponential(ret), 1)
        return ret


class VAE(nn.Module):

    def __init__(self, encoder, decoder, prior, beta=1, hyper_sigma=1.) -> None:
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior
        self.beta = beta

    def loss_function(self, x):
        self.forward(x)
        reconstruction_loss = self.decoder.log_prob(x).mean()  # Może suma
        print(self.decoder.log_prob(x))
        kl_loss = dist.kl.kl_divergence(self.encoder.distribution, self.prior).mean()
        return -1 * (reconstruction_loss - self.beta * kl_loss), -1 * reconstruction_loss, self.beta * kl_loss

    def forward(self, x):
        self.encoder.forward(x)
        z = self.encoder.sample()
        recons = self.decoder.forward(z)
        return recons

    def sample(self):
        z = self.prior.rsample()
        mu = self.decoder.forward(z)
        return self.decoder.sample()

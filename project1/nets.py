from typing import List
import torch
from torch.nn import functional as F
from torch import nn
from torch.nn import Linear


class NetEncoder(nn.Module):

    def __init__(self, input_size: int, layers: List[int], latent: int):
        """
        Makes simple MLP that's going to use Relu activations. Takes size of input, list of sizes of hidden layers,
        and size of latent space.
        :param input_size: size of input
        :param layers: list of sizes of hidden layers
        :param latent: size of latent space
        """
        super(NetEncoder, self).__init__()
        self.latent = latent
        self.input_size = input_size
        self.layers = nn.ModuleList(
            [Linear(input_size, layers[0])] +
            [Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        self.mu, self.log_sigma = Linear(layers[-1], latent), Linear(layers[-1], latent)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        for fc in self.layers:
            x = F.relu(fc(x))
        mu = self.mu(x)
        log_sig = self.log_sigma(x)
        return mu, log_sig


class NetDecoder(nn.Module):

    def __init__(self, input_size: int, layers: List[int], latent: int):
        """
        Makes simple MLP that's going to use Relu activations. Takes size of input, list of sizes of hidden layers,
        and size of latent space.
        :param input_size: size of input
        :param layers: list of sizes of hidden layers
        :param latent: size of latent space
        """
        super(NetDecoder, self).__init__()
        self.layers = nn.ModuleList(
            [Linear(latent, layers[-1])] +
            [Linear(layers[i], layers[i - 1]) for i in range(len(layers) - 1, 0, -1)])
        self.last = Linear(layers[0], input_size)
        self.latent = latent

    def forward(self, x):
        x = x.view(-1, self.latent)
        for fc in self.layers:
            x = F.relu(fc(x))
        x = self.last(x)
        return x


class NetDecoderGamma(nn.Module):

    def __init__(self, input_size: int, layers: List[int], latent: int):
        """
        Makes simple MLP that's going to use Relu activations. Takes size of input, list of sizes of hidden layers,
        and size of latent space.
        :param input_size: size of input
        :param layers: list of sizes of hidden layers
        :param latent: size of latent space
        """
        super(NetDecoderGamma, self).__init__()
        self.layers = nn.ModuleList(
            [Linear(latent, layers[-1])] +
            [Linear(layers[i], layers[i - 1]) for i in range(len(layers) - 1, 0, -1)])
        self.alpha = Linear(layers[0], input_size)
        self.latent = latent

    def forward(self, x):
        x = x.view(-1, self.latent)
        # print("Input:  ", torch.any(torch.isnan(x)))
        for fc in self.layers:
            x = F.relu(fc(x))
            # print("Forward loop :  ", torch.any(torch.isnan(x)))
        # print("Forward:  ", torch.any(torch.isnan(x)))
        alpha = self.alpha(x)
        return alpha


class NetDecoderGammaMix(nn.Module):

    def __init__(self, input_size: int, layers: List[int], latent: int):
        """
        Makes simple MLP that's going to use Relu activations. Takes size of input, list of sizes of hidden layers,
        and size of latent space.
        :param input_size: size of input
        :param layers: list of sizes of hidden layers
        :param latent: size of latent space
        """
        super(NetDecoderGammaMix, self).__init__()
        self.layers = nn.ModuleList(
            [Linear(latent, layers[-1])] +
            [Linear(layers[i], layers[i - 1]) for i in range(len(layers) - 1, 0, -1)])
        self.alpha = Linear(layers[0], input_size)
        self.beta = Linear(layers[0], input_size)
        self.pi = Linear(layers[0], input_size)
        self.latent = latent

    def forward(self, x):
        x = x.view(-1, self.latent)
        print("Input:  ", torch.any(torch.isnan(x)))
        for fc in self.layers:
            x = F.relu(fc(x))
            print("Forward loop :  ", torch.any(torch.isnan(x)))
        print("Forward:  ", torch.any(torch.isnan(x)))
        alpha = self.alpha(x)
        beta = self.beta(x)
        pi = F.softmax(self.pi(x))
        return alpha, beta, pi


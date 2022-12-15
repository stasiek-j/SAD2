import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
from torch import distributions as dist
from gaussian_VAE import VAE

def matplotlib_imshow(img, one_channel=False):
    """
    Helper function for inline image display
    :param img: Image in a form of np array, or torch tensor
    :param one_channel:Flag checking whether the image is black and white or not
    :return: None
    """
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap='gnuplot2')
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def train_one_epoch(model, device, optimizer, loader, pbar_flag=False):
    running_loss = 0.
    last_loss = 0.
    pbar = tqdm(loader) if pbar_flag else False
    losses  = []
    kls = []
    recons = []
    for i, data in enumerate(loader if not pbar else pbar):
        inputs = data.X
        # print(torch.any(data.X < 0))
        inputs = inputs.to(device)
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Compute the loss and its gradients
        loss, recon, kl = model.loss_function(inputs.view(-1, multiply(data.shape[1:])))
        loss = loss.sum()
        # print("Loss: ", loss, recon, kl)
        loss.backward()

        #Gradient clipping
        torch.nn.utils.clip_grad_value_(model.parameters(), 1)

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        losses.append(loss.item())
        kls.append(kl)
        recons.append(recon)
        if i % 100 == 99:
            last_loss = running_loss / 100  # loss per batch
            if pbar:
                pbar.set_description(f'batch {i + 1} loss: {last_loss}')
            else:
                print(f'batch {i + 1} loss: {last_loss}')
            running_loss = 0.
    return last_loss, mean(losses), mean(kls).item(), mean(recons).item()


def test_one_epoch(model, test_loader):
    losses = []
    kls = []
    recons = []
    for data in test_loader:
        loss, recon, kl = model.loss_function(data.X)
        losses.append(loss.item())
        kls.append(kl)
        recons.append(recon)
    return mean(losses), mean(kls).item(), mean(recons).item()



def create_models(encoder,
                  decoder,
                  net_encoder,
                  net_decoder,
                  input_size=5000,
                  layers=[2048, 1024, 512, 256],
                  latent=4,
                  hyper_sigma=.01,
                  beta=1,
                  device=torch.device('cpu')):

    encoder = encoder(net_encoder(input_size=input_size, layers=layers, latent=latent)).to(device)
    decoder = decoder(net_decoder(input_size=input_size, layers=layers, latent=latent), hyper_sigma).to(device)
    vae = VAE(encoder,
              decoder,
              dist.Independent(dist.Normal(torch.zeros([latent]).to(device),
                                           torch.ones([latent]).to(device)), 1),
              hyper_sigma=hyper_sigma,
              beta=beta).to(device)
    return encoder, decoder, vae



def plot_hist(sparse_mx, savefig=None, show=True, nonzero=False, threshold=10, normalize=False, ax=None,  xlabel="Number of genes", ylabel="Expression level", title=""):
    """
    Plots histogram of flattened sparse matrix, using only values that are less than threshold.
    :param sparse_mx: sparse matrix that is to be plotted
    :param savefig: path where to save the figure
    :param show: whether figure should be showed
    :param nonzero: True if only nonzero values in the matrix are to be plotted False otherwise.
    :param threshold: Biggest value to be plotted
    :param normalize: True if function should return density function, should be used only if nonzero is True
    :return: hist and bins like in np.histogram()
    """
    x, y = sparse_mx.shape
    whole_size = x * y
    indices = sparse_mx.nonzero()
    sparse_mx = sparse_mx[indices]
    nonzero_size = sparse_mx.size
    if threshold:
        sparse_mx = sparse_mx[sparse_mx < threshold]
    cleaned_size = sparse_mx.size
    print(f"Lost {1 - cleaned_size / nonzero_size} due to given threshold({threshold})")
    assert normalize is False or nonzero is True, "normalize should be True only if nonzero is true"
    hist, bins = np.histogram(sparse_mx, density=normalize, bins=100)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if not nonzero:
        hist[0] += whole_size - sum(hist)
        plt.yscale('log')
    if ax:

        ax.bar(bins[:-1], hist, width=threshold/len(bins))
    else:
        plt.bar(bins[:-1], hist, width=threshold/len(bins))

    if savefig:
        plt.savefig(savefig)

    if show:
        plt.show()

    return hist, bins, ax


def multiply(x):
    res = 1
    for i in x:
        res *= i
    return res

def mean(l):
    return sum(l)/len(l)
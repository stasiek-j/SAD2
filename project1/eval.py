import os

import matplotlib.pyplot as plt
import scanpy as sc
from anndata.experimental.pytorch import AnnLoader
from argparse import ArgumentParser
import torch
from sklearn.decomposition import PCA
from utils import *
from utils import create_models
from gaussian_VAE import *
from exponential_VAE import *
import pandas as pd
import numpy as np

device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

def MSE(vae, loader):
    import utils
    ret = []
    for i, data in enumerate(loader):
        r = vae.loss_function(data.X)[1].mean().detach().item()
        out = vae(data.X)
        print(np.mean(((out-data.X)**2).detach().to('cpu').numpy(), axis=1).mean())
        print(np.mean(((data.X)**2).detach().to('cpu').numpy(), axis=1).mean())
        ret += [r]
    return utils.mean(ret)

def loss_on_test(vaes, loader):
    ret = []
    # mses = [[]] * len(vaes)
    for i, vae in enumerate(vaes):
        ret.append((test_one_epoch(vae, loader)[0]))
    return ret
    # for i, data in enumerate(loader):
    #     for j in range(len(vaes)):
    #         r = vaes[j].loss_function(data.X)[1].mean().detach().item()
    #         out = vaes[j](data.X)
    #         mses[j] += [(np.mean(((out-data.X)**2).detach().to('cpu').numpy(), axis=1).mean(),
    #                         np.mean((data.X ** 2).detach().to('cpu').numpy(), axis=1).mean())]
    #         ret[j] += [r]
    #
    # return [mean(x) for x in ret], mses


def categories_ints(lst):
    d = {x: i for i, x in enumerate(set(lst))}
    return [d[x] for x in lst]


def plot_PCA(latent, cell_types, donor, batch, site, path, n_obs=-1):
    principalComponents = PCA(n_components=2).fit_transform(latent.to('cpu'))
    principalComponents = {'1st': principalComponents[:n_obs, 0], '2nd': principalComponents[:n_obs, 1], 'cell_type': categories_ints(cell_types)[:n_obs],
                           'donor': categories_ints(donor)[:n_obs],  'batch': categories_ints(batch)[:n_obs], 'site': categories_ints(site)[:n_obs]}
    df = pd.DataFrame(principalComponents)
    fig = plt.figure(figsize=(15, 12))
    df.plot.scatter(x='1st', y='2nd', c='cell_type', cmap='nipy_spectral', ylabel='y', xlabel='x', ax=fig.add_subplot(2,2,1), title="colored by cell type")
    df.plot.scatter(x='1st', y='2nd', c='donor', cmap='nipy_spectral', ylabel='y', xlabel='x', ax=fig.add_subplot(2,2,2), title="colored by donor id")
    df.plot.scatter(x='1st', y='2nd', c='batch', cmap='nipy_spectral', ylabel='y', xlabel='x', ax=fig.add_subplot(2,2,3), title="colored by batch")
    df.plot.scatter(x='1st', y='2nd', c='site', cmap='nipy_spectral', ylabel='y', xlabel='x', ax=fig.add_subplot(2,2,4), title='colored by site')
    plt.savefig(path)


def PCA_figs(vaes, test, batch_size):
    pca = PCA(.95)
    for vae, name in vaes:
        test_loader = AnnLoader(
            test, batch_size=batch_size,
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
            use_cuda = torch.cuda.is_available()
        )

        latent = []
        cell_types = []
        batch = []
        donor = []
        site = []

        for i, data in enumerate(tqdm(test_loader)):
            vae.encoder(data.X)
            latent.append(vae.encoder.distribution.sample().detach())
            cell_types += data.obs['cell_type'].values.to_list()
            batch += data.obs['batch'].values.to_list()
            donor += data.obs["DonorID"].tolist()
            site += data.obs["Site"].values.to_list()

        latent = torch.cat(latent, dim=0)
        pca.fit(latent.to('cpu'))
        print(f"Number of components explaining 95% of variancy: {pca.n_components_}")
        plot_PCA(latent, cell_types, donor, batch, site, f"images/PCA_gauss_{name}.png")


def main(PATHS):
    train, test = sc.read_h5ad(PATHS["train"]), \
                  sc.read_h5ad(PATHS["test"])



    plt.figure(figsize=(15, 12))
    plt.subplots_adjust(hspace=0.5)
    plt.suptitle("Histograms of training dataset", fontsize=18, y=0.95)

    plot_hist(train.X, ax=plt.subplot(221), show=False, title="Preprocessed dataset")

    plot_hist(train.layers['counts'], ax=plt.subplot(222), show=False,  title="Raw dataset")

    plot_hist(train.X, nonzero=True, ax=plt.subplot(223), show=False, title="Preprocessed dataset without zeroes")

    plot_hist(train.layers['counts'], nonzero=True, ax=plt.subplot(224), show=False, title="Raw dataset without zeroes")

    if not os.path.exists('images'):
        os.makedirs('images')

    plt.savefig("images/histograms.png")

    print(f"Variance of the preprocessed data: {(train.X.data * train.X.data).mean() - train.X.data.mean()**2} \n"
          f"As we can see it is not 1 so the data was not scaled to unit variance.\n")

    print(f"Maximum value of preprocessed data: {train.X.max()},  maximum value of raw data: {train.layers['counts'].max()}\n")

    print(f"sum of values in first 100 cells: {train.X[:100, :].sum()}, \nsum of values in second 100 cells: {train.X[101:200, :].sum()}\n")

    for word, key in [('patients', 'DonorID'), ("labs", "Site"), ("cell types", "cell_type") ]:
        print(f"Number of unique {word} is {len(set(train.obs[key]))}.")
    print("")

    ###### Normalization ##########
    sc.pp.log1p(train)
    sc.pp.log1p(test)

    ###############################
    batch_size_test = 1024
    test_loader = AnnLoader(
        test, batch_size=batch_size_test,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        use_cuda = torch.cuda.is_available()
    )

    _, _, vae1 = create_models(EncoderGaussian, DecoderGaussian, NetEncoder, NetDecoder, latent=2, device=device)
    _, _, vae2 = create_models(EncoderGaussian, DecoderGaussian, NetEncoder, NetDecoder, latent=64, device=device)
    _, _, vae3 = create_models(EncoderGaussian, DecoderGaussian, NetEncoder, NetDecoder, latent=200, device=device)

    vae1.load_state_dict(torch.load(PATHS["lat2"]))
    vae2.load_state_dict(torch.load(PATHS["lat64"]))
    vae3.load_state_dict(torch.load(PATHS["lat200"]))

    lot = loss_on_test([vae1, vae2, vae3], test_loader)

    print(f"Mean loss on test set: {lot}")


    ##############  PCA  #################
    PCA_figs([(vae1, 'lat2'), (vae2, 'lat64'), (vae3, 'lat200')], test, batch_size_test)

    _, _, vae_exp = create_models(EncoderGaussian, DecoderExponential, NetEncoder, NetDecoderGamma, latent=64, device=device)
    vae_exp.load_state_dict(torch.load(PATHS['exp']))
    PCA_figs([(vae_exp, "exp_lat64")], test, batch_size_test)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--lat2", default="trained_models/VAE_latent2_model", help="Path to gaussian model wit latent space size 2")
    parser.add_argument("--lat64", default="trained_models/VAE_latent64_model", help="Path to gaussian model wit latent space size 64")
    parser.add_argument("--lat200", default="trained_models/VAE_latent200_model", help="Path to gaussian model wit latent space size 200")
    parser.add_argument("--exp", default="trained_models/VAE_exp_latent64_model", help="Path to exponential model wit latent space size 64")
    parser.add_argument("--test", default="data/SAD2022Z_Project1_GEX_test.h5ad", help="Path to train dataset")
    parser.add_argument("--train",  default="data/SAD2022Z_Project1_GEX_train.h5ad", help="Path to test dataset")
    
    main(vars(parser.parse_args()))

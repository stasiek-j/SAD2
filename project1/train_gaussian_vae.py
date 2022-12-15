from argparse import ArgumentParser

import scanpy as sc
import torch.optim as optim
from anndata.experimental.pytorch import AnnLoader

from gaussian_VAE import *
from utils import *
from nets import *
import os


def main(n_epochs=20,
         batch_size_train=2048,
         batch_size_test=4096,
         learning_rate=0.0001,
         layers=[2048, 1024, 512, 256],
         latent=4,
         hyper_sigma=.01,
         beta=.01,
         save_to='vae_latest',
         train="data/SAD2022Z_Project1_GEX_train.h5ad",
         test="data/SAD2022Z_Project1_GEX_test.h5ad",
         demo=""):

    # Set device:
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("cuda")
    else:
        device = torch.device('cpu')
        print("cpu")

    # # Set seed:
    random_seed = 1
    torch.manual_seed(random_seed)

    # Load data:
    train, test = sc.read_h5ad(train), \
                  sc.read_h5ad(test)

    # Normalize data:
    sc.pp.log1p(train)
    sc.pp.log1p(test)
    # sc.pp.normalize_total(train, target_sum=1e4)
    # sc.pp.normalize_total(test, target_sum=1e4)
    sc.pp.scale(train)
    sc.pp.scale(test)

    train_loader = AnnLoader(
        train, batch_size=batch_size_train,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        use_cuda=torch.cuda.is_available())

    test_loader = AnnLoader(
        test, batch_size=batch_size_test,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        use_cuda=torch.cuda.is_available())

    input_size = train.X.shape[1]
    print(input_size)

    ############

    encoder = EncoderGaussian(NetEncoder(input_size=input_size, layers=layers, latent=latent)).to(device)
    decoder = DecoderGaussian(NetDecoder(input_size=input_size, layers=layers, latent=latent), hyper_sigma).to(device)
    vae = VAE(encoder,
              decoder,
              dist.Independent(dist.Normal(torch.zeros([latent]).to(device),
                                           torch.ones([latent]).to(device)), 1),
              hyper_sigma=hyper_sigma,
              beta=beta).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

    # Needed for reporting loss:
    epoch_number = 0
    losses = []
    kls = []
    recons = []
    losses_test = []
    kls_test = []
    recons_test = []

    for epoch in range(1, n_epochs + 1):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        vae.train(True)
        avg_loss, losss, kl, recon = train_one_epoch(vae, device, optimizer, train_loader, pbar_flag=True)

        # We don't need gradients on to do reporting
        vae.train(False)

        vae.eval()
        lt, klt, rect = test_one_epoch(vae, test_loader)

        # Update for reporting loss:
        epoch_number += 1
        losses.append(losss)
        kls.append(kl)
        recons.append(recon)
        losses_test.append(lt)
        kls_test.append(klt)
        recons_test.append(rect)

        # Create loss plots:
        if epoch == n_epochs:
            print("Losses plot:")

            plt.figure(figsize=(15, 12))
            plt.subplots_adjust(hspace=0.5)
            plt.suptitle("Training curves", fontsize=18, y=0.95)

            ax1 = plt.subplot(311)
            plt.yscale("log")
            plt.plot(losses,  label='training loss')
            plt.plot(losses_test,  label='test loss')
            plt.ylabel("$-ELBO$")
            plt.tick_params(
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False)  # labels along the bottom edge are of
            plt.legend()

            plt.subplot(312, sharex=ax1)
            plt.yscale("log")
            plt.ylabel("KL Loss")
            plt.plot(kls, label='kl loss')
            plt.plot(kls_test, label='test kl loss')
            plt.tick_params(
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False)  # labels along the bottom edge are of
            plt.legend()

            plt.subplot(313, sharex=ax1)
            plt.yscale("log")
            plt.ylabel("Reconstruction loss")
            plt.xlabel("Epoch number")
            plt.plot(recons, label='-reconstruction loss')
            plt.plot(recons_test, label='-test reconstruction loss')
            plt.legend()

            plt.savefig(f"images/loss_plot_{save_to.split('_')[-1]}")
            plt.show()

    torch.save(vae.state_dict(), f"{save_to}_model")
    return vae


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-e', '--n_epochs', default=20)
    parser.add_argument('--batch_size_train', default=2048)
    parser.add_argument('--batch_size_test', default=4096)
    parser.add_argument('-r', '--learning_rate', default=.0001)
    parser.add_argument('--layers', nargs='*', default=[1024, 256])
    parser.add_argument('--latent', default=4)
    parser.add_argument('--hyper_sigma', default=.01)
    parser.add_argument('--beta', default=1.)
    parser.add_argument('--save_to', default='vae_latest')
    parser.add_argument('--test', default="data/SAD2022Z_Project1_GEX_test.h5ad")
    parser.add_argument("--train", default="data/SAD2022Z_Project1_GEX_train.h5ad")
    parser.add_argument('--demo', action="store_true", default=True)
    args = vars(parser.parse_args())

    if not os.path.exists('trained_models'):
        os.makedirs('trained_models')
    if not os.path.exists('images'):
        os.makedirs('images')



    if not args["demo"]:
        args.pop('demo')
        main(**args)
    else:
        print("Demo mode:")
        main(latent=2, save_to="trained_models/VAE_latent2", test=args['test'], train=args['train'])
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        main(latent=64, save_to="trained_models/VAE_latent64", test=args['test'], train=args['train'])
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        main(latent=200, save_to="trained_models/VAE_latent200", test=args['test'], train=args['train'])
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


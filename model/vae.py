
from config import DEVICE, HYPERPARAMETERS as HP
from dataclasses import dataclass
from datetime import datetime
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.tensorboard as SummaryWriter
from torchvision.transforms import v2
from tqdm.auto import tqdm


@dataclass
class VAEOutput:
    '''
    Dataclass for VAE output

    Attributes:
        z_dist (torch.distributions.Distribution): distribution of the latent variable z
        z_sample (torch.Tensor): sampled value of the latent variable z
        x_recon (torch.Tensor): reconstructed output from VAE
        loss (torch.Tensor): overall (total) loss of the VAE
        loss_recon (torch.Tensor): reconstruction loss term of the VAE loss
        loss_kl (torch.Tensor): KL divergence term of the VAE loss
    '''
    z_dist: torch.distributions.Distribution
    z_sample: torch.Tensor
    x_recon: torch.Tensor
    loss: torch.Tensor
    loss_recon: torch.Tensor
    loss_kl: torch.Tensor


class VAE(nn.Module):
    '''
    Variational Autoencoder (VAE) model class

    Args:
        model_dims(dict): holds the dimensionality of the different model layers
            'input' (int) for the dimensionality of the input data
            'hidden' (int) for the dimensionality of the hidden layer
            'latent' (int) for the dimensionality of the latent space
    '''

    def __init__(self, **model_dims):
        super().__init__()

        input_dim = model_dims['input']
        hidden_dim = model_dims['hidden']
        latent_dim = model_dims['latent']

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(), # Swish activation function
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(), # Swish activation function
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.SiLU(), # Swish activation function
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.SiLU(), # Swish activation function
            nn.Linear(hidden_dim // 8, 2*latent_dim) # 2 for mean and variance
        )

        self.softplus = nn.Softplus()

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 8), 
            nn.SiLU(), # Swish activation function
            nn.Linear(hidden_dim // 8, hidden_dim // 4),
            nn.SiLU(), # Swish activation function
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.SiLU(), # Swish activation function
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.SiLU(), # Swish activation function
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x, eps: float=1e-8):
        '''
        Encodes the input data into the latent space

        Args:
            x (torch.Tensor): input data
            eps (float): small value to avoid numerical instability

        Returns:
            torch.distributions.MultvariateNormal: normal distribution of the encoded data
        '''
        x = self.encoder(x)
        mu, logvar = torch.chunk(x, 2, dim=-1)
        scale = self.softplus(logvar) + eps 
        scale_tril = torch.diag_embed(scale)
        return torch.distributions.MultvariateNormal(mu, scale_tril=scale_tril)

    def reparameterize(self, dist):
        '''
        Reparameterizes the encoded data to sample from the latent space

        Args:
            dist (torch.distributions.MultvariateNormal): normal distribution of the encoded data

        Returns:
            torch.Tensor: data sampled from the latent space
        '''
        return dist.rsample()

    def decode(self, z):
        '''
        Decodes the latent space data back into the original input space

        Args:
            z (torch.Tensor): latent space data

        Returns:
            torch.Tensor: Reconstructed data in the original input dimension/space
        '''
        return self.decoder(z)
    
    def forward(self, x, compute_loss: bool=True):
        '''
        Performs the forward pass of the VAE

        Args:
            x (torch.Tensor): input data
            compute_loss (bool): whether to compute the loss or not (train vs eval mode)

        Returns:
            VAEOutput: VAE output dataclass  
        '''
        dist = self.encode(x)
        z = self.reparameterize(dist)
        x_recon = self.decode(z)

        if not compute_loss:
            return VAEOutput(
                z_dist=dist,
                z_sample=z,
                x_recon=x_recon,
                loss=None,
                loss_recon=None,
                loss_kl=None
            )
        
        # compute loss 
        loss_recon = F.binary_cross_entropy(x_recon, x+0.5, reduction='none').sum(-1).mean()
        std_norm = torch.distributions.MultvariateNormal(
            torch.zeros_like(z, device=z.device),
            scale_tril=torch.eye(z.shape[-1], device=z.device).unsqueeze(0).expand(z.shape[0], -1, -1)
        )
        loss_kl = torch.distributions.kl.kl_divergence(dist, std_norm).mean()

        return VAEOutput(
            z_dist=dist,
            z_sample=z,
            x_recon=x_recon,
            loss=loss_recon+loss_kl,
            loss_recon=loss_recon,
            loss_kl=loss_kl
        )
    

def train_epoch(model, dataloader, optimizer, prev_updates, writer=None):
    '''
    Trains the model on the given data

    Args:
        model (nn.Module): the model to train (in this case, a VAE model)
        dataloader (torch.utils.data.DataLoader): the dataloader for the dataset
        optimizer (torch.optim.): the model optimizer
        prev_updates (int): the iterations the previous training session left off at 
        writer (torch.utils.tensorboard.SummaryWriter): creates model logs

    Returns:
        int: number of training batches so far
    '''
    # set model to training mode
    model.train()

    # train over batches from dataloader
    for batch_idx, (data, _) in enumerate(tqdm(dataloader)):
        # n_upd is the total amount of training batches for any epoch
        n_upd = prev_updates + batch_idx
        # load data onto device and zero the gradients
        data = data.to(DEVICE)
        optimizer.zero_grad()
        # forward pass through model and backpropagate loss to train parameters
        output = model(data)
        train_loss = output.loss
        train_loss.backward()

        # every 100 batches (across any epoch) print batch info and log info with writer
        if n_upd % 100 == 0:
            # calculate gradient norm
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item()**2
            total_norm = total_norm**(1./2)
            # set batch info variables
            loss = train_loss.item()
            loss_recon = output.loss_recon.item()
            loss_kl = output.loss_kl.item()
            n_smpl = n_upd*HP['BATCH_SIZE']
            # print batch info
            print(f'Step {n_upd:,} (N samples: {n_smpl:,}), Loss: {loss:.4f} (Recon: {loss_recon:.4f}, KL: {loss_kl:.4f}) Grad: {total_norm:.4f}')
            # log batch info with writer
            if writer is not None:
                writer.add_scalar('Loss/Train', loss, global_step=n_upd)
                writer.add_scalar('Loss/Train/BCE', loss_recon, global_step=n_upd)
                writer.add_scalar('Loss/Train/KLD', loss_kl, global_step=n_upd)
                writer.add_scalar('GradNorm/Train', total_norm, global_step=n_upd)

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # update model parameters
        optimizer.step()

    # return most recent (total) training batch number
    return prev_updates + len(dataloader)


def test_epoch(model, dataloader, curr_step, writer=None):
    '''
    Tests the model on the given data

    Args:
        model (nn.Module): the model to train (in this case, a VAE model)
        dataloader (torch.utils.data.DataLoader): the dataloader for the dataset
        curr_step (int): the current step the model is on 
        writer (torch.utils.tensorboard.SummaryWriter): creates model logs
    '''
    # set model to evaluation mode
    model.eval()

    test_loss = 0
    test_loss_recon = 0
    test_loss_kl = 0
    img_size = torch.Size([-1, 1, HP['IMG_PX_LEN'], HP['IMG_PX_LEN']])

    # evaluate without running gradient
    with torch.no_grad():
        for data, _ in tqdm(dataloader, desc='Testing'):
            # load data onto device, then flatten the data
            data = data.to(DEVICE)
            data = data.view(data.size(0), -1) 
            # forward pass through model and add loss to test loss variables
            output = model(data, compute_loss=True)
            test_loss += output.loss.item()
            test_loss_recon += output.loss_recon.item()
            test_loss_kl += output.loss_kl.item()

    # normalize test loss results and print out updated loss 
    num_data = len(dataloader)
    test_loss /= num_data
    test_loss_recon /= num_data
    test_loss_kl /= num_data
    print(f'====> Test set loss: {test_loss:.4f} (BCE: {test_loss_recon:.4f}, KLD: {test_loss_kl:.4f})')

    # log test info with writer
    if writer is not None:
        # log updated loss
        writer.add_scalar('Loss/Test', test_loss, global_step=curr_step)
        writer.add_scalar('Loss/Test/BCE', test_loss_recon, global_step=curr_step)
        writer.add_scalar('Loss/Test/KLD', test_loss_kl, global_step=curr_step)
        # log reconstructions
        writer.add_images('Test/Reconstructions', output.x_recon.view(img_size), global_step=curr_step)
        writer.add_images('Test/Originals', data.view(img_size), global_step=curr_step)
        # log random samples from latent space
        z = torch.randn(16, HP['LATENT_DIM']).to(DEVICE)
        samples = model.decode(z)
        writer.add_images('Test/Samples', samples.view(img_size), global_step=curr_step)


def main():
    # load and transform data
    #   data transformation
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Lambda(lambda x: x.view(-1) - 0.5)
    ])
    #   training data and dataloader
    # train_data = 
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=HP['BATCH_SIZE'], shuffle=True)
    #   test data and dataloader
    # test_data =
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=HP['BATCH_SIZE'], shuffle=False)
    
    # define model and optimizer
    nn_dims = {'input':HP['IMG_PX_LEN']**2, 'hidden':HP['HIDDEN_DIM'], 'latent':HP['LATENT_DIM']}
    model = VAE(nn_dims).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=HP['LR'], weight_decay=HP['WT_DECAY'])

    # define summary writer
    t0 = datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_location = f'runs/mnist/vae_{t0}'
    writer = SummaryWriter(summary_location)

    # training loop over each epoch 
    prev_updates = 0
    num_epochs = HP['NUM_EPOCHS']
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        prev_updates = train_epoch(model, train_loader, optimizer, prev_updates, writer)
        test_epoch(model, test_loader, prev_updates, writer)

    tf = datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f'Finished in {tf-t0}')
    return 0
    

if __name__ == '__main__':
    main()


# import config as cnfg
# import io
# import matplotlib
# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import ImageGrid
# import numpy as np
# import os
# import pandas as pd
# from pathlib import Path
# from PIL import Image
# import torch 
# from torch.distributions.normal import Normal
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from torchvision import datasets, transforms
# from torchvision.datasets.vision import VisionDataset
# from torchvision.utils import save_image, make_grid
# from xml.etree import ElementTree as ET

# # change the backend based on the non-gui backend available
# matplotlib.use("agg")


# class KanaKanjiDataset(Dataset):
#     '''KanaKanji dataset'''

#     def __init__(self, root, train, transform) -> None:
#         """
#         Arguments:
#             root_dir (string): Directory with all the images.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#         """
#         self.root_dir = root
#         self.png_dir = os.path.join(self.root_dir, 'kanakanji_png')
#         kanakanji_csv_path = os.path.join(self.root_dir, 'kanakanji_info.csv')
#         self.kanakanji_info = pd.read_csv(kanakanji_csv_path)

#         self.transform = transform

#     def __len__(self):
#         return len(self.kanakanji_info)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         img_name = os.path.join(self.png_dir, self.kanakanji_info['pngFile'].iloc[idx])
#         image = io.imread(img_name)
#         landmarks = self.landmarks_frame.iloc[idx, 1:]
#         landmarks = np.array([landmarks], dtype=float).reshape(-1, 2)
#         sample = {'image': image, 'landmarks': landmarks}

#         if self.transform:
#             sample = self.transform(sample)

#         return sample
    

# def get_train_loader():

#     return DataLoader(trainset, batch_size=cnfg.BATCH_SIZE, shuffle=True)

# def get_test_loader():

#     return DataLoader(testset, batch_size=cnfg.BATCH_SIZE, shuffle=True)


# class VAE(nn.Module):

#     def __init__(self, embedding_dim, shape_before_flattening):
#         super().__init__()

#         # encoder
#         #   define the convolutional layers (and activation layers) 
#         #       for downsampling and feature extraction
#         #   as well as the flatten layer
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 32, 3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, 3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 128, 3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Flatten()
#         )
#         #   define fully connected layers to transform the tensor into the desired embedding dimensions
#         H_fc = np.prod(shape_before_flattening)
#         self.fc_mean = nn.Linear(H_fc, embedding_dim)
#         self.fc_log_var = nn.Linear(H_fc, embedding_dim)
        
#         # decoder
#         #   define a fully connected layer to transform the latent vector back to the shape before flattening
#         self.fc = nn.Linear(embedding_dim, H_fc)
#         #   define a reshape function to reshape the tensor back to its original shape
#         self.reshape = lambda x: x.view(-1, *shape_before_flattening)
#         #   define the transposed convolutional layers (and activation layers)
#         #       for the decoder to upsample and generate the reconstructed image
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
#             nn.Sigmoid()
#         )

        
#     def sample(self, mean, log_var):
#         # get the shape of the tensor for the mean and log variance
#         batch, dim = mean.shape
#         # generate a normal random tensor (epsilon) with the same shape as z_mean
#         # this tensor will be used for reparameterization trick
#         epsilon = Normal(0, 1).sample((batch, dim)).to(mean.device)
#         # apply the reparameterization trick to generate the samples in the latent space
#         return mean + epsilon*torch.exp(log_var/2)
    
#     def encode(self, x):
#         # apply convolutional layers with relu activation function and flatten tensor
#         x = self.encoder(x)
#         # get the mean and log variance of the latent space distribution
#         z_mean = self.fc_mean(x)
#         z_log_var = self.fc_log_var(x)
#         # sample a latent vector using the reparameterization trick
#         z = self.sample(z_mean, z_log_var)
#         return z_mean, z_log_var, z

#     def decode(self, x):
#         # pass the latent vector through the fully connected layer
#         x = self.fc(x)
#         # reshape the tensor
#         x = self.reshape(x)
#         # apply transposed convolutional layers with given activation functions:
#         #   either relu or sigmoid depending on the layer
#         return self.decoder(x)
    
#     def forward(self, x):
#         # pass the input through the encoder to get the latent vector
#         z_mean, z_log_var, z = self.encode(x)
#         # pass the latent vector through the decoder to get the reconstructed image
#         x_recon = self.decode(z)
#         # return the mean, log variance and the reconstructed image
#         return z_mean, z_log_var, x_recon


# def reconstruction_loss(reconstruction, x):
#     bce_loss = nn.BCELoss()
#     return bce_loss(reconstruction, x)

# def gaussian_kl_loss(mu, log_var):
#     # see Appendix B from VAE paper:
#     # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
#     # https://arxiv.org/abs/1312.6114
#     KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
#     return KLD.mean()

# def vae_loss(y_pred, y_true):
#     mu, logvar, recon_x = y_pred
#     recon_loss = reconstruction_loss(recon_x, y_true)
#     kld_loss = gaussian_kl_loss(mu, logvar)
#     return 500 * recon_loss + kld_loss

# def eval_vae_loss(vae, data):
#     data = data.to(cnfg.DEVICE)
#     # forward pass through the VAE
#     pred = vae(data)
#     # compute and return the VAE loss
#     return vae_loss(pred, data)


# def train(vae, train_loader, optimizer):
#     # set the vae model to train mode
#     # and move it to CPU/GPU
#     vae.train()
#     vae.to(cnfg.DEVICE)
#     running_loss = 0.0
#     # loop over the batches of the training dataset
#     for batch_idx, (data, _) in enumerate(train_loader):
#         optimizer.zero_grad()
#         # compute vae loss
#         loss = eval_vae_loss(vae, data)
#         # backward pass and optimizer step
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#     # compute and return average loss for the epoch
#     return running_loss / len(train_loader)

# def validate(vae, test_loader):
#     # set the vae model to eval mode
#     # and move it to CPU/GPU
#     vae.eval()
#     vae.to(cnfg.DEVICE)
#     running_loss = 0.0
#     # loop over the batches of the training dataset
#     for data, _ in test_loader:
#         # compute vae loss
#         loss = eval_vae_loss(vae, data)
#         running_loss += loss.item()
#     return running_loss

# def train_vae(train_loader, test_loader):
#     # instantiate the vae model
#     vae = VAE(cnfg.EMBEDDING_DIM, cnfg.SHAPE_BEFORE_FLATTENING).to(cnfg.DEVICE)
#     # instantiate optimizer and scheduler
#     optimizer = optim.Adam(list(vae.parameters()), lr=cnfg.LR)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode="min", factor=0.1, patience=cnfg.PATIENCE, verbose=True
#     )

#     # initialize the best validation loss as infinity
#     best_val_loss = float("inf")
#     # start training by looping over the number of epochs
#     for epoch in range(cnfg.EPOCHS):
#         # compute average loss for the epoch
#         train_loss = train(vae, train_loader, optimizer)
#         # compute validation loss for the epoch
#         val_loss = validate(vae, test_loader)
#         # save best vae model weights based on validation loss
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             torch.save(
#                 {"vae": vae.state_dict()},
#                 cnfg.MODEL_WEIGHTS_PATH,
#             )
#         # print training and validation loss at every 20 epochs
#         if epoch % 20 == 0 or (epoch+1) == cnfg.EPOCHS:
#             print(
#                 f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
#             )
#         # adjust learning rate based on the validation loss
#         scheduler.step(val_loss)


# def main():

#     # define the transformation to be applied to the data
#     transform = transforms.Compose(
#         [transforms.Pad(padding=2), transforms.ToTensor()]
#     )
#     # print(datasets.FashionMNIST("data").__dir__())
#     # quit()
#     # load the FashionMNIST training data and create a dataloader

#     trainset = datasets.FashionMNIST(
#         "data", train=True, download=True, transform=transform
#     )
#     print(len(trainset))
#     print(trainset[0][0].size())

#     testset = datasets.FashionMNIST(
#         "data", train=False, download=True, transform=transform
#     )
#     print(len(testset))
#     print(testset[0][0].size())

#     trainset = KanaKanjiDataset(
#         'kanakanji_assets', train=True, transform=transform
#     )
#     # print(trainset[0])
#     # print(trainset[1])
#     # print(trainset[2])
#     # print(trainset[3])
    
#     # # load the FashionMNIST training data and create a dataloader
#     # trainset = datasets.FashionMNIST(
#     #     "data", train=True, download=True, transform=transform
#     # )
#     # print(len(trainset))
#     # print(trainset[0])


#     # train_vae()

    

#     # kanakanji_assets_folder = 'kanakanji_assets'
#     # kanakanji_info_csv = os.path.join(kanakanji_assets_folder, 'kanakanji_info.csv')
#     # png_folder = os.path.join(kanakanji_assets_folder, 'kanakanji_png')
#     # df = pd.read_csv(kanakanji_info_csv)
#     # png_file = os.path.join(png_folder, '09f9d.png')


#     return 0

# if __name__ == '__main__':
#     main()

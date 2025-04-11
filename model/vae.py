
from config import DEVICE, VAE_HYPERPARAMETERS as HP
from dataclasses import dataclass
from datetime import datetime
import os 
import pandas as pd
# from PIL import Image
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.utils.tensorboard as SummaryWriter
from torchvision.io import read_image
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
            'img_len' (int) for the pixel length of the images
            'out_ch' (int) for the number of the channels produced by the first conv2d layer
            'latent' (int) for the dimensionality of the latent space
    '''

    def __init__(self, **model_dims):
        super().__init__()

        img_len = model_dims['img_len']
        out_ch_dim = model_dims['out_ch']
        latent_dim = model_dims['latent']
        h_fc = img_len**2
        img_4Conv = img_len//(2**4)
        pre_flatten_size = torch.Size([8*out_ch_dim, img_4Conv, img_4Conv])

        self.encoder = nn.Sequential(
            nn.Conv2d(1, out_ch_dim, 4, stride=2, padding=1),
            nn.SiLU(), # Swish activation function
            nn.Conv2d(out_ch_dim, 2*out_ch_dim, 4, stride=2, padding=1),
            nn.SiLU(), # Swish activation function
            nn.Conv2d(2*out_ch_dim, 4*out_ch_dim, 4, stride=2, padding=1),
            nn.SiLU(), # Swish activation function
            nn.Conv2d(4*out_ch_dim, 8*out_ch_dim, 4, stride=2, padding=1),
            nn.SiLU(), # Swish activation function
            nn.Flatten(),
            nn.Linear(h_fc, 2*latent_dim) # 2 for mean and variance
        )
        # turns out: img_len**2 = 8*out_ch_dim*(img_4Conv)**2

        self.softplus = nn.Softplus()

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h_fc),
            nn.Unflatten(-1, pre_flatten_size),
            nn.ConvTranspose2d(8*out_ch_dim, 4*out_ch_dim, 4, stride=2, padding=1),
            nn.SiLU(), # Swish activation function
            nn.ConvTranspose2d(4*out_ch_dim, 2*out_ch_dim, 4, stride=2, padding=1),
            nn.SiLU(), # Swish activation function
            nn.ConvTranspose2d(2*out_ch_dim, out_ch_dim, 4, stride=2, padding=1),
            nn.SiLU(), # Swish activation function
            nn.ConvTranspose2d(out_ch_dim, 1, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x, eps: float=1e-8):
        '''
        Encodes the input data into the latent space

        Args:
            x (torch.Tensor): input data
            eps (float): small value to avoid numerical instability

        Returns:
            torch.distributions.MultivariateNormal: normal distribution of the encoded data
        '''
        x = self.encoder(x)
        mu, logvar = torch.chunk(x, 2, dim=-1)
        scale = self.softplus(logvar) + eps 
        scale_tril = torch.diag_embed(scale)
        return torch.distributions.MultivariateNormal(mu, scale_tril=scale_tril)

    def reparameterize(self, dist):
        '''
        Reparameterizes the encoded data to sample from the latent space

        Args:
            dist (torch.distributions.MultivariateNormal): normal distribution of the encoded data

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
        std_norm = torch.distributions.MultivariateNormal(
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
            n_smpl = n_upd*HP.batch_size
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
    img_size = torch.Size([-1, 1, HP.img_px_len, HP.img_px_len])

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
        z = torch.randn(16, HP.latent_dim).to(DEVICE)
        samples = model.decode(z)
        writer.add_images('Test/Samples', samples.view(img_size), global_step=curr_step)


class KanaKanjiDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None):
        self.img_dir = os.path.join(root_dir, 'kanakanji_png')
        self.img_labels = pd.read_csv(os.path.join(root_dir, 'kanakanji_info.csv'))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = self.img_labels['pngFile'].iloc[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = read_image(img_path)
        label = self.img_labels['kanakanji'].iloc[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    

def main():
    data_root_dir = 'kanakanji_assets'
    # load and transform data
    #   data transformation
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Lambda(lambda x: x - 0.5)
    ])
    #   training data and dataloader
    train_data = KanaKanjiDataset(root_dir=data_root_dir, train=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=HP.batch_size, shuffle=True)
    #   test data and dataloader
    test_data = KanaKanjiDataset(root_dir=data_root_dir, train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=HP.batch_size, shuffle=False)
    
    # define model and optimizer
    nn_dims = {'img_len':HP.img_px_len, 'out_ch':HP.out_ch_dim, 'latent':HP.latent_dim}
    model = VAE(nn_dims).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=HP.lr, weight_decay=HP.wt_decay)

    # define summary writer
    t0 = datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_location = f'{data_root_dir}/runs/vae_{t0}'
    writer = SummaryWriter(summary_location)

    # training loop over each epoch 
    prev_updates = 0
    num_epochs = HP.num_epochs
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        prev_updates = train_epoch(model, train_loader, optimizer, prev_updates, writer)
        test_epoch(model, test_loader, prev_updates, writer)

    tf = datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f'Finished in {tf-t0}')
    return 0
    

if __name__ == '__main__':
    # main()

    
    data_root_dir = 'kanakanji_assets'
    # load and transform data
    #   data transformation
    # transform = v2.Compose([
    #     v2.ToImage(),
    #     v2.ToDtype(torch.float32, scale=True),
    #     v2.Lambda(lambda x: x - 0.5)
    # ])
    train_data = KanaKanjiDataset(root_dir=data_root_dir)#, train=True)#, transform=transform)
    print(len(train_data))


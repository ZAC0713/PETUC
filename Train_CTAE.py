import os
import time
import yaml
import torch
import nibabel as nib
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from model.ViTAE import ViTAE


def clear_memory():
    """Clear GPU memory cache to prevent memory leaks"""
    torch.cuda.empty_cache()


class CT_Dataset(Dataset):
    """Custom Dataset for loading CT images in NIfTI format"""

    def __init__(self, images_folder):
        self.images_folder = images_folder

        # List all NIfTI files in the folder (.nii or .nii.gz extensions)
        self.images = sorted(
            [img for img in os.listdir(images_folder) if img.endswith(".nii") or img.endswith(".nii.gz")])

        self.num_samples = len(self.images)

    def __len__(self):
        """Return the total number of samples in the dataset"""
        return self.num_samples

    def __getitem__(self, index):
        """Load and preprocess a single CT image at the given index"""
        img_name = os.path.join(self.images_folder, self.images[index])
        image = nib.load(img_name).get_fdata()  # Load NIfTI data as numpy array
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        return image


def train(config: str):
    """Main training function for the AE model

    Args:
        config: Path to the YAML configuration file
    """
    # Load configuration from YAML file
    config = yaml.load(open(config, 'rb'), Loader=yaml.FullLoader)

    # Extract model configuration from config with default values
    in_channels = config.get('in_channels', 1)
    out_channels = config.get('out_channels', 1)
    img_size = config.get('img_size', 128)
    patch_size = config.get('patch_size', 4)
    dim = config.get('dim', 512)
    depth = config.get('depth', 8)
    num_heads = config.get('num_heads', 8)
    mlp_dim = config.get('mlp_dim', 2048)

    # Extract training parameters from config with default values
    epochs = config.get('epochs', 200)
    warm_epochs = config.get('warm_epochs', 20)
    batch_size = config.get('batch_size', 8)
    batch_print_interval = (config.get('batch_print_interval', 50))
    checkpoint_save_interval = config.get('checkpoint_save_interval', 5)
    save_path = config.get('save_path', './checkpoints')
    device = config.get('device', 'cuda')

    # Print model configuration
    print('Model config:')
    print(f'in_channels: {in_channels}  out_channels: {out_channels}')
    print(f'img_size: {img_size}  patch_size: {patch_size}')
    print(f'dim: {dim}  depth: {depth}  num_heads: {num_heads}  mlp_dim: {mlp_dim}')
    # Print training parameters
    print('Training config:')
    print(f'epochs: {epochs}  warm_epochs: {warm_epochs}')
    print(f'batch_size: {batch_size}')
    print(f'batch_print_interval: {batch_print_interval}')
    print(f'checkpoint_save_interval: {checkpoint_save_interval}')
    print(f'save_path: {save_path}')
    print(f'device: {device}')

    # Dataset acquisition
    image_folder_CT = "Your path"  # Replace with actual path to CT images
    dataset = CT_Dataset(images_folder=image_folder_CT)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model loading
    # Initialize Vision Transformer AE model
    model = ViTAE(
        in_channels=in_channels,out_channels=out_channels, img_size=img_size,patch_size=patch_size,dim=dim,depth=depth,num_heads=num_heads,mlp_dim=mlp_dim)
    model.to(device)  # Move model to specified device (CPU/GPU)
    model.train()

    # Optimizer and learning rate scheduler setup
    optimizer = Adam([{'params': model.parameters()}], lr=1e-4, eps=1e-8, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=epochs - warm_epochs,  # Schedule length after warmup
        eta_min=1e-6,  # Minimum learning rate
    )

    # Create checkpoint directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Main training loop
    for epoch in range(epochs):
        clear_memory()  # Clear GPU memory at the start of each epoch
        print('---------------------------------------------------')
        start_time = time.time()

        # Iterate through batches
        for batch, (images) in enumerate(dataloader):
            optimizer.zero_grad()  # Clear gradients from previous batch

            # Move data to device
            CT = images.to(device)

            # Forward pass: reconstruct input images
            CT_rcon = model(CT)

            # Compute reconstruction loss (Mean Squared Error)
            loss = F.mse_loss(CT, CT_rcon)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            end_time1 = time.time()

            # Print training progress at specified intervals
            if batch % batch_print_interval == 0:
                print(
                    f'[Epoch {epoch}] [batch {batch}] loss: {loss.item():.5f} Time: {end_time1 - start_time:.2f} seconds')

        # Update learning rate after warmup period
        if epoch >= warm_epochs:
            scheduler.step()

        # Save model checkpoint at specified intervals
        if (epoch > 0 and epoch % checkpoint_save_interval == 0) or epoch == epochs - 1 or epoch == 0:
            print(f'Saving model {epoch} to {save_path}...')
            save_dict = dict(model=model.state_dict())
            torch.save(save_dict,os.path.join(save_path, f'CTAE_{epoch}.pth'))


if __name__ == '__main__':
    # Start training with the specified configuration file
    train(config="./config/train_CTAE.yaml")

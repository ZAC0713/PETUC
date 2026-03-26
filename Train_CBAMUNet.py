import os
import torch
import yaml
import time
import nibabel as nib
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from model.CBAMUNET_MICCAI import CBAMUNet_MICCAI
from model.CBAMUNET_NPJ import CBAMUNet_NPJ
from model.ViTAE import ViTEncoder
from Loss import Loss_Zoo

def clear_memory():
    torch.cuda.empty_cache()

class PETMR_Correction_Dataset(Dataset):
    """Custom Dataset for loading paired PETCT, PETMR and MRI images in NIfTI format"""

    def __init__(self, images_folder1, images_folder2, images_folder_MRI):
        self.images_folder1 = images_folder1
        self.images_folder2 = images_folder2
        self.images_folder_MRI = images_folder_MRI

        # List all NIfTI files in both folders
        self.images1 = sorted(
            [img for img in os.listdir(images_folder1) if img.endswith(".nii") or img.endswith(".nii.gz")])
        self.images2 = sorted(
            [img for img in os.listdir(images_folder2) if img.endswith(".nii") or img.endswith(".nii.gz")])
        self.images_MRI = sorted(
            [img for img in os.listdir(images_folder_MRI) if img.endswith(".nii") or img.endswith(".nii.gz")])

        # Ensure both folders have the same number of images
        if len(self.images1) != len(self.images2):
            raise ValueError("Folders must contain the same number of images.")
        if len(self.images1) != len(self.images_MRI):
            raise ValueError("Folders must contain the same number of images.")

        self.num_samples = len(self.images1)

    def __len__(self):
        """Return the total number of sample pairs in the dataset"""
        return self.num_samples

    def __getitem__(self, index):
        """Load and preprocess a paired PETCT ,PETMR and MRI image at the given index"""
        img1_name = os.path.join(self.images_folder1, self.images1[index])
        image1 = nib.load(img1_name).get_fdata()

        img2_name = os.path.join(self.images_folder2, self.images2[index])
        image2 = nib.load(img2_name).get_fdata()

        img_MRI_name = os.path.join(self.images_folder_MRI, self.images_MRI[index])
        image_MRI = nib.load(img_MRI_name).get_fdata()

        image1 = torch.tensor(image1, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        image2 = torch.tensor(image2, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        image_MRI = torch.tensor(image_MRI, dtype=torch.float32).unsqueeze(0)

        return image1, image2, image_MRI

def train(config: str):
    """Main training function for PETMR Correction with Anaotomical Guided

    Args:
        config: Path to the YAML configuration file
    """
    # Load configuration from YAML file
    config = yaml.load(open(config, 'rb'), Loader=yaml.FullLoader)

    # Extract CBAMUNet configuration from config with default values
    in_channels = config.get('in_channels', 1)
    out_channels = config.get('out_channels', 1)
    base_channels = config.get('base_channels', 64)

    # Extract MRI_Encoder configuration from config with default values
    in_channels_me = config.get('in_channels_me', 1)
    out_channels_me = config.get('out_channels_me', 1)
    img_size = config.get('img_size', 128)
    patch_size = config.get('patch_size', 4)
    dim = config.get('dim', 512)
    depth = config.get('depth', 8)
    num_heads = config.get('num_heads', 8)
    mlp_dim = config.get('mlp_dim', 2048)

    # Extract training parameters from config with default values
    epochs = config.get('epochs', 200)
    warm_epochs = config.get('warm_epochs', 10)
    batch_size = config.get('batch_size', 32)
    loss_type = config.get('loss_type', "mse+lpips")
    lpips_weight = config.get('lpips_weight', 0.1)
    net_type = config.get('net_type', 'vgg')
    batch_print_interval = config.get('batch_print_interval', 50)
    checkpoint_save_interval = config.get('checkpoint_save_interval', 5)
    save_path = config.get('save_path', './checkpoints')
    device = config.get('device', 'cuda')

    # Print model configuration
    print('CBAMUNet config:')
    print(f'in_channels: {in_channels}  out_channels: {out_channels} base_channels: {base_channels}')

    print('MRI_Encoder config:')
    print(f'in_channels: {in_channels_me}  out_channels: {out_channels_me}')
    print(f'img_size: {img_size}  patch_size: {patch_size}')
    print(f'dim: {dim}  depth: {depth}  num_heads: {num_heads}  mlp_dim: {mlp_dim}')

    # Print training parameters
    print('Training config:')
    print(f'epochs: {epochs}  warm_epochs: {warm_epochs}')
    print(f'batch_size: {batch_size}')
    print(f'loss_type: {loss_type},')
    print(f'lpips_weight: {lpips_weight}, net_type: {net_type}')
    print(f'batch_print_interval: {batch_print_interval}')
    print(f'checkpoint_save_interval: {checkpoint_save_interval}')
    print(f'save_path: {save_path}')
    print(f'device: {device}')

    # Dataset loading
    image_folder_PETCT = "Your PETCT path"  # PETCT image folder
    image_folder_PETMR = "Your PETMR path"  # PETMR image folder
    image_folder_MRI = "Your MRI path"  # MRI image folder

    dataset = PETMR_Correction_Dataset(images_folder1=image_folder_PETCT, images_folder2=image_folder_PETMR, images_folder_MRI=image_folder_MRI)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # CBAMUNet Initialization
    model = CBAMUNet_NPJ(in_channels, out_channels, base_channels=base_channels).to(device)
    model.train()

    # Load pre-trained MRI_Encoder weights
    MRI_Encoder = ViTEncoder(img_size, patch_size, dim, depth, num_heads, mlp_dim, in_channels).to(device)
    checkpoint = torch.load('Your weight path')
    MRI_Encoder.load_state_dict(checkpoint['model'])
    # Freeze MRI_Encoder parameters
    for param in MRI_Encoder.parameters():
        param.requires_grad = False
    MRI_Encoder.eval()

    # Optimizer loading
    optimizer = Adam([{'params': model.parameters()}], lr=1e-4, eps=1e-8, amsgrad=True)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=epochs - warm_epochs,
        eta_min=1e-5,
    )
    # Loss function
    criterion = Loss_Zoo(lpips_weight=lpips_weight, net_type=net_type).to(device)

    # Create folders in advance
    os.makedirs(save_path, exist_ok=True)

    # Training loop
    for epoch in range(epochs):
        clear_memory()
        print('---------------------------------------------------')
        start_time = time.time()

        for batch, (images1, images2, images_MRI) in enumerate(dataloader):
            optimizer.zero_grad()

            PETCT = images1.to(device)
            PETMR = images2.to(device)
            MRI = images_MRI.to(device)

            # Forward pass through models
            with torch.no_grad():
                cond = MRI_Encoder(MRI)

            pred_PETCT = model(PETMR, cond)

            # Calculate loss
            loss = criterion.reconstruction_loss(pred_PETCT, PETCT, loss_type="mse+lpips")

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            end_time1 = time.time()

            # Print training progress at specified intervals
            if batch % batch_print_interval == 0:
                print(f'[Epoch {epoch}] [batch {batch}] loss: {loss.item():.5f} Time: {end_time1 - start_time:.2f} seconds')

        # Update learning rate after warmup period
        if epoch >= warm_epochs:
            scheduler.step()

        # Save model checkpoint at specified intervals
        if (epoch > 0 and epoch % checkpoint_save_interval == 0) or epoch == epochs - 1 or epoch == 0:
            # Save first epoch for quick testing, can be removed
            print(f'Saving model {epoch} to {save_path}...')
            save_dict = dict(model=model.state_dict())
            torch.save(save_dict,os.path.join(save_path, f'CBAMUNet_{epoch}.pth'))

if __name__ == '__main__':
    train(config="./config/train_CBAMUNet.yaml")

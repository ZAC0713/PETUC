import os
import time
import yaml
import torch
import nibabel as nib
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from model.ViTAE import ViTEncoder, ViTAE
from Loss import Loss_Zoo


def clear_memory():
    """Clear GPU memory cache to prevent memory leaks"""
    torch.cuda.empty_cache()

class CT_MRI_Alignment_Dataset(Dataset):
    """Custom Dataset for loading paired MRI and CT images in NIfTI format"""

    def __init__(self, images_folder1, images_folder2):
        self.images_folder1 = images_folder1
        self.images_folder2 = images_folder2

        # List all NIfTI files in both folders
        self.images1 = sorted(
            [img for img in os.listdir(images_folder1) if img.endswith(".nii") or img.endswith(".nii.gz")])
        self.images2 = sorted(
            [img for img in os.listdir(images_folder2) if img.endswith(".nii") or img.endswith(".nii.gz")])

        # Ensure both folders have the same number of images
        if len(self.images1) != len(self.images2):
            raise ValueError("Folders must contain the same number of images.")

        self.num_samples = len(self.images1)

    def __len__(self):
        """Return the total number of sample pairs in the dataset"""
        return self.num_samples

    def __getitem__(self, index):
        """Load and preprocess a paired MRI and CT image at the given index"""
        img1_name = os.path.join(self.images_folder1, self.images1[index])
        image1 = nib.load(img1_name).get_fdata()

        img2_name = os.path.join(self.images_folder2, self.images2[index])
        image2 = nib.load(img2_name).get_fdata()

        image1 = torch.tensor(image1, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        image2 = torch.tensor(image2, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        return image1, image2


def train(config: str):
    """Main training function for MRI-CT feature alignment with contrastive learning

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
    margin = config.get('margin', 1.0)
    epochs = config.get('epochs', 100)
    warm_epochs = config.get('warm_epochs', 10)
    batch_size = config.get('batch_size', 8)
    batch_print_interval = config.get('batch_print_interval', 50)
    checkpoint_save_interval = config.get('checkpoint_save_interval', 3)
    save_path = config.get('save_path', './checkpoints')
    device = config.get('device', 'cuda')

    # Print model configuration
    print('Model config:')
    print(f'in_channels: {in_channels}  out_channels: {out_channels}')
    print(f'img_size: {img_size}  patch_size: {patch_size}')
    print(f'dim: {dim}  depth: {depth}  num_heads: {num_heads}  mlp_dim: {mlp_dim}')

    # Print training parameters
    print('Training config:')
    print(f'margin: {margin}')
    print(f'epochs: {epochs}  warm_epochs: {warm_epochs}')
    print(f'batch_size: {batch_size}')
    print(f'batch_print_interval: {batch_print_interval}')
    print(f'checkpoint_save_interval: {checkpoint_save_interval}')
    print(f'save_path: {save_path}')
    print(f'device: {device}')

    # Dataset loading
    image_folder_MRI = "Your MRI path"  # MRI image folder
    image_folder_CT = "Your CT path"  # CT image folder

    dataset = CT_MRI_Alignment_Dataset(images_folder1=image_folder_MRI, images_folder2=image_folder_CT)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model Initialization
    model = ViTEncoder(img_size, patch_size, dim, depth, num_heads, mlp_dim, in_channels)
    model.to(device)
    model.train()

    # Load pre-trained AE weights
    AE_frozen = ViTAE().to(device)
    checkpoint = torch.load('Your weight path')
    AE_frozen.load_state_dict(checkpoint['model'])

    # Freeze AE parameters
    for param in AE_frozen.parameters():
        param.requires_grad = False
    AE_frozen.eval()

    # Optimizer loading
    optimizer = Adam([{'params': model.parameters()}], lr=1e-3, eps=1e-8, amsgrad=True)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=epochs - warm_epochs,
        eta_min=1e-4,
    )

    # Loss function
    criterion = Loss_Zoo().to(device)

    # Create folders in advance
    os.makedirs(save_path, exist_ok=True)

    # Training loop
    for epoch in range(epochs):
        clear_memory()
        print('---------------------------------------------------')
        start_time = time.time()

        for batch, (images_MRI, images_CT) in enumerate(dataloader):
            optimizer.zero_grad()

            MRI = images_MRI.to(device)
            CT = images_CT.to(device)

            # Forward pass through models
            MRI_latent = model(MRI)
            CT_latent = AE_frozen.encoder(CT)

            # Calculate contrastive loss
            loss = criterion.contrastive_loss(MRI_latent, CT_latent, margin=margin)

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
            # Save first epoch for quick testing, can be removed
            print(f'Saving model {epoch} to {save_path}...')
            save_dict = dict(model=model.state_dict())
            torch.save(save_dict,
                       os.path.join(save_path, f'MRI_ViT_Encoder_{epoch}.pth'))


if __name__ == '__main__':
    # Start training with the specified configuration file
    train(config="./config/train_MRI-CT_align.yaml")

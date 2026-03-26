import os
import torch
import yaml
import nibabel as nib
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from torch.utils.data import Dataset, DataLoader
from model.ViTAE import ViTAE

def clear_memory():
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

def infer(config, checkpoint_path, device='cuda'):
    """Inference and evaluation function for the ViTAE model.

    Loads a trained ViTAE checkpoint, reconstructs CT images, and computes
    average PSNR and SSIM over the entire dataset.

    Args:
        config: Path to the YAML configuration file
        checkpoint_path: Path to the trained model checkpoint (.pth file)
        device: Target device for inference, either 'cuda' or 'cpu'

    Returns:
        PSNR_mean: Mean PSNR across all test samples
        SSIM_mean: Mean SSIM across all test samples
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

    # Print model configuration
    print('Model config:')
    print(f'in_channels: {in_channels}  out_channels: {out_channels}')
    print(f'img_size: {img_size}  patch_size: {patch_size}')
    print(f'dim: {dim}  depth: {depth}  num_heads: {num_heads}  mlp_dim: {mlp_dim}')

    # Initialize Vision Transformer AE model
    model = ViTAE(
        in_channels=in_channels, out_channels=out_channels, img_size=img_size, patch_size=patch_size, dim=dim,
        depth=depth, num_heads=num_heads, mlp_dim=mlp_dim)
    model.to(device)  # Move model to specified device (CPU/GPU)
    model.eval()

    # Load trained weights from checkpoint
    print(f'The path of the model weights loaded this time : {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])

    image_folder_CT = "Your CT path"  # CT image folder

    dataset = CT_Dataset(images_folder=image_folder_CT)
    # batch_size=1 and shuffle=False to evaluate samples sequentially
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        PSNR = 0
        SSIM = 0
        count = 0

        for image_CT in dataloader:

            CT = image_CT.to(device)
            CT_rcon = model(CT)

            count += 1

            # Squeeze batch and channel dims to get 2D numpy arrays for metric computation
            PSNR += psnr(CT.squeeze(0).squeeze(0).cpu().numpy(), CT_rcon.squeeze(0).squeeze(0).cpu().numpy())
            SSIM += ssim(CT.squeeze(0).squeeze(0).cpu().numpy(), CT_rcon.squeeze(0).squeeze(0).cpu().numpy(),
                         win_size=5, gaussian_weights=True, multichannel=False, data_range=1.0,
                         K1=0.01, K2=0.03, sigma=0.6)

    # Compute mean metrics over all samples
    PSNR_mean = PSNR / count
    SSIM_mean = SSIM / count
    print(f'PSNR:{PSNR_mean},SSIM:{SSIM_mean}')

    return PSNR_mean, SSIM_mean


if __name__ == '__main__':
    PSNR, SSIM = infer(
        config="./config/train_CTAE.yaml",
        checkpoint_path='Your weight path',
        device='cuda'
    )

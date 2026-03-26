import os
import torch
import yaml
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from model.CBAMUNET_MICCAI import CBAMUNet_MICCAI
from model.CBAMUNET_NPJ import CBAMUNet_NPJ
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from model.ViTAE import ViTEncoder

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

def infer(config, checkpoint_path, device='cuda'):
    """Inference and evaluation function for the anatomically guided PETMR correction model.

    Loads a trained CBAMUNet checkpoint alongside a frozen ViTEncoder, performs
    PETCT reconstruction from PETMR with MRI-derived anatomical conditioning, and
    computes mean PSNR and SSIM against ground-truth PETCT over the entire dataset.

    Args:
        config: Path to the YAML configuration file
        checkpoint_path: Path to the trained CBAMUNet checkpoint (.pth file)
        device: Target device for inference, either 'cuda' or 'cpu'

    Returns:
        PSNR_mean: Mean PSNR across all test samples
        SSIM_mean: Mean SSIM across all test samples
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
    loss_type = config.get('loss_type', "mse+lpips")
    lpips_weight = config.get('lpips_weight', 0.1)
    net_type = config.get('net_type', 'vgg')

    # Print model configuration
    print('CBAMUNet config:')
    print(f'in_channels: {in_channels}  out_channels: {out_channels} base_channels: {base_channels}')

    print('MRI_Encoder config:')
    print(f'in_channels: {in_channels_me}  out_channels: {out_channels_me}')
    print(f'img_size: {img_size}  patch_size: {patch_size}')
    print(f'dim: {dim}  depth: {depth}  num_heads: {num_heads}  mlp_dim: {mlp_dim}')

    # Print Lpips Loss parameters
    print('Training config:')
    print(f'loss_type: {loss_type},')
    print(f'lpips_weight: {lpips_weight}, net_type: {net_type}')

    # Dataset loading
    image_folder_PETCT = "Your PETCT path"  # PETCT image folder
    image_folder_PETMR = "Your PETMR path"  # PETMR image folder
    image_folder_MRI = "Your MRI path"  # MRI image folder

    dataset = PETMR_Correction_Dataset(images_folder1=image_folder_PETCT, images_folder2=image_folder_PETMR, images_folder_MRI=image_folder_MRI)
    # batch_size=1 and shuffle=False to evaluate samples sequentially
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Load trained CBAMUNet weights from checkpoint
    print(f'The path of the model weights loaded this time : {checkpoint_path}')
    model = CBAMUNet_NPJ(in_channels, out_channels, base_channels=base_channels).to(device)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # Load pre-trained MRI_Encoder weights; the encoder is frozen and used solely for anatomical feature extraction
    MRI_Encoder = ViTEncoder(img_size, patch_size, dim, depth, num_heads, mlp_dim, in_channels).to(device)
    checkpoint = torch.load('Your weight path')
    MRI_Encoder.load_state_dict(checkpoint['model'])
    MRI_Encoder.eval()

    with torch.no_grad():
        PSNR = 0
        SSIM = 0
        count = 0

        for images1, images2, images_MRI in dataloader:

            PETCT = images1.to(device)
            PETMR = images2.to(device)
            MRI = images_MRI.to(device)

            # Extract anatomical conditioning features from MRI via the frozen encoder
            cond = MRI_Encoder(MRI)

            # Forward pass through CBAMUNet with anatomical conditioning
            pred_PETCT = model(PETMR, cond)

            # Squeeze batch and channel dims to get 2D numpy arrays for metric computation
            PSNR += psnr(PETCT.squeeze(0).squeeze(0).cpu().numpy(), pred_PETCT.squeeze(0).squeeze(0).cpu().numpy())
            SSIM += ssim(PETCT.squeeze(0).squeeze(0).cpu().numpy(), pred_PETCT.squeeze(0).squeeze(0).cpu().numpy(),
                         win_size=5, gaussian_weights=True, multichannel=False, data_range=1.0,
                         K1=0.01, K2=0.03, sigma=0.6)

        # Compute mean metrics over all samples
        PSNR_mean = PSNR / count
        SSIM_mean = SSIM / count
        print(f'PSNR:{PSNR_mean},SSIM:{SSIM_mean}')

        return PSNR_mean, SSIM_mean


if __name__ == '__main__':
    PSNR, SSIM = infer(
        config="./config/train_CBAMUNet.yaml",
        checkpoint_path='Your weight path',
        device='cuda'
    )

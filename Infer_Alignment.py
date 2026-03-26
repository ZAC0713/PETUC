import os
import torch
import yaml
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from model.ViTAE import ViTAE, ViTEncoder
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

def infer(config, checkpoint_path, device='cuda'):
    """Inference and evaluation function for the ViTEncoder model.Loads a trainedViTEncoder checkpoint alongside a frozen ViTAE, extracts
    latent representations from paired MRI and CT images, and evaluates the
    mean contrastive loss over the entire dataset.

    Args:
        config: Path to the YAML configuration file
        checkpoint_path: Path to the trained ViTEncoder checkpoint (.pth file)
        device: Target device for inference, either 'cuda' or 'cpu'

    Returns:
        Loss_mean: Mean contrastive loss across all test sample pairs
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

    # Dataset loading
    image_folder_MRI = "Your MRI path"  # MRI image folder
    image_folder_CT = "Your CT path"  # CT image folder

    dataset = CT_MRI_Alignment_Dataset(images_folder1=image_folder_MRI, images_folder2=image_folder_CT)
    # batch_size=1 and shuffle=False to evaluate samples sequentially
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Load trained ViTEncoder weights from checkpoint
    print(f'The path of the model weights loaded this time : {checkpoint_path}')
    model = ViTEncoder(img_size, patch_size, dim, depth, num_heads, mlp_dim, in_channels).to(device)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # Load pre-trained AE weights; parameters are frozen since only the encoder is used for feature extraction
    AE_frozen = ViTAE().to(device)
    checkpoint = torch.load('Your weight path')
    AE_frozen.load_state_dict(checkpoint['model'])
    AE_frozen.eval()

    # Loss function
    criterion = Loss_Zoo().to(device)

    with torch.no_grad():
        count = 0
        Loss = 0

        for image_MRI, image_CT in dataloader:
            MRI = image_MRI.to(device)
            CT = image_CT.to(device)

            # Extract latent representation of MRI viaViTEncoder
            mri_latent = model(MRI)
            # Extract latent representation of CT via the frozen AE encoder
            ct_latent = AE_frozen.Encoder(CT)

            # Calculate contrastive loss between MRI and CT latent spaces
            Loss += criterion.contrastive_loss(mri_latent, ct_latent, margin=1.0)
            count += 1  # count自增

    # Compute contrastive loss over all samples
    Loss_mean = Loss / count
    print(f'loss: {Loss_mean}')

    return Loss_mean


if __name__ == '__main__':
    loss = infer(
        config="./config/train_CTAE.yaml",
        checkpoint_path='Your weight path',
        device='cuda'
    )

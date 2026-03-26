import torch
import torch.nn as nn
import torch.nn.functional as F
from lpips_pytorch.modules.lpips import LPIPS

class Loss_Zoo(nn.Module):
    def __init__(self, lpips_weight: float = 0.1, net_type: str = 'vgg'):
        super(Loss_Zoo, self).__init__()
        self.lpips_weight = lpips_weight
        self.net_lpips = LPIPS(net_type=net_type, version='0.1')
        self.net_lpips.requires_grad_(False)

    def reconstruction_loss(self, pred, GT, loss_type: str = "mse+lpips"):
        batch_size = pred.shape[0]

        if loss_type == "l1":
            return F.l1_loss(pred, GT)
        elif loss_type == "mse":
            return F.mse_loss(pred, GT)
        elif loss_type == "mse+lpips":
            mse = F.mse_loss(pred, GT)
            lpips = self.net_lpips(pred, GT).squeeze()/batch_size
            return mse + self.lpips_weight * lpips
        else:
            raise ValueError(f"Unsupported reconstruction loss: {loss_type}")

    # Define contrastive loss
    def contrastive_loss(self,mri_features, ct_features, margin=1.0):
        """Calculate contrastive loss between MRI and CT features

        Args:
            mri_features: MRI latent features with shape (B, num_patches, embed_dim)
            ct_features: CT latent features with shape (B, num_patches, embed_dim)
            margin: Margin for negative sample loss calculation

        Returns:
            Contrastive loss value
        """
        # Ensure input features have shape (B, num_patches, embed_dim)
        B, num_patches, embed_dim = mri_features.shape

        # Calculate positive sample distance
        pos_distance = torch.norm(mri_features - ct_features, dim=-1)  # Shape: (B, num_patches)
        pos_loss = 0.5 * pos_distance ** 2
        pos_loss = pos_loss.mean()  # Average positive sample loss

        # Calculate negative sample distance
        mri_expanded = mri_features.unsqueeze(2)  # Shape: (B, num_patches, 1, embed_dim)
        ct_expanded = ct_features.unsqueeze(1)  # Shape: (B, 1, num_patches, embed_dim)
        neg_distance = torch.norm(mri_expanded - ct_expanded, dim=-1)  # Shape: (B, num_patches, num_patches)

        # Generate mask matrix to exclude i=j cases
        eye_mask = 1 - torch.eye(num_patches, device=mri_features.device).unsqueeze(
            0)  # Shape: (1, num_patches, num_patches)

        # Calculate negative sample loss and apply mask
        neg_loss = 0.5 * torch.clamp(margin - neg_distance, min=0.0) ** 2
        neg_loss = neg_loss * eye_mask  # Mask i=j positions

        # Calculate average of valid negative sample losses
        valid_elements = eye_mask.sum() * B  # Total number of valid elements
        neg_loss = neg_loss.sum() / valid_elements.clamp(min=1e-6)  # Avoid division by zero

        # Total loss
        total_loss = pos_loss + neg_loss
        return total_loss

# import necessary libraries
import numpy as np
import SimpleITK as sitk
import torchio as tio
import torch

from pathlib import Path
from torch.utils.data import Dataset
from torch import Tensor

def sitk_to_numpy(filename):
    image = sitk.ReadImage(filename)
    spacing = image.GetSpacing()
    offset = image.GetOrigin()

    # convert to (z, y, x)
    spacing = np.array(spacing)[::-1]
    offset = np.array(offset)[::-1]

    image = sitk.GetArrayFromImage(image)
    return image, spacing, offset

def get_masks_per_channel(filename, labels: list = None) -> tuple[Tensor, Tensor]:
    """
    Opens `filename` as nii.gz and returns the labelmap as (H, W)
    and separated masks by channel as (C, H, W).
    """
    labelmap, _, _ = sitk_to_numpy(filename)

    if labels is None:
        labels = torch.arange(labelmap.max())
        
    masks = torch.stack([ torch.tensor(labelmap.squeeze() == idx) for idx in labels])
    labelmap = torch.tensor(labelmap)

    return labelmap, masks

def get_scan_from_nifti(filename) -> Tensor:
    """Opens `filename` and returns the first slice of image as tensor (H, W)"""
    image, _, _ = sitk_to_numpy(filename)
    image = torch.tensor(image[0], dtype=torch.float) # get first slice always 

    return image

def crop_used_region(image: Tensor) -> tuple[int, int, int, int]:
    """
    Crop all used region in an image to a minimal bounding box.
    
    Returns left, right, up, down coordinates of rectangle.
    """
    H, W = image.shape
    temp = (image != 0).to(torch.float)
    
    left_margins = torch.argmax(temp, 1)
    left = torch.min(left_margins[left_margins>0])

    right_margins = torch.argmax(torch.flip(temp, [1]), 1)
    right = torch.min(right_margins[right_margins>0])

    up_margins = torch.argmax(temp, 0)
    up = torch.min(up_margins[up_margins>0])

    down_margins = torch.argmax(torch.flip(temp, [0]), 0)
    down = torch.min(down_margins[down_margins>0])
    
    return left, W - right, up, H - down

class ISLES2024Dataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.root = Path(config.root)

        # get ids
        self.id_list = self.root.glob('CTATr/sub-*_0000.nii.gz')
        self.id_list = [p.name.split('_')[0] for p in self.id_list]
        self.id_list.sort()

    def __len__(self):
        return len(self.id_list)
    
    def __getitem__(self, idx):
        id_ = self.id_list[idx]

        # get paths
        fluor_path = self.root / 'FTr' / id_
        fluor_msk_path = self.root / 'F_maskTr' / id_
        fluor_metadata_path = self.root / 'F_metadataTr' / id_
        cta_path = self.root / 'CTATr' / f'{id_}_0000.nii.gz'
        cta_msk_path = self.root / 'CTA_skullTr' / f'{id_}.nii.gz'

        return {"Fluor": fluor_path, "Fluor_mask": fluor_msk_path, "Fluor_metadata": fluor_metadata_path,
                "CTA": cta_path, "CTA_mask": cta_msk_path}

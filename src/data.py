# import necessary libraries
import numpy as np
import SimpleITK as sitk
import torchio as tio
import torch

from pathlib import Path
from torch.utils.data import Dataset


def sitk_to_numpy(filename):
    image = sitk.ReadImage(filename)
    spacing = image.GetSpacing()
    offset = image.GetOrigin()

    # convert to (z, y, x)
    spacing = np.array(spacing)[::-1]
    offset = np.array(offset)[::-1]

    image = sitk.GetArrayFromImage(image)
    return image, spacing, offset

def get_masks_per_channel(filename, labels: list = None):
    labelmap, _, _ = sitk_to_numpy(filename)

    if labels is None:
        labels = torch.arange(labelmap.max())
        
    masks = torch.stack([ torch.tensor(labelmap.squeeze() == idx) for idx in labels])

    return labelmap, masks

def get_scan_from_nifti(filename):
    image, _, _ = sitk_to_numpy(filename)
    image = torch.tensor(image[0], dtype=torch.float) # get first slice always 

    return image


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

# import necessary libraries
import cv2
import hydra
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import shutil
import time
import torch
import torch.nn as nn
import torchio as tio
from torchvision.transforms import v2
from torch import Tensor

from diffdrr.data import read
from diffdrr.drr import convert, DRR
from diffdrr.metrics import *
from functools import wraps
from monai.losses.dice import *
from pathlib import Path
from scipy.ndimage import label
from skimage.transform import resize
from torch.optim import *
from torch.optim.lr_scheduler import *
from .data import get_masks_per_channel, get_scan_from_nifti, crop_used_region, sitk_to_numpy
from PIL import Image

# try to import bilateral_filter_layer, fall back to cv2 if not available
try:
    from bilateral_filter_layer import BilateralFilter3d
    HAS_BILATERAL_LAYER = True
except ImportError:
    HAS_BILATERAL_LAYER = False


def time_it(func):
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time

        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        logging.info(f"{func.__name__} completed in {int(hours):02d}:{int(minutes):02d}:{seconds:06.3f}")
        return result
    return wrapper


class FluoresenceReg(nn.Module):
    def __init__(self, config, id_dict):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # initialize bilateral filter layer if available
        if HAS_BILATERAL_LAYER:
            self.layer = BilateralFilter3d(*config.sigmas, use_gpu=torch.cuda.is_available()).to(self.device)
        else:
            self.layer = None

        # create output dir
        base_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        id_ = Path(id_dict['CTA']).stem.split('_')[0]
        self.output_dir = base_dir / id_ / config.timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir = self.output_dir / 'images'
        self.image_dir.mkdir(exist_ok=True)

        # load data
        paths = self._get_fluor_paths(id_dict)
        cta = read(id_dict['CTA'], id_dict['CTA_mask'], labels=range(74))
        fluor, labelmap, fluor_mask = self._get_fluor_images(paths)
        self.common_masks_index = self._get_common_channels(cta.mask.data, labelmap)
        self.fluor, self.fluor_mask = self._prepare_fluor_data(fluor, fluor_mask, labelmap, self.common_masks_index) 
        self.dists = {}
        self.drrs = {}
        self.params = nn.ParameterDict()

        with open(paths[2]) as f:
            metadata = json.load(f)

        self.dists = metadata['DistanceSourceToDetector'] - config.detector_dist # what is detector_dist?
        self.drrs = DRR(cta, sdd=metadata['DistanceSourceToDetector'],
                            height=config.detector_size[0], width=config.detector_size[1], delx=config.detector_spacing[0],
                            stop_gradients_through_grid_sample=True).to(self.device)
        
        mean_mask_center = self._get_mean_center(cta.mask, self.common_masks_index)
        self.params = self._initialize_params(mean_mask_center)

        # loss
        self.criterion = eval(config.criterion_img)(**(config.criterion_img_kwargs or {}))
        self.dice_loss = eval(config.criterion_msk)()

        # setup optimizer
        self.current_iter = 0
        self.optimizer = eval(config.optimizer)(self.parameters(), **(self.config.optimizer_kwargs or {}))
        self.scheduler = eval(config.scheduler)(self.optimizer, **(config.scheduler_kwargs or {}))
        
    def _get_fluor_paths(self, id_dict):
        img_paths = str(id_dict['Fluor']) + '.nii.gz'
        msk_paths = str(id_dict['Fluor_mask']) + '.nii.gz'
        meta_paths = str(id_dict['Fluor_metadata']) + '.json'
        return img_paths, msk_paths, meta_paths

    def _get_common_channels(self, input: Tensor, target: Tensor) -> Tensor:
        input_channels = torch.bincount(input.ravel()).nonzero(as_tuple=True)[0].int()[1:]
        target_channels = torch.bincount(target.ravel()).nonzero(as_tuple=True)[0].int()[1:]

        channels = list(set(target_channels.tolist()) & set(input_channels.tolist()))
        return torch.tensor(channels, device=self.device, dtype=int)

    def _get_fluor_images(self, paths: list) -> tuple[Tensor, Tensor, Tensor]:
        # Get img and masks as numpy
        img = torch.flip(get_scan_from_nifti(paths[0]).to(device=self.device), [0, 1]) # (H, W)
        img = img.max() - img # invert colors

        labelmap, msk = get_masks_per_channel(paths[1], labels=range(73)) # TODO: Autodetect amount of labels here
        msk = torch.flip(msk.to(device=self.device), [1, 2]) # (C, H, W)

        return img, labelmap, msk

    def _prepare_fluor_data(self, img: Tensor, msk: Tensor, labelmap: Tensor, channels: Tensor) -> tuple[Tensor, Tensor]:
        imgs, msks = None, None
        
        # Crop msk and image to only contain elements in CTA
        used_msk = torch.sum(msk[channels], 0) # Flatten all used mask channel to one tensor
        left, right, up, down = crop_used_region(used_msk)
        img = img[up:down, left:right]
        msk = msk[:, up:down, left:right]

        # calculate metrics over valid region
        q = torch.Tensor([0.25, 0.75]).to(device=self.device)
        vmin, vmax = torch.quantile(img, q)
        if vmin == vmax:
            vmax += 40

        # get largest mask component
        labeled, _ = label(labelmap)
        # component_sizes = np.bincount(labeled.ravel())[1:]
        # largest_component = np.argmax(component_sizes) + 1
        # msk = (labeled == largest_component).astype(float)

        # norm
        img = torch.clip(img, vmin, vmax)
        img -= torch.min(img)

        # reshape
        if img.shape != self.config.detector_size:
            img = v2.Resize(self.config.detector_size, antialias=True)(img[None]).squeeze()
            msk = v2.Resize(self.config.detector_size, antialias=True)(msk)

        # filter - use bilateral_filter_layer if available, otherwise cv2
        if HAS_BILATERAL_LAYER and self.layer is not None:
            # TODO: Adapt this for new imgs
            raise NotImplementedError
            img = torch.tensor(img, device=self.device, dtype=torch.float32)[None, None, None]
            msk_tensor = torch.tensor(msk, device=self.device, dtype=torch.float32)[None, None]
            with torch.no_grad():
                img = self.layer(img)[:, :, 0] * msk_tensor
            imgs = img
            msks = msk_tensor
        else:
            img = cv2.bilateralFilter(img.cpu().numpy().astype(np.float32), *self.config.sigmas)
            # img *= msk # TODO: Uncomment this, and use `labeled` to cut out background
            imgs = torch.tensor(img, device=self.device, dtype=torch.float32)[None, None]
            msks = msk[None]

        return imgs, msks

    def _get_mean_center(self, mask: tio.LabelMap, channels: Tensor) -> Tensor:
        """
        Get mean center of nonzero elements in mask.
        Returns a tensor with [x, y, z]
        """
        spacing = torch.tensor(list(mask.spacing), device=self.device)
        image = mask.data

        image = image.to(device=self.device)
        _, W, H, D = image.shape
        image = torch.stack(
            [image == i for i in channels]
        ).sum(dim=0) # Collect all mask of channels into one

        # Flip tensor first as images loaded through ScalarImage has its origin at bottom
        nonzero_indices = torch.flip(image, [1, 2, 3]).nonzero().float()

        center_mean = (nonzero_indices.mean(dim=0)[1:] - torch.tensor([W, H, D], device=self.device) / 2) # Center the mean as origin is at center of volume
        return center_mean * spacing
    
    def _initialize_params(self, tra: Tensor):
        rot_init = [[0, 0, 0]]    
        tra_init = [[-tra[0], self.dists, -tra[2]]]

        rot_init = torch.tensor(rot_init, device=self.device, dtype=torch.float32) / 180 * torch.pi
        tra_init = torch.tensor(tra_init, device=self.device, dtype=torch.float32) / self.config.multiplier

        return nn.ParameterList([nn.Parameter(rot_init), nn.Parameter(tra_init)])

    def _extract_parameters(self, to_cpu=True, to_list=True):
        """Extract rotation and translation parameters as lists."""
        rot = self.params[0]
        tra = self.params[1] * self.config.multiplier

        if to_cpu:
            rot, tra = rot.detach().cpu(), tra.detach().cpu()
        if to_list:
            rot, tra = rot.tolist(), tra.tolist()
        return rot, tra
    
    def _compute_loss(self):
        rot, tra = self._extract_parameters(False, False)
        estimate = self.drrs(rot, tra, parameterization='euler_angles',
                                convention='ZYX', mask_to_channels=True)
        # ncc loss
        ncc_loss = -self.criterion(self.fluor, estimate.sum(dim=1, keepdim=True))

        # dice loss
        estimate_pred_common = estimate[:, self.common_masks_index]
        fluor_mask_common = self.fluor_mask[:, self.common_masks_index]
        estimate_pred = torch.sigmoid(estimate_pred_common) # this should pick the second channel, bcs of mask_to_channels=True
        dsc_loss = self.dice_loss(estimate_pred_common, fluor_mask_common)
        return ncc_loss, dsc_loss

    def _plot(self, ncc_losses, dsc_losses):
        """Save comparison plot of ground truth DSA and current estimates."""
        fig, axes = plt.subplots(2, 1, figsize=(12, 5))

        # compute estimate
        rot, tra = self._extract_parameters(to_cpu=False, to_list=False)
        estimate = self.drrs(rot, tra, parameterization='euler_angles',
                                convention='ZYX', mask_to_channels=True)
        estimate = estimate.sum(dim=1, keepdim=True)

        # plot ground truth DSA
        dsa_np = self.fluor.squeeze().detach().cpu().numpy()
        axes[0].imshow(dsa_np, cmap='gray')
        axes[0].set_title(f'fluor')
        axes[0].axis('off')

        # plot current estimate
        estimate_np = estimate.squeeze().detach().cpu().numpy()
        axes[1].imshow(estimate_np, cmap='gray')
        axes[1].set_title(f'Estimate (NCC: {ncc_losses.item():.4f}, DSC: {dsc_losses.item():.4f})')
        axes[1].axis('off')

        plt.tight_layout()
        plt.savefig(self.image_dir / f'iteration_{self.current_iter:04d}.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _create_video_from_images(self, fps=30):
        """Create MP4 video from saved iteration images and delete the images."""
        # get all image files sorted by iteration number
        image_files = sorted(self.image_dir.glob('iteration_*.png'))

        # read first image to get dimensions
        first_img = cv2.imread(str(image_files[0]))
        height, width, _ = first_img.shape

        # create video writer
        video_path = self.output_dir / 'optimization_progress.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

        # write all images to video
        for img_file in image_files:
            img = cv2.imread(str(img_file))
            video.write(img)
        video.release()

        # delete the image folder and all its contents
        shutil.rmtree(self.image_dir)
        logging.info(f"Video saved to {video_path}")

    @time_it
    def fit(self):
        ncc = []
        dsc = []
        params = {'rot': [], 'tra': []}
        
        # initial loss
        ncc_loss, dsc_loss = self._compute_loss()
        
        ncc.append(ncc_loss.item())
        dsc.append(dsc_loss.item())

        # track best
        ncc_best = ncc[0]
        dsc_best = dsc[0]
        rot_best = self._extract_parameters()[0]
        tra_best = self._extract_parameters()[1]

        for i in range(1, self.config.num_iter + 1):
            self.current_iter = i

            rot, tra = self._extract_parameters()
            params['rot'].append(rot)
            params['tra'].append(tra)

            if i == 1:
                logging.info(f"Initial parameters:")
                logging.info(f"    NCC initial: {ncc_best:.4f}")
                logging.info(f"    DSC initial: {dsc_best:.4f}")
                logging.info(f"    Rotation (alpha, beta, gamma): {rot_best}")
                logging.info(f"    Translation (bx, by, bz): {tra_best}")

            # optimization step
            self.optimizer.zero_grad()

            ncc_loss, dsc_loss = self._compute_loss()
            loss = 0.0
            loss += self.config.alpha * ncc_loss + (1 - self.config.alpha) * dsc_loss

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # plot
            self._plot(ncc_loss, dsc_loss)

            # log
            ncc.append(ncc_loss.item())
            dsc.append(dsc_loss.item())

            rot, tra = self._extract_parameters()
            params['rot'].append(rot)
            params['tra'].append(tra)

            if ncc_loss.item() < ncc_best:
                ncc_best = ncc_loss.item()
                dsc_best = dsc_loss.item()
                rot_best, tra_best = self._extract_parameters()
                logging.info(f"Iter {i}: New best NCC: {ncc_best:.4f}")
                logging.info(f"          DSC: {dsc_best:.4f}")

        # save best parameters to JSON
        best_params = {
            'ncc_best': ncc_best,
            'dsc_best': dsc_best,
            'rot_best': rot_best,
            'tra_best': tra_best
        }
        with open(self.output_dir / 'best_parameters.json', 'w') as f:
            json.dump(best_params, f, indent=2)

        # create video from images and delete PNGs
        self._create_video_from_images()

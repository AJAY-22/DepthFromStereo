import re
import torch
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def read_pfm(file):
    """Read a PFM file and return image data and scale."""
    with open(file, "rb") as f:
        header = f.readline().decode("utf-8").rstrip()
        if header == "PF":
            color = True
        elif header == "Pf":
            color = False
        else:
            raise Exception("Not a PFM file.")

        dims = f.readline().decode("utf-8").rstrip()
        while dims.startswith("#"):
            dims = f.readline().decode("utf-8").rstrip()
        width, height = map(int, dims.split())
        
        scale_line = f.readline().decode("utf-8").rstrip()
        scale = float(scale_line)
        endian = "<" if scale < 0 else ">"
        scale = abs(scale)
        
        num_channels = 3 if color else 1
        data = np.fromfile(f, endian + "f")
        expected_elems = width * height * num_channels
        if len(data) != expected_elems:
            raise Exception("Mismatch in data size, expected %d elements, got %d." % (expected_elems, len(data)))
        
        if color:
            data = np.reshape(data, (height, width, 3))
        else:
            data = np.reshape(data, (height, width))
        data = np.flipud(data)  # flip vertically
    return data, scale


def compute_mse(patch1, patch2):
    return np.mean((patch1 - patch2) ** 2)


def stereo_match_cuda(left_img, right_img, kernel_size=5, stride_x=1, stride_y=1, max_disp=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    half_k = kernel_size // 2
    disp_map = np.zeros_like(left_img)
    h, w, _ = left_img.shape
    for y in tqdm(range(half_k, h - half_k, stride_y)):
        for x in range(half_k, w - half_k, stride_x):
            best_mse = float('inf')
            best_disp = 0
            valid_disp = min(max_disp, x - half_k + 1)
            left_patch = left_img[y - half_k:y + half_k + 1, x - half_k:x + half_k + 1]
            candidate_patches = np.stack([
                right_img[y - half_k:y + half_k + 1, x - half_k - d:x + half_k + 1 - d]
                for d in range(valid_disp)
            ], axis=0)
            left_patch_stack = np.repeat(left_patch[None, ...], valid_disp, axis=0)
            candidate_patches = torch.tensor(candidate_patches, dtype=float).to(device)
            left_patch_stack = torch.tensor(left_patch_stack, dtype=float).to(device)
            mses = torch.mean((candidate_patches - left_patch_stack) ** 2, dim=(1, 2, 3))
            best_disp = int(torch.argmin(mses).cpu().numpy())
            # disp_map[y, x] = best_disp
            # disp_map[y - half_k:y + half_k + 1, x - half_k:x + half_k + 1] = best_disp
            # disp_map[y - half_k: y + half_k + 1, x] = best_disp
            # disp_map[y - half_k: y + half_k + 1, x - half_k: x + half_k + 1] = best_disp
            disp_map[y - stride_y//2: y + stride_y//2 + 1, x - stride_x//2: x + stride_x//2 + 1] = best_disp
    return disp_map

def stereo_match(left_img, right_img, kernel_size=5, stride_x=1, stride_y=1, max_disp=64):
    half_k = kernel_size // 2
    disp_map = np.zeros(left_img.shape[:2])
    h, w, _ = left_img.shape
    # cnt = 50
    for y in tqdm(range(half_k + 100, h - half_k, stride_y)):
        for x in range(half_k, w - half_k, stride_x):
            best_disp = float('inf')
            valid_disp = min(max_disp, x - half_k + 1)
            left_patch = left_img[y - half_k:y + half_k + 1, x - half_k:x + half_k + 1]
            candidate_patches = np.stack([
                right_img[y - half_k:y + half_k + 1, x - half_k - d:x + half_k + 1 - d]
                for d in range(valid_disp)
            ], axis=0)
            left_patch_stack = np.repeat(left_patch[None, ...], valid_disp, axis=0)
            mses = np.mean((candidate_patches - left_patch_stack) ** 2, axis=(1, 2, 3))
            best_disp = int(np.argmin(mses))
            # Sets only single pixel disparity
            # disp_map[y, x] = best_disp

            # Sets disparity for entire patch
            # disp_map[y - half_k:y + half_k + 1, x - half_k:x + half_k + 1] = best_disp
            
            # Sets disparity for entire column of a patch
            # disp_map[y - half_k: y + half_k + 1, x] = best_disp

            # Sets disparity for stride_x x stride_y patch 
            disp_map[y - stride_y//2: y + stride_y//2 + 1, x - stride_x//2: x + stride_x//2 + 1] = best_disp
        # cnt -= stride_y
        # if cnt == 0:
        #     break  
    return disp_map

def parse_calib(calib_path):
    """Extract calibration parameters from calib.txt.
       Returns:
         dict: A dictionary containing:
           - focal_length (float): from cam0 (in pixels)
           - baseline (float): in meters (converted from mm)
           - max_disp (int): from ndisp
           - vmin (float): from vmin (if available)
           - vmax (float): from vmax (if available)
    """
    focal_length = None
    baseline = None
    max_disp = 64  # fallback default if ndisp is not found
    vmin = None
    vmax = None

    with open(calib_path, 'r') as f:
        for line in f:
            # Parse cam0 to get focal length
            if line.startswith('cam0'):
                # Example line: cam0=[1758.23 0 953.34; 0 1758.23 552.29; 0 0 1]
                matches = re.findall(r"[\d\.]+", line)
                if matches:
                    focal_length = float(matches[0])
            if line.startswith('baseline'):
                try:
                    # baseline in calib.txt is in mm, convert to meters
                    baseline = float(line.strip().split('=')[1]) / 1000.
                except ValueError:
                    pass
            if line.startswith('ndisp'):
                try:
                    max_disp = int(line.strip().split('=')[1])
                except ValueError:
                    pass
            if line.startswith('vmin'):
                try:
                    vmin = float(line.strip().split('=')[1])
                except ValueError:
                    pass
            if line.startswith('vmax'):
                try:
                    vmax = float(line.strip().split('=')[1])
                except ValueError:
                    pass

    if focal_length is None or baseline is None:
        print("Warning: Unable to retrieve some calibration parameters from {}".format(calib_path))
    return {
        "focal_length": focal_length,
        "baseline": baseline,
        "max_disp": max_disp,
        "vmin": vmin,
        "vmax": vmax,
    }
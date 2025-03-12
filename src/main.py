import cv2
import numpy as np
import argparse
from metrics import compute_ssim

from utils import *
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Stereo matching depth estimation")
    parser.add_argument("--image1", 
                        # default='./im0e1.png',
                        default='./scenel.jpg',
                        help="Path to the left image")
    parser.add_argument("--image2", 
                        # default='./im1e1.png',
                        default='./scener.jpg',
                        help="Path to the right image")
    parser.add_argument("--calib", 
                        default='./calib.txt',
                        help="Path to the calibration file")
    parser.add_argument("--disp0", 
                        default='./disp0.pfm',
                        help="Path to the ground truth disparity file for left image")
    parser.add_argument("--disp1",
                        default='./disp1.pfm',
                        help="Path to the ground truth disparity file for right image")
    parser.add_argument("--kernel_size", type=int, 
                        default=55, 
                        help="Kernel size for sliding window search")
    parser.add_argument("--strideX", type=int, 
                        default=5, 
                        help="Stride for sliding window search")
    parser.add_argument("--strideY", type=int,
                        default=5,
                        help="Stride for sliding window search")
    parser.add_argument("--output_dir",
                        default='./output',
                        help="Output directory for saving results")
    args = parser.parse_args()

    output_dir = args.output_dir
    if args.strideX % 2 == 0:
        print(f"Warning: strideX is even ({args.strideX}). Incrementing by 1 to make it odd.")
        args.strideX += 1

    if args.strideY % 2 == 0:
        print(f"Warning: strideY is even ({args.strideY}). Incrementing by 1 to make it odd.")
        args.strideY += 1
    # Override file paths from command-line arguments
    left_img_path = args.image1
    right_img_path = args.image2
    calib_path = args.calib  # Use calib file as needed for calibration parameters if required
    disp0_path = args.disp0
    disp1_path = args.disp1

    # Replace file paths in subsequent code
    left_img = cv2.imread(left_img_path, cv2.IMREAD_COLOR)
    
    right_img = cv2.imread(right_img_path, cv2.IMREAD_COLOR)
    right_img = cv2.imread(right_img_path, cv2.IMREAD_COLOR)

    if left_img is None or right_img is None:
        print("Error loading left/right images.")
        exit(1)

    # Compute disparity map using sliding window search with MSE
    calib_data = parse_calib(calib_path)
    # focal_length = 1733.74
    focal_length = 1
    # baseline = calib_data['baseline']
    baseline = 1
    # max_disp = calib_data['max_disp']
    max_disp = 128
    print(20 * '#')
    print(f"Calibration parameters: focal length={focal_length}, baseline={baseline}, max_disp={max_disp}")
    print(f'Kernel size: {args.kernel_size}, strideX: {args.strideX}, strideY: {args.strideY}')
    print(20 * '#')
    print("Computing disparity map...")

    # computed_disp = stereo_match_cuda(left_img, right_img, kernel_size=args.kernel_size, stride_x=args.strideX, stride_y=args.strideY, max_disp=max_disp)
    computed_disp = stereo_match(left_img, right_img, kernel_size=args.kernel_size, stride_x=args.strideX, stride_y=args.strideY, max_disp=max_disp)
    epsilon = 1e-6
    # Compute depth from computed disparity: z = (b * f) / disparity
    computed_depth = baseline * focal_length / (computed_disp + epsilon)


    # Read ground-truth disparity files (disp0.pfm and disp1.pfm)
    gt_disp0, scale0 = read_pfm("disp0.pfm")
    gt_disp1, scale1 = read_pfm("disp1.pfm")
    # Here we assume disp0 corresponds to the left image ground truth.
    gt_depth = baseline * focal_length / (gt_disp1 + epsilon)

    # Normalize depth maps for display (scale to [0,255])
    # comp_disp_norm = cv2.normalize(computed_disp, None, 0, 255, cv2.NORM_MINMAX)
    # comp_depth_norm = cv2.normalize(computed_depth, None, 0, 255, cv2.NORM_MINMAX)
    # gt_depth_norm = cv2.normalize(gt_depth, None, 0, 255, cv2.NORM_MINMAX)
    # comp_depth_norm = np.uint8(comp_depth_norm)
    # gt_depth_norm = np.uint8(gt_depth_norm)

    import matplotlib.pyplot as plt
    # Display computed depth map and ground truth depth map using matplotlib with a colormap
    tight_min_disp = 55
    tight_max_disp = 142

    depth_min = baseline * focal_length / tight_max_disp
    depth_max = baseline * focal_length / tight_min_disp
    # plt.figure(figsize=(12, 6))
    
    # plt.subplot(1, 2, 1)
    # plt.imshow(computed_depth, cmap='rainbow', vmin=depth_min, vmax=depth_max)
    # plt.imshow(computed_disp, cmap='rainbow')
    dmin = np.min(computed_disp)
    dmax = np.max(computed_disp)
    plt.imshow(computed_disp, cmap='rainbow', vmax=dmax, vmin=dmin)

    # plt.colorbar()
    plt.title("Computed Depth Map")
    plt.axis("off")
    
    # plt.subplot(1, 2, 2)
    # plt.imshow(gt_depth, cmap='rainbow', vmin=depth_min, vmax=depth_max)
    # # plt.colorbar()
    # plt.title("Ground Truth Depth Map")
    # plt.axis("off")
    
    # plt.show()
    
    # mask = ~np.isinf(gt_disp1)
    # mse = np.sqrt(np.mean((computed_disp[:,:,0][mask] - gt_disp1[mask]) ** 2))
    # ssim = compute_ssim(computed_disp[:,:,0][mask], gt_disp1[mask], data_range=tight_max_disp - tight_min_disp)

    os.makedirs(output_dir, exist_ok=True)
    # plt.savefig(os.path.join(output_dir, f"n_{args.kernel_size}_{args.strideX}_{mse:.2f}_{ssim:.2f}.png"))
    plt.savefig((os.path.join(output_dir, 'watch.png')))
    print("Kernel Size:", args.kernel_size)
    print("Mean Squared Error in disparity:", mse)
    print("SSIM:", ssim)
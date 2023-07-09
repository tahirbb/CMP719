import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

orig_dir = '/home/thr/CMP719/AutoEncoderPart/Proje_VERILERI_3TEMMUZ/Karsilastirma/Orig_images'
output_dir = '/home/thr/CMP719/AutoEncoderPart/Proje_VERILERI_3TEMMUZ/Karsilastirma/RestormerProject'

psnr_values = []
ssim_values = []

# Iterate over the images in the output directory
for filename in os.listdir(output_dir):
    if filename.endswith('.png'):
        output_path = os.path.join(output_dir, filename)
        orig_filename = filename.replace('_blur.png', '_orig.png')
        orig_path = os.path.join(orig_dir, orig_filename)

        # Read the images
        output_img = cv2.imread(output_path)
        orig_img = cv2.imread(orig_path)

        # Convert the images to grayscale
        output_gray = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
        orig_gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)

        # Calculate PSNR and SSIM
        psnr = peak_signal_noise_ratio(orig_gray, output_gray)
        ssim = structural_similarity(orig_gray, output_gray)

        # Append the values to the lists
        psnr_values.append(psnr)
        ssim_values.append(ssim)

# Calculate the mean values
mean_psnr = np.mean(psnr_values)
mean_ssim = np.mean(ssim_values)

print('RestormerProject Mean PSNR:', mean_psnr)
print('RestormerProject Mean SSIM:', mean_ssim)

from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import cv2
import random
import os

# Load an image using Pillow
image_path = "./data/image/MAD_0755.jpg"
image = Image.open(image_path)
output_dir = "./result_data/image"

# 1. Dịch chuyển (Translation)
def translate_image(image, tx, ty):
    return image.transform(image.size, Image.AFFINE, (1, 0, tx, 0, 1, ty))

# 2. Xoay (Rotation)
def rotate_image(image, angle):
    return image.rotate(angle)

# 3. Lật (Flipping)
def flip_image(image, direction):
    if direction == 'horizontal':
        return ImageOps.mirror(image)
    elif direction == 'vertical':
        return ImageOps.flip(image)
    else:
        raise ValueError("Direction should be 'horizontal' or 'vertical'")

# 4. Thay đổi độ sáng (Brightness Adjustment)
def adjust_brightness(image, factor):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

# 5. Cắt xén (Cropping)
def crop_image(image, crop_box):
    return image.crop(crop_box)

# 6. Thêm nhiễu (Adding Noise)
def add_noise(image, mean=0, std=25):
    np_image = np.array(image)
    noise = np.random.normal(mean, std, np_image.shape).astype(np_image.dtype)
    noisy_image = np.clip(np_image + noise, 0, 255)
    return Image.fromarray(noisy_image)

# 7. Biến dạng hình học (Geometric Transformations)
def geometric_transform(image, dx, dy):
    np_image = np.array(image)
    rows, cols, ch = np_image.shape
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[50 + dx, 50 + dy], [200 + dx, 50 + dy], [50 + dx, 200 + dy]])
    M = cv2.getAffineTransform(pts1, pts2)
    dst = cv2.warpAffine(np_image, M, (cols, rows))
    return Image.fromarray(dst)

# Example usage

translated_image = translate_image(image, tx=50, ty=30)
translated_image.save(os.path.join(output_dir, "translated_image.jpg"))

rotated_image = rotate_image(image, angle=-10)
rotated_image.save(os.path.join(output_dir, "rotated_image.jpg"))

flipped_image = flip_image(image, direction='horizontal')
flipped_image.save(os.path.join(output_dir, "flipped_image.jpg"))

bright_image = adjust_brightness(image, factor=1.5)
bright_image.save(os.path.join(output_dir, "bright_image.jpg"))

cropped_image = crop_image(image, crop_box=(1000, 1000, 3000, 3000))
cropped_image.save(os.path.join(output_dir, "cropped_image.jpg"))

noisy_image = add_noise(image)
noisy_image.save(os.path.join(output_dir, "noisy_image.jpg"))

geometric_image = geometric_transform(image, dx=100, dy=10)
geometric_image.save(os.path.join(output_dir, "geometric_image.jpg"))

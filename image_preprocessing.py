import torch
from PIL import Image
from skimage.transform import resize
import numpy as np
import torchvision.transforms as transforms


def tensor_to_image(x):
    return x.detach().numpy().squeeze().transpose(1, 2, 0).clip(0, 1)


def image_to_tensor(x):
    if x.ndim == 2:
        return torch.Tensor(x).unsqueeze(0).unsqueeze(0)
    if x.ndim == 3:
        return torch.Tensor(x.transpose(2, 0, 1)).unsqueeze(0)
    if x.ndim == 4:
        return torch.Tensor(x.transpose(0, 3, 1, 2))
    raise RuntimeError("np.array's ndim is out of range 2, 3 or 4.")


def extract_masks(segment):
    extracted_colors = []
    
    # BLUE
    mask_r = (segment[..., 0] < 0.1)
    mask_g = (segment[..., 1] < 0.1)
    mask_b = (segment[..., 2] > 0.9)
    mask = mask_r & mask_g & mask_b
    extracted_colors.append(mask)
    
    # GREEN
    mask_r = (segment[..., 0] < 0.1)
    mask_g = (segment[..., 1] > 0.9)
    mask_b = (segment[..., 2] < 0.1)
    mask = mask_r & mask_g & mask_b
    extracted_colors.append(mask)
    
    # BLACK
    mask_r = (segment[..., 0] < 0.1)
    mask_g = (segment[..., 1] < 0.1)
    mask_b = (segment[..., 2] < 0.1)
    mask = mask_r & mask_g & mask_b
    extracted_colors.append(mask)
    
    # WHITE
    mask_r = (segment[..., 0] > 0.9)
    mask_g = (segment[..., 1] > 0.9)
    mask_b = (segment[..., 2] > 0.9)
    mask = mask_r & mask_g & mask_b
    extracted_colors.append(mask)
    
    # RED
    mask_r = (segment[..., 0] > 0.9)
    mask_g = (segment[..., 1] < 0.1)
    mask_b = (segment[..., 2] < 0.1)
    mask = mask_r & mask_g & mask_b
    extracted_colors.append(mask)
    
    # YELLOW
    mask_r = (segment[..., 0] > 0.9)
    mask_g = (segment[..., 1] > 0.9)
    mask_b = (segment[..., 2] < 0.1)
    mask = mask_r & mask_g & mask_b
    extracted_colors.append(mask)
    
    # GREY
    mask_r = (segment[..., 0] > 0.4) & (segment[..., 0] < 0.6)
    mask_g = (segment[..., 1] > 0.4) & (segment[..., 1] < 0.6)
    mask_b = (segment[..., 2] > 0.4) & (segment[..., 2] < 0.6)
    mask = mask_r & mask_g & mask_b
    extracted_colors.append(mask)
    
    # LIGHT_BLUE
    mask_r = (segment[..., 0] < 0.1)
    mask_g = (segment[..., 1] > 0.9)
    mask_b = (segment[..., 2] > 0.9)
    mask = mask_r & mask_g & mask_b
    extracted_colors.append(mask)
    
    # PURPLE
    mask_r = (segment[..., 0] > 0.9)
    mask_g = (segment[..., 1] < 0.1)
    mask_b = (segment[..., 2] > 0.9)
    mask = mask_r & mask_g & mask_b
    extracted_colors.append(mask)
        
    return extracted_colors


def get_all_masks(path):
    image = Image.open(path)
    np_image = np.array(image, dtype=np.float) / 255
    return extract_masks(np_image)


def is_nonzero(mask, thrs=0.01):
    return np.sum(mask) / mask.size > thrs


def get_masks(path_style, path_content):
    masks_style = get_all_masks(path_style)
    masks_content = get_all_masks(path_content)
    
    non_zero_masks = [is_nonzero(mask_c) and is_nonzero(mask_s) for mask_c, mask_s in zip(masks_content, masks_style)]
    
    masks_style = [mask for mask, cond in zip(masks_style, non_zero_masks) if cond]
    masks_content = [mask for mask, cond in zip(masks_content, non_zero_masks) if cond]
    
    return masks_style, masks_content


def resize_masks(masks_style, masks_content, size):
    resize_mask = lambda mask: resize(mask, size, mode='reflect')
    
    masks_style = [resize_mask(mask) for mask in masks_style]
    masks_content = [resize_mask(mask) for mask in masks_content]
    
    return masks_style, masks_content


def masks_to_tensor(masks_style, masks_content):
    masks_style = [image_to_tensor(mask) for mask in masks_style]
    masks_content = [image_to_tensor(mask) for mask in masks_content]
    
    return masks_style, masks_content


def image_loader(image_name, size):
    loader = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()])

    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image
import torch
#from basic_lib import *

# H: la chieu cao cua tensor, W la chieu rong tensor
# So o chia khong deu nhau
def get_subboxes_full_img(H, W, H_oval, W_oval):
    box_list = []
    H_step, W_step = H // H_oval, W // W_oval
        
    H_residual, W_residual = H % H_oval, W % W_oval
    
    if H_residual > 0:
        row_max = H_step + 1
    else:
        row_max = H_step
        
    if W_residual > 0:
        col_max = W_step + 1
    else:
        col_max = W_step
        
    total_oval = row_max * col_max
    
    for i in range(total_oval):
        row_i = (i) // col_max # phan du
        col_i = (i) % col_max # phan nguyen
        x1, y1 = row_i*H_oval, col_i*W_oval
        
        if (W_residual > 0) & (col_i == W_step):
            y2 = W
        else:
            y2 = (col_i+1)*W_oval
        
        if (H_residual > 0) & (row_i == H_step):
            x2 = H
        else:
            x2 = (row_i+1)*H_oval
                
        box_list.append([0, x1, y1, x2, y2])
    
    return torch.tensor(box_list, dtype=torch.float32)

import numpy as np
from PIL import Image, ImageDraw
from PIL import ImageFont
from typing import Union, Optional, List, Tuple, Text, BinaryIO
from torchvision.transforms.functional import to_tensor

# image: tensor [C,H,W], (0 --> 255)
# boxes: tensor [N,4], XYXY
# labels: list as [1, 3, 4, ...]
# colors: list ['red', 'green', 'blue', ..]
def draw_bounding_boxes(
    image: torch.Tensor,
    boxes: torch.Tensor,
    labels: Optional[List[str]] = None,
    colors: Optional[List[Union[str, Tuple[int, int, int]]]] = None,
    width: int = 1,
    font: Optional[str] = None,
    font_size: int = 10
) -> torch.Tensor:

    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Tensor expected, got {type(image)}")
    elif image.dtype != torch.uint8:
        raise ValueError(f"Tensor uint8 expected, got {image.dtype}")
    elif image.dim() != 3:
        raise ValueError("Pass individual images, not batches")

    ndarr = image.permute(1, 2, 0).numpy()
    img_to_draw = Image.fromarray(ndarr)

    img_boxes = boxes.to(torch.int64).tolist()

    draw = ImageDraw.Draw(img_to_draw)
    txt_font = ImageFont.load_default() if font is None else ImageFont.truetype(font=font, size=font_size)

    for i, bbox in enumerate(img_boxes):
        color = None if colors is None else colors[i]
        draw.rectangle(bbox, width=width, outline=color)

        if labels is not None:
            draw.text((bbox[0], bbox[1]), labels[i], fill=color, font=txt_font)

    return torch.from_numpy(np.array(img_to_draw)).permute(2, 0, 1)


from torch.utils.data import Dataset
import os
class getImg(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        files = os.listdir(img_dir)
        self.imgs = list(sorted([img for img in files]))
        
    def __getitem__(self, idx):
        img_ts = to_tensor(Image.open("{}/{}".format(self.img_dir,self.imgs[idx])))
        return self.imgs[idx], img_ts
    def __len__(self):
        return len(self.imgs)





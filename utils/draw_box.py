import torch
import torchvision
import numpy as np
from PIL import Image, ImageDraw
from PIL import ImageFont
from torchvision.transforms.functional import to_tensor, to_pil_image
from torchvision.io import read_image
from typing import Union, Optional, List, Tuple, Text, BinaryIO

def draw_bounding_boxes(
    image: torch.Tensor,
    boxes: torch.Tensor,
    labels: Optional[List[str]] = None,
    colors: Optional[List[Union[str, Tuple[int, int, int]]]] = None,
    width: int = 1,
    font: Optional[str] = None,
    font_size: int = 10
) -> torch.Tensor:

    """
    Draws bounding boxes on given image.
    The values of the input image should be uint8 between 0 and 255.
    Args:
        image (Tensor): Tensor of shape (C x H x W)
        bboxes (Tensor): Tensor of size (N, 4) containing bounding boxes in (xmin, ymin, xmax, ymax) format. Note that
            the boxes are absolute coordinates with respect to the image. In other words: `0 <= xmin < xmax < W` and
            `0 <= ymin < ymax < H`.
        labels (List[str]): List containing the labels of bounding boxes.
        colors (List[Union[str, Tuple[int, int, int]]]): List containing the colors of bounding boxes. The colors can
            be represented as `str` or `Tuple[int, int, int]`.
        width (int): Width of bounding box.
        font (str): A filename containing a TrueType font. If the file is not found in this filename, the loader may
            also search in other directories, such as the `fonts/` directory on Windows or `/Library/Fonts/`,
            `/System/Library/Fonts/` and `~/Library/Fonts/` on macOS.
        font_size (int): The requested font size in points.
    """

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

#img = to_tensor(Image.open('us.jpg'))
img = read_image('us.jpg')
print(img.dtype)

color_cls = {
	1: "red",
	2: "white",
	3: "blue"
}

boxes = torch.tensor(
[[100,100, 500, 500],
[1000, 1000, 1500,1500]]
)
labels = [2,3]
colors = [color_cls[i] for i in labels]

result = draw_bounding_boxes(img, boxes, colors=colors, width=2)/255
torchvision.utils.save_image(result, 'usbox.jpg')



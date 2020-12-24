# https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator


backbone = torchvision.models.mobilenet_v2(pretrained=True).features
backbone.out_channels = 1280


anchor_sizes = ((4,))
anchor_ratios = ((0.5, 1.0, 2.),) * len(anchor_sizes)
anchor_generator = AnchorGenerator(
	sizes = anchor_sizes,
	aspect_ratios = anchor_ratios
)


roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1'],
                                                output_size=7,
                                                sampling_ratio=2)

model = FasterRCNN(
	backbone,
	num_classes=2,
	rpn_anchor_generator=anchor_generator,
	box_roi_pool=roi_pooler
).to("cuda:0")
model.eval()
print(model)

x = torch.stack([torch.rand(3, 300, 400), torch.rand(3, 300, 400)]).to("cuda:0")
with torch.no_grad():
	x_out = model(x)
	print(x_out)



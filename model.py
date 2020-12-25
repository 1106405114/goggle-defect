# author: ddthuan@pdu.edu.vn
# buil model Faster RCNN

from utils.basic_lib import *
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import FasterRCNN

def build_model(num_cls):
	# Custom backbone
	backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone('resnet50', True)
	backbone.out_channels = 256

	# custom RPN
	anchor_sizes = ((8,), (16,), (32,), (64,), (128,),) # notice that backbone & fpn will output 5 feature maps.
	anchor_ratios = ((0.5, 1, 1.7, 2.),) * len(anchor_sizes)
	ag = AnchorGenerator(sizes= anchor_sizes, aspect_ratios = anchor_ratios)

	# custom RoIHead
	roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=14, sampling_ratio=2)
	
	# build model
	model = FasterRCNN(backbone, num_classes = num_cls, rpn_anchor_generator = ag, box_roi_pool = roi_pooler)
	return model

"""
rcnn = build_model(3)
rcnn.eval()
x = torch.randn((2,3,512,512), dtype=torch.float32)

with torch.no_grad():
	x_out = rcnn(x)
	print(x_out)
	#print(x_out.dtype)
"""


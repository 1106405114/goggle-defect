# https://github.com/pytorch/vision/issues/978

# Custom anchor
# https://stackoverflow.com/questions/56962533/cant-change-the-anchors-in-faster-rcnn
from utils.basic_lib import *
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead

#torch.backends.cudnn.deterministic = True
#torch.set_deterministic(bool)

# load mo hinh faster RCNN
rcnn_pretrain = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
"""
# tao Anchor cho mang FPN
ag = AnchorGenerator(
	sizes = ((16,), (32,), (64,), (128,)),
	aspect_ratios = tuple([(0.5, 1., 1.5, 2.) for _ in range(4)])
)
model.rpn.anchor_generator = ag
model.rpn.head = RPNHead(256, ag.num_anchors_per_location()[0])
"""

###########  Tao Faster RCNN tu backbone, AnchorGenerator, RoIPool

# 1. Thay doi Backbone
#backbone = torchvision.models.mobilenet_v2(pretrained=True).features
#backbone.out_channels = 1280
backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone('resnet50', True)
backbone.out_channels = 256

# 2. So doi tuong
num_cls = 6


# 3. Thay doi mang Anchor
# tao Anchor mang thong thuong
anchor_sizes = ((4,), (8,), (16,), (32,), (64,))
anchor_ratios = ((0.5, 1.0, 2.),) * len(anchor_sizes)
ag = AnchorGenerator(
	sizes = anchor_sizes,
	aspect_ratios = anchor_ratios
)

# 4. Thay doi mang RoIHeads
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3', '4'], output_size=14, sampling_ratio=2)


# Tao model FasterRCNN
rcnn_custom = FasterRCNN(backbone, num_classes=num_cls, rpn_anchor_generator = ag, box_roi_pool = roi_pooler).to("cuda:0")
#print(rcnn_custom)
rcnn_custom.eval()

x = torch.stack([torch.rand(3, 300, 400), torch.rand(3, 300, 400)]).to("cuda:0")
#x = torch.randn((1,3,512,512), dtype=torch.float32).to("cuda:0")
with torch.no_grad():
	x_out = rcnn_custom(x)
	print(x_out)








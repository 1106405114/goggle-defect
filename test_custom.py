# author: ddthuan@pdu.edu.vn

import numpy as np
import cv2
import os
import torch
from tqdm import tqdm
from model import build_model

from config import TEST_PATH, TEST_RESULT, PRE_THRES, NUM_CLS
from torchvision.ops import batched_nms, nms
# set the computation device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

cls_name = {1: "dust", 2: "splotlight1", 3: "splotlight2", 4: "string"}

# load the model and the trained weights
model = build_model(NUM_CLS).to(device)
model.load_state_dict(torch.load('checkpoints/goggle4Cls.pth'))

DIR_TEST = TEST_PATH
test_images = os.listdir(DIR_TEST)
print(f"Validation instances: {len(test_images)}")


def draw_box(img, boxes, name_cls):
    for (box, name_cls) in zip(boxes, labels):
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 1)
        cv2.putText(img, str(int(name_cls)), (int(box[0]), int(box[1]-5)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
    #cv2.imwrite(f"{TEST_RESULT}/{test_images[i]}", img,)    

detection_threshold = PRE_THRES
model.eval()
with torch.no_grad():
    for i, image in tqdm(enumerate(test_images), total=len(test_images)):
        orig_image = cv2.imread(f"{DIR_TEST}/{test_images[i]}", cv2.IMREAD_COLOR)
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # make the pixel range between 0 and 1
        image /= 255.0
        image = np.transpose(image, (2, 0, 1)).astype(np.float)
        image = torch.tensor(image, dtype=torch.float).cuda()
        image = torch.unsqueeze(image, 0)
        cpu_device = torch.device("cpu")
        outputs = model(image)
        
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        scores = outputs[0]['scores'].data.numpy()
        
       
        if len(outputs[0]['boxes']) != 0:
            scores = outputs[0]['scores'].data.numpy()
            labels = outputs[0]["labels"]
            boxes = outputs[0]["boxes"].data.numpy()
            print("score output: ", len(scores))
            print("label output: ", labels)
            print("boxes output: ", len(boxes))
            
            
            print("after filting")
            id_filter = scores >= detection_threshold
            score_filter = scores[id_filter]
            label_filter = labels[id_filter]
            box_filter = boxes[id_filter]
            
            #print(label_filter)
            
            box_filter_nms = torch.tensor(box_filter, requires_grad=True)
            score_filter_nms = torch.tensor(score_filter, requires_grad=True)
            
# =============================================================================
#             print(score_filter)
#             print(label_filter)
#             print(box_filter)
# =============================================================================
            
            #rs_nms = batched_nms(box_filter_nms, score_filter_nms, label_filter, torch.tensor([0.7]))
            rs_nms = nms(box_filter_nms, score_filter_nms, torch.tensor(0.2))
            print(len(rs_nms))
            box_filter_nms[rs_nms]
            print("===============================================")
            
            draw_box(orig_image, box_filter_nms[rs_nms], torch.ones(len(rs_nms)))
# =============================================================================
#         if len(outputs[0]['boxes']) != 0:
#             for counter in range(len(outputs[0]['boxes'])):
#                 boxes = outputs[0]['boxes'].data.numpy()
#                 scores = outputs[0]['scores'].data.numpy()                
#                 
#                 id_filting = scores >= detection_threshold
#                 boxes = boxes[id_filting].astype(np.int32)
#                 
#                 labels = outputs[0]['labels'].data
#                 labels = labels[id_filting]                
#                 
#                 draw_boxes = boxes.copy()
#                 #name_clses = labels.copy()
#                 
#             for (box, name_cls) in zip(draw_boxes, labels):
#                 cv2.rectangle(orig_image,
#                             (int(box[0]), int(box[1])),
#                             (int(box[2]), int(box[3])),
#                             (0, 0, 255), 1)
#                 cv2.putText(orig_image, str(int(name_cls)), 
#                             (int(box[0]), int(box[1]-5)),
#                             cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 
#                             2, lineType=cv2.LINE_AA)
#             cv2.imwrite(f"{TEST_RESULT}/{test_images[i]}", orig_image,)
# =============================================================================
        cv2.imwrite(f"{TEST_RESULT}/{test_images[i]}", orig_image,)
print('TEST PREDICTIONS COMPLETE')


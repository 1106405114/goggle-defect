# Developer: ddthuan@pdu.edu.vn

import torch
import torchvision

import numpy as np
import os
import platform

from torchvision.utils import save_image
from torchvision import transforms as T
from torchvision.ops import roi_align, nms
from torchvision.transforms.functional import to_tensor, to_pil_image

from utils.basic_fn import get_subboxes_full_img, draw_bounding_boxes

os.environ["PYLON_CAMEMU"] = "3"

from pypylon import genicam
from pypylon import pylon
import sys

countOfImagesToGrab = 3

maxCamerasToUse = 3

# The exit code of the sample application.
exitCode = 0

camera_id = {
    0:"cam1",
    1:"cam2",
    2:"cam3",
}


sample_name = sys.argv[1]
sample_side = int(sys.argv[2])


roi = torch.tensor(
[[
  [0., 110, 300, 4096,3000], #cam1
  [0, 10, 250,3800,2900], #cam2
  [0, 10, 250, 3980, 2950], #cam3
  ],
 [
  [0, 150, 0, 4096, 2650], #cam1
  [0, 0, 0, 3900, 2800], #cam2
  [0, 0, 0, 3950, 2500], #cam3
  ]
 ], 
)


from model import build_model
from config import TEST_RESULT, PRE_THRES, NUM_CLS


# set the computation device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

cls_name = {1: "dust", 2: "splotlight1", 3: "splotlight2", 4: "string"}
cls_color = {1: "red", 2: "white", 3: "green", 4: "blue"}

# load the model and the trained weights
model = build_model(NUM_CLS).to(device)
model.load_state_dict(torch.load('checkpoints/goggle4Cls.pth', map_location=torch.device('cpu')))
detection_threshold = PRE_THRES
model.eval()


def pre_roi(img_name, img_ts, img_side, cam_id):
    box = roi[img_side-1, cam_id]
    W_img, H_img = int(box[3]-box[1]), int(box[4]-box[2])
    img_roi = roi_align(img_ts[None], box[None], (H_img, W_img))
    
    save_image(img_roi[0], "roi\{}".format(img_name))
    
    # bat dau xu li chia anh
    #boxes_div = get_subboxes_full_img(img_roi.shape[3], img_roi.shape[2], 1500, 2048)
    boxes_div = get_subboxes_full_img(img_roi.shape[3], img_roi.shape[2], 512, 512)
    #boxes_div = get_subboxes_full_img(img_roi.shape[3], img_roi.shape[2], 512, 512)
    box_len = len(boxes_div)
    img_div = roi_align(img_roi, boxes_div, output_size = (512, 512))
    #for i in range(box_len):
    #    save_image(img_div[i], "div\{}_{}.jpg".format(img_name[:-4], i))    
    save_image(img_div, "div\{}".format(img_name))
    
    # bat dau do tim 
    outputs = model(img_div)
    outputs = [{k: v for k, v in t.items()} for t in outputs]
    
    for i, (outputi) in enumerate(outputs):
        boxes = outputi["boxes"]
        scores = outputi["scores"]
        labels = outputi["labels"]
        
        if len(boxes) != 0:
            # loc theo box confidence
            img_uint8 = (img_div[i]*255).type(torch.uint8)
            
            id_thre = scores >= detection_threshold
            score_thold = scores[id_thre]
            label_thold = labels[id_thre]
            box_thold = boxes[id_thre]
            
            
            # loc theo NMS
            box_nms = box_thold.clone()
            box_nms.require_grad = True
            
            score_nms = score_thold.clone()
            score_nms.require_grad = True
                        
            rs_nms = nms(box_nms, score_nms, torch.tensor(0.2))
            score_nms = score_nms[rs_nms]
            box_nms = box_nms[rs_nms].type(torch.int64)
            label_nms = label_thold[rs_nms]
            #print("label: ", label_nms.clone().numpy())
            
            colors = [cls_color[int(i)] for i in label_nms]            
            result = draw_bounding_boxes(img_uint8, box_nms, colors=colors, width=2)/255
            save_image(result, f"{TEST_RESULT}/{img_name[:-5]}_{i}.jpg")
        else:
            save_image(img_div[i], f"{TEST_RESULT}/{img_name[:-5]}_{i}.jpg")
    

try:

    
    # Get the transport layer factory.
    tlFactory = pylon.TlFactory.GetInstance()

    # Get all attached devices and exit application if no device is found.
    devices = tlFactory.EnumerateDevices()
    
    # chi dinh camera capture anh
    devices = (devices[1], devices[4], devices[3])

    if len(devices) == 0:
        raise pylon.RuntimeException("No camera present.")

    # Create an array of instant cameras for the found devices and avoid exceeding a maximum number of devices.
    cameras = pylon.InstantCameraArray(min(len(devices), maxCamerasToUse))

    l = cameras.GetSize()

    # Create and attach all Pylon Devices.
    for i, cam in enumerate(cameras):
        cam.Attach(tlFactory.CreateDevice(devices[i]))

        # Print the model name of the camera.
        print("Using device ", cam.GetDeviceInfo().GetModelName())

    cameras.StartGrabbing()

    # Grab c_countOfImagesToGrab from the cameras.
    img = pylon.PylonImage()

    for i in range(countOfImagesToGrab):
        if not cameras.IsGrabbing():
            break

        grabResult = cameras.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        #print(grabResult)
        camera_name = camera_id[i%maxCamerasToUse]
        
        img_name = "{}_{}_{}.jpeg".format(sample_name, sample_side, camera_name)
        
        # chuyen doi dinh dang
        converter = pylon.ImageFormatConverter()
        converter.OutputPixelFormat = pylon.PixelType_RGB8packed
        converted = converter.Convert(grabResult)
        img_rgb = converted.GetArray()
        img_ts = to_tensor(img_rgb)
        pre_roi(img_name, img_ts, sample_side, i)
        
        
        #save_image(to_tensor(img_rgb), "ts_{}_{}.jpeg".format(camera_name, i))
        
        # Luu anh chup tu camera
        img.AttachGrabResultBuffer(grabResult)
        if platform.system() == 'Windows':
            ipo = pylon.ImagePersistenceOptions()
            #quality = 250 - i * 50
            quality = int(np.random.randint(99,100, 1))
            ipo.SetQuality(quality)            
            #filename = "saved_pypylon_img_%d.jpeg" % quality
            #filename = "{}\{}_{}_{}_{}.jpeg".format(preprocessing_path, sample_name, camera_name, i, quality, )
            filename = "raw\{}_{}_{}.jpeg".format(sample_name, sample_side, camera_name)
            #print(save_path, img_name)
            print(filename)
            img.Save(pylon.ImageFileFormat_Jpeg, filename, ipo)
            
        print("=============== Finish ==================")
        cameraContextValue = grabResult.GetCameraContext()


except genicam.GenericException as e:
    # Error handling
    print("An exception occurred.", e.GetDescription())
    exitCode = 1

# Comment the following two lines to disable waiting on exit.
sys.exit(exitCode)
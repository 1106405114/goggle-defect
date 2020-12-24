import torch
import torchvision

import numpy as np
import os
import platform

from torchvision.utils import save_image
from torchvision import transforms as T
from torchvision.transforms.functional import to_tensor, to_pil_image 

os.environ["PYLON_CAMEMU"] = "3"

from pypylon import genicam
from pypylon import pylon
import sys

countOfImagesToGrab = 10

maxCamerasToUse = 5

# The exit code of the sample application.
exitCode = 0

camera_id = {
    0:"cam_le",
    1:"cam_lt",
    2:"cam_ri",
    3:"cam_rt",
    4:"cam_to"
}

#print(camera_id[2])

try:

    # Get the transport layer factory.
    tlFactory = pylon.TlFactory.GetInstance()

    # Get all attached devices and exit application if no device is found.
    devices = tlFactory.EnumerateDevices()
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

    # Starts grabbing for all cameras starting with index 0. The grabbing
    # is started for one camera after the other. That's why the images of all
    # cameras are not taken at the same time.
    # However, a hardware trigger setup can be used to cause all cameras to grab images synchronously.
    # According to their default configuration, the cameras are
    # set up for free-running continuous acquisition.
    cameras.StartGrabbing()

    # Grab c_countOfImagesToGrab from the cameras.
    img = pylon.PylonImage()

    for i in range(countOfImagesToGrab):
        if not cameras.IsGrabbing():
            break

        grabResult = cameras.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        #print(grabResult)
        camera_name = camera_id[i%maxCamerasToUse]
        
        # chuyen doi dinh dang
        converter = pylon.ImageFormatConverter()
        converter.OutputPixelFormat = pylon.PixelType_RGB8packed
        converted = converter.Convert(grabResult)
        img_rgb = converted.GetArray()
        save_image(to_tensor(img_rgb), "ts_{}_{}.jpeg".format(camera_name, i))
        #print("shape: ", img_rgb.shape)
        #print(img_rgb[200:205,200:205,0])
        
        # Luu anh chup tu camera
        img.AttachGrabResultBuffer(grabResult)
        if platform.system() == 'Windows':
            ipo = pylon.ImagePersistenceOptions()
            #quality = 250 - i * 50
            quality = int(np.random.randint(99,100, 1))
            ipo.SetQuality(quality)            
            #filename = "saved_pypylon_img_%d.jpeg" % quality
            filename = "{}_{}_{}.jpeg".format(camera_name, i, quality)
            print(filename, img)
            img.Save(pylon.ImageFileFormat_Jpeg, filename, ipo)
            
        print("=================================")
        # When the cameras in the array are created the camera context value
        # is set to the index of the camera in the array.
        # The camera context is a user settable value.
        # This value is attached to each grab result and can be used
        # to determine the camera that produced the grab result.
        cameraContextValue = grabResult.GetCameraContext()

        # Print the index and the model name of the camera.
        #print("Camera ", cameraContextValue, ": ", cameras[cameraContextValue].GetDeviceInfo().GetModelName())

        # Now, the image data can be processed.
        #print("GrabSucceeded: ", grabResult.GrabSucceeded())
        #print("SizeX: ", grabResult.GetWidth())
        #print("SizeY: ", grabResult.GetHeight())
        #img = grabResult.GetArray()
        #print("Gray value of first pixel: ", img[0, 0])

except genicam.GenericException as e:
    # Error handling
    print("An exception occurred.", e.GetDescription())
    exitCode = 1

# Comment the following two lines to disable waiting on exit.
sys.exit(exitCode)
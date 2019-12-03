from __future__ import print_function
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import logging
#hello
import zipfile
import tkinter
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
import matplotlib
from PIL import Image
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)
# Just disables the warning shown at the end, doesn't enable AVX/FMA(Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')
print("Trying to import tf")
import tensorflow as tf
print("imported tf")
sys.path.append("..")
from utils import label_map_util
from utils import visualization_utils as vis_util
from utils import backbone

import json
from livepredictor import cumulative_object_counting_x_axis

input_video = "rtsp://user2:user#55@46.52.182.186:37778/cam/realmonitor?channel=2&subtype=0"


detection_graph, category_index = backbone.set_model('oldtestped_inference_graph', 'object-detection.pbtxt')

is_color_recognition_enabled = 0 # set it to 1 for enabling the color prediction for the detected objects

label = "pedestrian"

#for door

x= 820
y = 66
w = 468
h = 800

'''
x= 1500
y = 22
w = 468
h = 300
'''
print("starting")
total, framecount , vidnameused = cumulative_object_counting_x_axis(input_video, detection_graph, category_index,is_color_recognition_enabled,x,y,w,h,label ,write=False, display=True, brighten = False)
exit(0)


#muting all warnings
import logging
import sys
import os



import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
print("Trying to import tf")
import tensorflow as tf
print("Importing tensorflow")
import zipfile
import tkinter
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
import matplotlib
from PIL import Image
import cv2

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops


from utils import label_map_util

from utils import visualization_utils as vis_util
matplotlib.use('TkAgg')
# What model to use.
MODEL_NAME = 'oldtestped_inference_graph'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'object-detection.pbtxt')

NUM_CLASSES = 1



#starting the video feed

# Just disables the warning shown at the end, doesn't enable AVX/FMA(Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# WARNING: Logging before flag parsing goes to stderr.)
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')



category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)



# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ ]

#cycle through images in dir
import os

for filename in os.listdir(PATH_TO_TEST_IMAGES_DIR):
    if filename.endswith(".jpg") : 
        print(filename)
        TEST_IMAGE_PATHS.append(os.path.join(PATH_TO_TEST_IMAGES_DIR, filename))
         # print(os.path.join(directory, filename))
        continue
    else:
        continue




# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                            tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image.shape[1], image.shape[2])
                detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                                         feed_dict={image_tensor: image})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                    'detection_classes'][0].astype(np.int64)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict



print(TEST_IMAGE_PATHS)
for image_path in TEST_IMAGE_PATHS:
    print()
    print("here")
    image = Image.open(image_path)
    
    print("Image opened")
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    print("Loading img into np array")
    image_np = load_image_into_numpy_array(image)
    print(image_np.shape)
    
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    print("expandeding the img")
    image_np_expanded = np.expand_dims(image_np, axis=0)
    
    # Actual detection.
    print("detecting")
    output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
    image_height, image_width, _ = image_np.shape
    output = []
    threshold = 0.5
    for index, score in enumerate(output_dict['detection_scores']):

        if score < threshold:
            continue

        # Get data(label, xmin, ymin, xmax, ymax)
        label = category_index[output_dict['detection_classes'][index]]['name']
        ymin, xmin, ymax, xmax = output_dict['detection_boxes'][index]

        output.append((label, int(xmin * image_width), int(ymin * image_height), int(xmax * image_width),int(ymax * image_height)))

    print(len(output))
    if len(output) > 0:
        print('stuff detected')
    # Visualization of the results of a detection.
    '''
    print("Visualizing")
    vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=8)

    plt.figure(figsize=IMAGE_SIZE)
    print("Lauching window")
    plt.imshow(image_np)
    plt.show()
    '''



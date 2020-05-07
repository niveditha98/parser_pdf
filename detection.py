import os
import numpy as np
import tensorflow as tf
from PIL import Image
from utils import label_map_util
from utils import visualization_utils as vis_util
import sys
from object_detection.utils import ops as utils_ops
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
import csv

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
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile('models/research/training/output_inference_graph_v1.pb/frozen_inference_graph.pb', 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

NUM_CLASSES=1
label_map = label_map_util.load_labelmap('models/research/training/data/table_detection.pbtxt')
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
    
# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)
PATH_TO_TEST_IMAGES_DIR = 'Test'
'''TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'Doc{}.pdfpage_{}.jpg'.format(i,j)) for i in range(2, 4) for j in range(1,2)]
TEST_IMAGE_PATHS.append('Test/Doc4.pdfpage_3.jpg')
TEST_IMAGE_PATHS.append('Test/Doc4.pdfpage_4.jpg')
TEST_IMAGE_PATHS.append('Test/jpg0.jpg')'''
TEST_IMAGE_PATHS=[]
file=sys.argv[1]
for f in os.listdir('C:/Users/Admin/AppData/Local/Programs/Python/Python36/Lib/site-packages/example/'):
            if f.startswith(file+'page_'):
                if f.endswith(".jpg"):
                    fi=os.path.join("C:/Users/Admin/AppData/Local/Programs/Python/Python36/Lib/site-packages/example/", f)
                    TEST_IMAGE_PATHS.append(fi)
cut=TEST_IMAGE_PATHS[0].split('.pdf')
name=cut[0]+'.pdf.csv'

for image_path in TEST_IMAGE_PATHS:
  image = Image.open(image_path)
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = load_image_into_numpy_array(image)
  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
  image_np_expanded = np.expand_dims(image_np, axis=0)
  # Actual detection.
  output_dict = run_inference_for_single_image(image_np, detection_graph)
  boxes=output_dict['detection_boxes'].copy()
  scores=output_dict['detection_scores']
  w,h=image.size
  new_boxes = []
  for i, box in enumerate(np.squeeze(boxes)):
    if(np.squeeze(scores)[i] > 0.8):
        box1=box.tolist()
        box1.append(image_path)
        box1[0]=int(box[0]*h)
        box1[1]=int(box[1]*w)
        box1[2]=int(box[2]*h)
        box1[3]=int(box[3]*w)
        new_boxes.append(box1)
  with open(name, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(new_boxes)
  # Visualization of the results of a detection.
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
  plt.imshow(image_np)
  im = Image.fromarray(image_np)
  im.save(image_path+'.jpg')
  

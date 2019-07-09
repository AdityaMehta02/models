import os
import sys

from io import BytesIO
import tempfile
from six.moves import urllib

import matplotlib
matplotlib.use('agg')
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import argparse

import tensorflow as tf
from tensorflow.core.framework import *

from os import walk

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--indir', action='store', help='image path')
parser.add_argument('-t', '--target', action='store', help='ground truth path')
parser.add_argument('-o', '--outdir', action='store', help='output dir', default='Prediction')
args = parser.parse_args()


flist = []
for (dirpath, dirnames, filenames) in walk(args.indir):
    flist.extend(filenames)
    break

print("Images:", flist)



class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, graph_path):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()

    graph_def = graph_pb2.GraphDef()
    with open(graph_path, "rb") as pbfile:
        graph_def.ParseFromString(pbfile.read())

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(graph=self.graph)

  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    return resized_image, seg_map


def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=int)
  ind = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

  return colormap


def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_pascal_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]


def vis_segmentation(image, seg_map):
  """Visualizes input image, segmentation map and overlay view."""
  plt.figure(figsize=(15, 5))
  grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

  plt.subplot(grid_spec[0])
  plt.imshow(image)
  plt.axis('off')
  plt.title('input image')

  plt.subplot(grid_spec[1])
  seg_image = label_to_color_image(seg_map).astype(np.uint8)
  plt.imshow(seg_image)
  plt.axis('off')
  plt.title('segmentation map')

  plt.subplot(grid_spec[2])
  plt.imshow(image)
  plt.imshow(seg_image, alpha=0.7)
  plt.axis('off')
  plt.title('segmentation overlay')

  unique_labels = np.unique(seg_map)
  ax = plt.subplot(grid_spec[3])
  plt.imshow(
      FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
  ax.yaxis.tick_right()
  plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
  plt.xticks([], [])
  ax.tick_params(width=0.0)
  plt.grid('off')
  plt.show()


LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

download_path = '/home/adi/Workspace/models/research/deeplab/datasets/goat_molt_seg/exp/train_on_trainval_set/export/frozen_inference_graph.pb'
MODEL = DeepLabModel(download_path)

SAMPLE_IMAGE = 'image1'  # @param ['image1', 'image2', 'image3']
IMAGE_URL = ''  #@param {type:"string"}

_SAMPLE_URL = ('https://github.com/tensorflow/models/blob/master/research/'
               'deeplab/g3doc/img/%s.jpg?raw=true')

def serve_pil_image(pil_img):
    img_io = BytesIO()
    pil_img.save(img_io, 'JPEG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')


def run_visualization(url):
  """Inferences DeepLab model and visualizes result."""
  try:
    f = urllib.request.urlopen(url)
    jpeg_str = f.read()
    original_im = Image.open(BytesIO(jpeg_str))
  except IOError:
    print('Cannot retrieve image. Please check url: ' + url)
    return

  print('running deeplab on image %s...' % url)
  resized_im, seg_map = MODEL.run(original_im)

  vis_segmentation(resized_im, seg_map)


#image_url = IMAGE_URL or _SAMPLE_URL % SAMPLE_IMAGE
#run_visualization(image_url)

def calc_iou():
    for fname in flist:
        fpath = args.indir + '/' + fname
        tpath = args.target + '/' + fname.split('.')[0] + '.png'
        original_im = Image.open(fpath)
        target_im = Image.open(tpath)
        resized_im, seg_map = MODEL.run(original_im)
        target_img = target_im.resize((resized_im.size[0], resized_im.size[1]))
        target_map = np.array(target_img)
        intersection = np.logical_and(target_map, seg_map)
        union = np.logical_or(target_map, seg_map)
        iou_score = np.sum(intersection) / np.sum(union)
        print("Image:", fname, "IOU:", iou_score)
        seg_image = label_to_color_image(seg_map).astype(np.uint8)
        seg_img = Image.fromarray(seg_image, 'RGB')
        seg_img.save(args.outdir + '/' + fname )
    
if __name__ == '__main__':
    if not (args.target):
        print("Ground truth Target not provided")
        parser.print_help()
        sys.exit(0)
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    calc_iou()

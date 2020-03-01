import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import time
from PIL import Image, ImageDraw

saved_model_path = "checkpoints/disable_hflip"
im_path = 'demo.jpg'
out_path = 'demo_detected.jpg'

def draw_boxes(img, boxes, color='green', width=2):
    '''
    draw the boxes in the img
    :param img: Pillow Image or numpy
    :param boxes: boxes, [[ymax, xmax, ymin, xmin]...]
    :param color: color
    :return: Image drawed boxes
    '''
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype(np.uint8), mode='RGB')
    elif not isinstance(img, Image.Image):
        raise ValueError("image must be a Image or ndarray.")
    draw = ImageDraw.Draw(img)
    for box in boxes:
        draw.rectangle([box[1], box[0], box[3], box[2]], outline=color, width=width)
    return img

def run():
    sess = tf.Session()
    tf.saved_model.load(sess, ['serve'], saved_model_path, )

    inp = sess.graph.get_tensor_by_name('input_images:0')
    boxes_tensor = sess.graph.get_tensor_by_name('boxes:0')
    scores_tensor = sess.graph.get_tensor_by_name('scores:0')

    im = Image.open(im_path)
    im = np.array(im).astype(np.float32)
    boxes, scores = sess.run([boxes_tensor,scores_tensor], feed_dict={inp: np.expand_dims(im, axis=0)})

    boxes = boxes[0]
    scores = scores[0]
    mask = scores>0.8
    boxes = boxes[mask]
    scores = scores[mask]
    print('boxes')
    print(boxes)
    print('scores')
    print(scores)
    out_im = draw_boxes(im, boxes)
    out_im.save(out_path)

if __name__ =='__main__':
    run()
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import tensorflow as tf
from PIL import Image
import numpy as np
import time
from widerface import WiderEval, save_wider_result

saved_model_path = "checkpoints/disable_hflip"

imgs_dir = '/home/luojiapeng/datasets/widerface/WIDER_val/images'
anno_mat = '/home/luojiapeng/datasets/widerface/wider_face_split/wider_face_val.mat'

output_dir = 'wider_val_result'

def run():
    sess = tf.Session()
    tf.saved_model.load(sess, ['serve'], saved_model_path, )

    inp = sess.graph.get_tensor_by_name('input_images:0')
    boxes_tensor = sess.graph.get_tensor_by_name('boxes:0')
    scores_tensor = sess.graph.get_tensor_by_name('scores:0')

    dataset = WiderEval(imgs_dir=imgs_dir, anno_mat=anno_mat)
    
    results = []
    for i in range(len(dataset)):
        if i % 100 == 0:
            print(f'processing {i}')
        im = dataset[i]
        boxes, scores = sess.run([boxes_tensor,scores_tensor], feed_dict={inp: np.expand_dims(im, axis=0)})
        results.append({
            'bboxes': boxes,
            'scores': scores
        })
    save_wider_result(output_dir, dataset.imgs, results)

if __name__ == '__main__':
    run()

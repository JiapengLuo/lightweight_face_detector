import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import time
from PIL import Image

saved_model_path = "checkpoints/disable_hflip"

fddb_imgs_file = '/home/luojiapeng/datasets/fddb/imgList.txt'
fddb_base_dir = '/home/luojiapeng/datasets/fddb'

outFile = os.path.join('detectBboxes.txt')

def get_fddb_imgs():
    anno_file = open(fddb_imgs_file, 'r')
    fnames = []
    for line in anno_file.readlines():
        line = line.rstrip()
        if line:
            fnames.append(line)
    return fnames

def save_fddb_result(outFile, fnames, result):
    with open(outFile,'w') as fid:
        i=0
        for fname, res in zip(fnames, result):
            bboxes, scores = res['bboxes'], res['scores']
            bboxes, scores = bboxes[0], scores[0]
            assert len(bboxes) == len(scores)

            fid.write(fname+'\n')
            if bboxes is None:
                fid.write(str(1) + '\n')
                fid.write('%d %d %d %d %.8f\n' % (0, 0, 0, 0, 0))
                continue
            else:
                fid.write(str(len(bboxes)) + '\n')
                for _i in range(len(scores)):
                    s, b =scores[_i], bboxes[_i]
                    b=[int(np.round(x)) for x in b]
                    fid.write('%d %d %d %d %.8f\n' % (b[1], b[0], b[3] - b[1], b[2] - b[0], s))
        fid.close()

def run():
    sess = tf.Session()
    tf.saved_model.load(sess, ['serve'], saved_model_path, )

    inp = sess.graph.get_tensor_by_name('input_images:0')
    boxes_tensor = sess.graph.get_tensor_by_name('boxes:0')
    scores_tensor = sess.graph.get_tensor_by_name('scores:0')
    
    fnames = get_fddb_imgs()
    image_path_list = [os.path.join(fddb_base_dir, x)+'.jpg' for x in fnames]
    results = []
    for i in range(len(image_path_list)):
        if i % 100 == 0:
            print(f'processing {i}')
        im = Image.open(image_path_list[i]).convert('RGB')
        im = np.array(im).astype(np.float32)
        boxes, scores = sess.run([boxes_tensor,scores_tensor], feed_dict={inp: np.expand_dims(im, axis=0)})
        results.append({
            'bboxes': boxes,
            'scores': scores
        })    
    save_fddb_result(outFile, fnames, results)

if __name__ == '__main__':
    run()

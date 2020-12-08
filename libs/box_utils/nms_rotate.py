# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pyximport
pyximport.install()

import numpy as np
import cv2
from libs.configs import cfgs
import tensorflow as tf
from libs.box_utils.rotate_polygon_nms import rotate_gpu_nms
from libs.box_utils.coordinate_convert import forward_convert


def nms_rotate(decode_boxes, scores, iou_threshold, max_output_size,
               use_angle_condition=False, angle_threshold=0, use_gpu=True, gpu_id=0):
    """
    :param boxes: format [x_c, y_c, w, h, theta]
    :param scores: scores of boxes
    :param threshold: iou threshold (0.7 or 0.5)
    :param max_output_size: max number of output
    :return: the remaining index of boxes
    """

    if use_gpu:
        keep = nms_rotate_gpu(boxes_list=decode_boxes,
                              scores=scores,
                              iou_threshold=iou_threshold,
                              angle_gap_threshold=angle_threshold,
                              use_angle_condition=use_angle_condition,
                              device_id=gpu_id)

        keep = tf.cond(
            tf.greater(tf.shape(keep)[0], max_output_size),
            true_fn=lambda: tf.slice(keep, [0], [max_output_size]),
            false_fn=lambda: keep)

    else:
        keep = tf.py_func(nms_rotate_cpu,
                          inp=[decode_boxes, scores, iou_threshold, max_output_size],
                          Tout=tf.int64)
    return keep


def rot_nms(filter_score, filter_quads, max_boxes, nms_thresh):
    max_score_idx = tf.argmax(filter_score)
    best_quad = filter_quads[max_score_idx]
    y_diff = best_quad[..., 7] + best_quad[..., 5] - best_quad[..., 3] - best_quad[..., 1]
    x_diff = best_quad[..., 6] + best_quad[..., 4] - best_quad[..., 2] - best_quad[..., 0]
    angle = tf.atan2(y_diff, x_diff)
    temp_quads = tf.reshape(filter_quads, [-1, 4, 2])

    rot_x = tf.stack([tf.cos(angle), -tf.sin(angle)], -1)
    rot_y = tf.stack([tf.sin(angle), tf.cos(angle)], -1)
    rot_mat = tf.stack([rot_x, rot_y], -2)
    # rot_mat_repeat = tf.stack([rot_mat, rot_mat, rot_mat, rot_mat], -2)
    rot_quads = tf.einsum('jk,lij->lik', rot_mat, temp_quads)
    rot_quads = tf.reshape(rot_quads, [-1, 8])
    rot_boxes = tf.stack(
        [tf.minimum(tf.minimum(rot_quads[..., 0], rot_quads[..., 2]), tf.minimum(rot_quads[..., 4], rot_quads[..., 6])),
         tf.minimum(tf.minimum(rot_quads[..., 1], rot_quads[..., 3]), tf.minimum(rot_quads[..., 5], rot_quads[..., 7])),
         tf.maximum(tf.maximum(rot_quads[..., 0], rot_quads[..., 2]), tf.maximum(rot_quads[..., 4], rot_quads[..., 6])),
         tf.maximum(tf.maximum(rot_quads[..., 1], rot_quads[..., 3]),
                    tf.maximum(rot_quads[..., 5], rot_quads[..., 7]))],
        axis=-1)

    nms_indices = tf.image.non_max_suppression(boxes=rot_boxes,
                                               scores=filter_score,
                                               max_output_size=max_boxes,
                                               iou_threshold=nms_thresh, name='nms_indices')
    return nms_indices


def gpu_nms(quads, scores, num_classes, max_boxes=50, score_thresh=0.5, nms_thresh=0.5, apply_rotate=True):
    """
    Perform NMS on GPU using TensorFlow.
    params:
        boxes: tensor of shape [1, 10647, 4] # 10647=(13*13+26*26+52*52)*3, for input 416*416 image
        scores: tensor of shape [1, 10647, num_classes], score=conf*prob
        num_classes: total number of classes
        max_boxes: integer, maximum number of predicted boxes you'd like, default is 50
        score_thresh: if [ highest class probability score < score_threshold]
                        then get rid of the corresponding box
        nms_thresh: real value, "intersection over union" threshold used for NMS filtering
    """

    boxes_list, label_list, score_list, quads_list = [], [], [], []
    max_boxes = tf.constant(max_boxes, dtype='int32')

    quads = tf.reshape(quads, [-1, 8])
    # boxes = tf.stack([tf.minimum(quads[..., 0],quads[..., 2]), tf.minimum(quads[..., 1],quads[..., 7]), tf.maximum(quads[..., 4],quads[..., 6]), tf.maximum(quads[..., 3],quads[..., 5])], axis=-1)
    boxes = tf.stack([tf.minimum(tf.minimum(quads[..., 0],quads[..., 2]), tf.minimum(quads[..., 4],quads[..., 6])),
                      tf.minimum(tf.minimum(quads[..., 1],quads[..., 3]), tf.minimum(quads[..., 5],quads[..., 7])),
                      tf.maximum(tf.maximum(quads[..., 0],quads[..., 2]), tf.maximum(quads[..., 4],quads[..., 6])),
                      tf.maximum(tf.maximum(quads[..., 1],quads[..., 3]), tf.maximum(quads[..., 5],quads[..., 7]))],
                     axis=-1)
    # boxes = tf.gather(quads, indices = [0, 1, 4, 5], axis=-1)
    # since we do nms for single image, then reshape it
    boxes = tf.reshape(boxes, [-1, 4]) # '-1' means we don't konw the exact number of boxes
    score = tf.reshape(scores, [-1, num_classes])
    labels = tf.argmax(score, axis=1)
    score = tf.reduce_max(score, axis=1)
    # score = tf.reduce_max(score[:,0], score[:,1])
    #
    # print("boxes size", tf.size(boxes))
    # print("quads size", tf.size(quads))

    # Step 1: Create a filtering mask based on "box_class_scores" by using "threshold".
    mask = tf.greater_equal(score, tf.constant(score_thresh))
    # Step 2: Do non_max_suppression for each class
    # for i in range(num_classes):
        # Step 3: Apply the mask to scores, boxes and pick them out
    filter_labels = tf.boolean_mask(labels, mask)
    filter_boxes = tf.boolean_mask(boxes, mask)
    filter_score = tf.boolean_mask(score, mask)
    filter_quads = tf.boolean_mask(quads, mask)

    if apply_rotate:
        nms_indices = tf.cond(tf.greater(tf.shape(filter_score)[0], 0),
                              lambda:rot_nms(filter_score, filter_quads, max_boxes, nms_thresh),
                              lambda:tf.image.non_max_suppression(boxes=filter_boxes,
                                                       scores=filter_score,
                                                       max_output_size=max_boxes,
                                                       iou_threshold=nms_thresh, name='nms_indices')
                              )
    else:
        nms_indices = tf.image.non_max_suppression(boxes=filter_boxes,
                                                   scores=filter_score,
                                                   max_output_size=max_boxes,
                                                   iou_threshold=nms_thresh, name='nms_indices')



    label_list.append(tf.gather(filter_labels, nms_indices))
    boxes_list.append(tf.gather(filter_boxes, nms_indices))
    score_list.append(tf.gather(filter_score, nms_indices))
    quads_list.append(tf.gather(filter_quads, nms_indices))

    boxes = tf.concat(boxes_list, axis=0)
    score = tf.concat(score_list, axis=0)
    label = tf.concat(label_list, axis=0)
    quad = tf.concat(quads_list, axis=0)

    return boxes, score, label, quad


def nms_rotate_cpu(boxes, scores, iou_threshold, max_output_size):

    keep = []

    order = scores.argsort()[::-1]
    num = boxes.shape[0]

    suppressed = np.zeros((num), dtype=np.int)

    for _i in range(num):
        if len(keep) >= max_output_size:
            break

        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        r1 = ((boxes[i, 0], boxes[i, 1]), (boxes[i, 2], boxes[i, 3]), boxes[i, 4])
        area_r1 = boxes[i, 2] * boxes[i, 3]
        for _j in range(_i + 1, num):
            j = order[_j]
            if suppressed[i] == 1:
                continue

            if np.sqrt((boxes[i, 0] - boxes[j, 0])**2 + (boxes[i, 1] - boxes[j, 1])**2) > (boxes[i, 2] + boxes[j, 2] + boxes[i, 3] + boxes[j, 3]):
                inter = 0.0
            else:

                r2 = ((boxes[j, 0], boxes[j, 1]), (boxes[j, 2], boxes[j, 3]), boxes[j, 4])
                area_r2 = boxes[j, 2] * boxes[j, 3]
                inter = 0.0

                try:
                    int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]

                    if int_pts is not None:
                        order_pts = cv2.convexHull(int_pts, returnPoints=True)

                        int_area = cv2.contourArea(order_pts)

                        inter = int_area * 1.0 / (area_r1 + area_r2 - int_area + cfgs.EPSILON)

                except:
                    """
                      cv2.error: /io/opencv/modules/imgproc/src/intersection.cpp:247:
                      error: (-215) intersection.size() <= 8 in function rotatedRectangleIntersection
                    """
                    # print(r1)
                    # print(r2)
                    inter = 0.9999

            if inter >= iou_threshold:
                suppressed[j] = 1

    return np.array(keep, np.int64)

def nms_rotate_cpu_tf(boxes, scores, iou_threshold, max_output_size):

    #keep = []
    keep = tf.placeholder(shape=[max_output_size], dtype=tf.float32)

    #order = scores.argsort()[::-1]
    order = tf.argsort(scores, direction='DESCENDING')

    num = boxes.shape[0]

    #suppressed = np.zeros((num), dtype=np.int)
    suppressed = tf.zeros([num], tf.int32)


    for _i in range(num):
        if len(keep) >= max_output_size:
            break

        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        r1 = ((boxes[i, 0], boxes[i, 1]), (boxes[i, 2], boxes[i, 3]), boxes[i, 4])
        area_r1 = boxes[i, 2] * boxes[i, 3]
        for _j in range(_i + 1, num):
            j = order[_j]
            if suppressed[i] == 1:
                continue

            if np.sqrt((boxes[i, 0] - boxes[j, 0])**2 + (boxes[i, 1] - boxes[j, 1])**2) > (boxes[i, 2] + boxes[j, 2] + boxes[i, 3] + boxes[j, 3]):
                inter = 0.0
            else:

                r2 = ((boxes[j, 0], boxes[j, 1]), (boxes[j, 2], boxes[j, 3]), boxes[j, 4])
                area_r2 = boxes[j, 2] * boxes[j, 3]
                inter = 0.0

                try:
                    int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]

                    if int_pts is not None:
                        order_pts = cv2.convexHull(int_pts, returnPoints=True)

                        int_area = cv2.contourArea(order_pts)

                        inter = int_area * 1.0 / (area_r1 + area_r2 - int_area + cfgs.EPSILON)

                except:
                    """
                      cv2.error: /io/opencv/modules/imgproc/src/intersection.cpp:247:
                      error: (-215) intersection.size() <= 8 in function rotatedRectangleIntersection
                    """
                    # print(r1)
                    # print(r2)
                    inter = 0.9999

            if inter >= iou_threshold:
                suppressed[j] = 1

    return np.array(keep, np.int64)


def nms_rotate_gpu(boxes_list, scores, iou_threshold, use_angle_condition=False, angle_gap_threshold=0, device_id=0):
    if use_angle_condition:
        x_c, y_c, w, h, theta = tf.unstack(boxes_list, axis=1)
        boxes_list = tf.transpose(tf.stack([x_c, y_c, w, h, theta]))
        det_tensor = tf.concat([boxes_list, tf.expand_dims(scores, axis=1)], axis=1)
        keep = tf.py_func(rotate_gpu_nms,
                          inp=[det_tensor, iou_threshold, device_id],
                          Tout=tf.int64)
        return keep
    else:
        x_c, y_c, w, h, theta = tf.unstack(boxes_list, axis=1)
        boxes_list = tf.transpose(tf.stack([x_c, y_c, w, h, theta]))
        det_tensor = tf.concat([boxes_list, tf.expand_dims(scores, axis=1)], axis=1)
        keep = tf.py_func(rotate_gpu_nms,
                          inp=[det_tensor, iou_threshold, device_id],
                          Tout=tf.int64)
        keep = tf.reshape(keep, [-1])
        return keep


if __name__ == '__main__':
    boxes = np.array([[50, 50, 100, 100, 0],
                      [60, 60, 100, 100, 0],
                      [50, 50, 100, 100, -45.],
                      [200, 200, 100, 100, 0.]])

    scores = np.array([0.99, 0.88, 0.66, 0.77])

    keep = nms_rotate(tf.convert_to_tensor(boxes, dtype=tf.float32), tf.convert_to_tensor(scores, dtype=tf.float32),
                      0.7, 5)

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    with tf.Session() as sess:
        print(sess.run(keep))

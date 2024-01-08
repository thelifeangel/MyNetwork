import json
import os
import sys
import time
import cv2
import pickle
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')  # 使用适合你系统的后端
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ROOT_DIR = os.path.abspath("../../")

# To find local version of the library
sys.path.append(ROOT_DIR)
import mrcnn.model as modellib
from mrcnn import utils
from pathlib import Path
import coco
from collections import namedtuple, defaultdict, deque, Counter
from sklearn.metrics import confusion_matrix, classification_report
import itertools

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
from tqdm import tqdm
from keras import backend as K


session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1,
                              allow_soft_placement=True)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

ap = argparse.ArgumentParser()
ap.add_argument(
    "-d",
    "--dataset",
    help="Path to dataset folder that contained json annotation file("
         "via_region_data.json)",
    required=False,
    default='D:\PycharmProjects\Mask_RCNN-master\cat_dog'
)
ap.add_argument("-c", "--checkpoint", help="Path to checkpoint",
                required=False,
                default=r'D:\PycharmProjects\Mask_RCNN-master\logs\coco20231123T1421\mask_rcnn_coco_0069.h5')
ap.add_argument(
    "-m",
    "--mode",
    help="CPU or GPU, default is CPU (/cpu:0 or /gpu:0)",
    default="/gpu:0",
)
args = vars(ap.parse_args())

# trained weights
MODEL_WEIGHTS_PATH = args["checkpoint"]
DATASET_DIR = args["dataset"]
DEVICE = args["mode"]

Detection = namedtuple("Detection", ["gt_class", "pred_result", "overlapscore"])


## dice coefficient: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
def dice_coef(y_true, y_pred):
    intersec = y_true * y_pred
    union = y_true + y_pred
    if intersec.sum() == 0:
        dice_coef = 0
    else:
        dice_coef = round(intersec.sum() * 2 / union.sum(), 2)
    return dice_coef


def coeff_per_image(metric_name, pred_mask, gt_mask):
    #计算每张图片的dice值
    gt_sum = np.zeros((gt_mask.shape[0], gt_mask.shape[1]))
    for gt_num in range(gt_mask.shape[2]):  # as there may be over one mask per class
        gt = gt_mask[:, :, gt_num]
        gt_sum = gt_sum + gt
    gt_union = (gt_sum > 0).astype(int)
    mask_sum = np.zeros((pred_mask.shape[0], pred_mask.shape[1]))
    for mask_num in range(pred_mask.shape[2]):
        mask = pred_mask[:, :, mask_num]
        mask_sum = mask_sum + mask
    mask_union = (mask_sum > 0).astype(int)
    if metric_name == 'jaccard index':
        coeff_dict=jaccard_coef(mask_union, gt_union) #mask_union是预测mask求和>0
    elif metric_name == 'dice':
        coeff_dict=dice_coef(mask_union,gt_union)
    return coeff_dict

# kernel = np.ones((3, 3), np.uint8)
def compute_batch_detections_by_class(model, image_ids,cls_id,class_num):
    # Compute VOC-style Average Precision
    # imgae_ids是某一个类的所有image_id
    detections = []
    dice_dic = {}
    all_gt_match=None
    all_pred_match=None
    for image_id in tqdm(image_ids):
        # Load image
        image, _, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(
            dataset, config, image_id, use_mini_mask=False
        )
        cls_index = np.where(gt_class_id == cls_id)[0]  #只取每张图片指定类别的那一部分
        gt_class_id=gt_class_id[cls_index]
        gt_bbox=gt_bbox[cls_index]
        gt_mask=gt_mask[:,:, cls_index]
        # w, h = image.shape[1], image.shape[0]
        # if w < h:
        #     w = h
        # else:
        #     h = w
        # image = cv2.addWeighted(image, alpha, np.zeros(img.shape, img.dtype), 0, beta)
        # image = cv2.resize(image, (w, h),interpolation=cv2.INTER_CUBIC)

        # Run object detection
        results = model.detect([image], verbose=0)
        # Compute AP
        r = results[0]
        #便利
        cls_index=np.where(r['class_ids']==cls_id)[0]
        gt_match, pred_match,overlaps=utils.compute_matches(gt_bbox,
            gt_class_id,
            gt_mask,
            r["rois"][cls_index],#只取每张图片指定类别的那一部分结果
            r["class_ids"][cls_index],
            r["scores"][cls_index],
            r["masks"][:,:,cls_index],
            iou_threshold=0.5)
        if all_gt_match is None:
            all_gt_match=gt_match
            all_pred_match=pred_match
        else:
            all_gt_match=np.concatenate((all_gt_match,gt_match))
            all_pred_match=np.concatenate((all_pred_match,pred_match))
        # APs.append(AP)
        # PRs.append(precisions)
        # REs.append(recalls)

        # list_overlaps
        detection = Detection(gt_class_id, r, overlaps)
        detections.append(detection)
        dice_dic[image_id] = coeff_per_image('dice', r['masks'], gt_mask) #为每张图片计算dice指标
    AP, precisions, recalls=utils.compute_ap_by_class(all_pred_match,all_gt_match)
    try:
        print("[INFO] Dice Coefficient: ", )
        # dice_path = os.getcwd() + "dice_coeff.p"
        # pickle.dump(dice_dic, open(dice_path, 'wb'))
        print(dice_dic)
    except Exception:
        pass

    return detections,AP,precisions,recalls




config = coco.CocoConfig()
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.8
    DETECTION_NMS_THRESHOLD = 0.33
    ROI_POSITIVE_RATIO = 0.33
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    BACKBONE = "resnet101"
    # IMAGE_MAX_DIM = 1920
    # POST_NMS_ROIS_INFERENCE = 1500
    RPN_NMS_THRESHOLD = 0.8

    """ THIS CONFIGURATION GAVE mAP: 0.245
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.5
    DETECTION_NMS_THRESHOLD = 0.66
    ROI_POSITIVE_RATIO = 0.66
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    BACKBONE = "resnet101"    
    #IMAGE_MAX_DIM = 1920
    POST_NMS_ROIS_INFERENCE = 5000
    PRE_NMS_LIMIT = 8000    
    """

if __name__ == "__main__":
    start = time.time()
    config = InferenceConfig()
    config.display()
    class_num=config.NUM_CLASSES-1
    # Load validation dataset
    dataset = coco.CocoDataset()
    coco=dataset.load_coco(DATASET_DIR,return_coco=True,subset= "val")
    # Must call before using the dataset
    dataset.prepare()
    print(
        "Number of Images: {}\nClasses: {}".format(
            len(dataset.image_ids), dataset.class_names[1:]
        )
    )
    # Create model in inference mode
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(
            mode="inference", model_dir=MODEL_WEIGHTS_PATH, config=config
        )

    # Load weights
    print("[INFO] Loading weights ", MODEL_WEIGHTS_PATH)
    model.load_weights(MODEL_WEIGHTS_PATH, by_name=True)
    Aps=[]
    for cls_id in dataset.class_ids[1:]:
        img_ids=list(set(coco.catToImgs[cls_id]))
        img_ids=[dataset.image_from_source_map['coco.{}'.format(img_id)] for img_id in img_ids] #转换为内置ID
        result_detection, AP, precision, recall = compute_batch_detections_by_class(
            model,img_ids,cls_id,class_num
        )
        Aps.append(AP)
        print("[INFO] AP IoU=50: {:.4f} for class {}".format(AP,dataset.class_names[cls_id]))
        plt.plot( recall, precision, 'b', label='PR')
        plt.title('precision-recall curve class:{}'.format(dataset.class_names[cls_id]))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        pic_name=f'class_{dataset.class_names[cls_id]}'+'.png'
        # pic_name='class:{}.png'.format(dataset.class_names[cls_id])
        plt.savefig(pic_name)
        plt.show()
    print("[INFO] mAP IoU=50: {:.4f}".format(np.mean(Aps)))
    # try:
    #     cm = confusion_matrix(name_true, name_pred, labels=class_names)
    #     print_cm(cm, class_names)
    #     print("\nClassification Report\n")
    #     print(classification_report(name_true, name_pred))
    # except:
    #     print("None!")
    #
    # execution_time = (time.time() - start) / 60
    # print("\n[INFO] Execution Time: {} minutes".format(execution_time))

import os
import sys
import time
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')  # 使用适合你系统的后端
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ROOT_DIR = os.path.abspath("../../")
# To find local version of the library
sys.path.append(ROOT_DIR)
import mrcnn.model as modellib
from mrcnn import utils
import coco
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import argparse
from tqdm import tqdm
from keras import backend as K
import pandas as pd
import numpy as np
import os
import my_utils
from sklearn.metrics import confusion_matrix, classification_report

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
ap.add_argument("-c", "--checkpoint", help="Path to checkpoint", required=False,
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

def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    print("\n[INFO] Confusion Matrix")

    columnwidth = max([len(x) for x in labels] + [5])
    empty_cell = " " * columnwidth

    # Begin CHANGES
    fst_empty_cell = (columnwidth - 3) // 2 * " " + "t/p" + (columnwidth - 3) // 2 * " "

    if len(fst_empty_cell) < len(empty_cell):
        fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
    # Print header
    print("    " + fst_empty_cell, end=" ")
    # End CHANGES

    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")

    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()

# kernel = np.ones((3, 3), np.uint8)
def compute_batch_detections(model, image_ids):
    # Compute VOC-style Average Precision
    gt_tot = np.array([])
    pred_tot = np.array([])
    mAP_ = []
    for image_id in tqdm(image_ids):
        # Load image
        image, _, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(
            dataset, config, image_id, use_mini_mask=False
        )
        # Run object detection
        results = model.detect([image], verbose=0)
        # Compute AP
        r = results[0]
        #便利
        gt, pred = my_utils.gt_pred_lists(gt_class_id, gt_bbox, r['class_ids'], r['rois'])
        gt_tot = np.append(gt_tot, gt)
        pred_tot = np.append(pred_tot, pred)
        AP_, precision_, recall_, overlap_=utils.compute_ap(gt_bbox,
            gt_class_id,
            gt_mask,
            r["rois"],#只取每张图片指定类别的那一部分结果
            r["class_ids"],
            r["scores"],
            r["masks"],
            iou_threshold=0.5)
        mAP_.append(AP_)
        # check if the vectors len are equal
        # print("the actual len of the gt vect is : ", len(gt_tot))
        # print("the actual len of the pred vect is : ", len(pred_tot))
        # print("Average precision of this image : ", AP_)
        # print("The actual mean average precision for the whole images (matterport methode) ", sum(mAP_) / len(mAP_))
        # print("Ground truth object : " + [dataset.class_names[i]] for i in gt)
        # print("Predicted object : " + [dataset.class_names[i]] for i in pred)
    gt_tot = gt_tot.astype(int)
    pred_tot = pred_tot.astype(int)
    # save the vectors of gt and pred
    save_dir = "output"
    gt_pred_tot_json = {"gt_tot": gt_tot, "pred_tot": pred_tot}
    df = pd.DataFrame(gt_pred_tot_json)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    df.to_json(os.path.join(save_dir, "gt_pred_test.json"))

    # tp, fp, fn = my_utils.plot_confusion_matrix_from_data(gt_tot, pred_tot, columns==[]fz=10, figsize=(10, 10), lw=0.3)
    return gt_tot,pred_tot


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
    # Aps=[]
    # for cls_id in dataset.class_ids[1:]:
    #     img_ids=list(set(coco.catToImgs[cls_id]))
    #     img_ids=[dataset.image_from_source_map['coco.{}'.format(img_id)] for img_id in img_ids] #转换为内置ID
    gt_tot,pred_tot = compute_batch_detections(
            model,dataset.image_ids
    )
    cm = confusion_matrix(np.array(gt_tot), np.array(pred_tot))
    print_cm(cm, dataset.class_names)
    print("\nClassification Report\n")
    print(classification_report(gt_tot, pred_tot))


    # tp, fp, fn = my_utils.plot_confusion_matrix_from_data(gt_tot, pred_tot, columns = dataset.class_names,
    # fz = 10, figsize = (10, 10), lw = 0.3)
    # print('tp: {} fp:{} fn:{}'.format(tp,fp,fn))

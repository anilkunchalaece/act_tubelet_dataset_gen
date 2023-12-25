# person_detector

from torchvision.io.image import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn,FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import torchvision.transforms as transforms

import torch
import cv2

import os
import numpy as np
import json

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


"""
pred_out -> predictions (list of detection out consist of gpu/cpu tensors)
         -> keys are boxes, labels, scores
"""
def process_predictions(image_names, pred_out, label_to_filter=1):
    out = []
    for idx, pred in enumerate(pred_out) :
        bbox = pred['boxes'].detach().cpu().numpy()
        labels = pred['labels'].detach().cpu().numpy()
        scores = pred['scores'].detach().cpu().numpy()
        # get all the person detections
        labels_index = np.transpose(np.argwhere(labels == 1))[0]
        bbox_filtered = bbox[labels_index].astype('int32').tolist()
        if len(bbox_filtered) > 0 :
            out.append(
                {
                    "image_name" : os.path.basename(image_names[idx].split(".")[0]),
                    "bbox" : bbox_filtered[0]
                }
            )
    return out



"""
run inference on images using pre-trained models
batches -> list of list of images [[im1,im2],[im3,im4],[im5,im6]]
"""
def run_inference(batches) :
    torch.cuda.empty_cache()
    # weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    preprocess = weights.transforms()
    # model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.75)
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights, box_score_thresh=0.90)
    model.to(device)
    model.eval()
    out_data = []
    for batch in batches :
        c_batch = [preprocess(read_image(x)).to(device) for x in batch]
        predictions = model(c_batch) # we are assuming only 
        bbox_info = process_predictions(batch,predictions)
        out_data.extend(bbox_info)
    
    with open("bbox_predictions.json","w") as fw :
        json.dump(out_data,fw)
    return out_data

# split the images into batches
def generate_batches(dir_name,batch_size=10):
    all_files = sorted(os.listdir(dir_name))
    all_batches = []
    for idx in range(0, len(all_files) , batch_size) :
        c_files = all_files[idx:idx+batch_size]
        all_batches.append([os.path.join(dir_name,f) for f in c_files])
    return all_batches

# main function binding other fcn's
def get_person_bboxes_from_dir(dir_path):
    batches = generate_batches(dir_path)
    bbox_detections = run_inference(batches)
    
    out = {}
    for each_bbox in bbox_detections :
        out[each_bbox["image_name"]] = each_bbox["bbox"]

    return out


if __name__ == "__main__" :
    # src_dir = "ex_jrdb-act_data/images/image_0/tressider-2019-04-26_2"
    src_dir = "/home/akunchala/Documents/PhDStuff/action_tracklet_parser/kth_frames/handclapping/person01_handclapping_d2_uncomp_1/"
    bboxes = get_person_bboxes_from_dir(src_dir)
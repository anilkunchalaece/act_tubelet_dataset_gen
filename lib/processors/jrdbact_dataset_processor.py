"""
Parse JRDB-ACT annotations

JRDB-ACT dataset consist of frame level annotations as a json file.
Each json file consist of annotations for all the images in a single directory

Labels of interest are
- label_id -> pedestrian label id - unique for each pedestrian (format "pedestrian:XX", where XX is pedestrian ID)
- action_label -> activity label - format (dict(action_label_name : action_label_id))
- box -> bounding box information - format(x_top, y_top, width, height)
- file_id -> file name 

Steps to extract tracklets
1. Get the Unique person Id's in the given JSON file 
2. for each person get the list of activities, respective frames and bboxes
"""

import os
import json
from loguru import logger
import pybboxes
import random

import cv2
import multiprocessing

class JRDBActDatasetProcessor() :
    def __init__(self, config) :
        logger.info(F"initialized JRDBAct Dataset processor")
        self.config = config
        keys_to_check = ['labels_dir', 'src_dir', 'bbox_info', 'data_format']
        for k in keys_to_check :
            assert(self.config.get(k,None) != None), F"{k} is missing in the JRDBAct config"
        
    
    def __call__(self):
        return self.process_jrdb_annotations()
    
    def process_jrdb_annotations(self) :
        p_groups = {}
        for f_name in os.listdir(self.config['labels_dir']) :
            logger.info(F"processing {f_name}")
            f_name = os.path.join(self.config['labels_dir'],f_name)

            with open(f_name) as fd :
                file_data = json.load(fd)
            
            for _, image_data in file_data["labels"].items() : # this is a annotations for each image with mutliple people
                for each_annotation in image_data :
                    p_id = each_annotation["label_id"].split(":")[-1]
                    activity_label = [ x.lower() for x in each_annotation.get("action_label",{}).keys()]

                    if len(activity_label) == 0 :
                        continue # skip there there are no activities 
                
                    frame_no = each_annotation["file_id"]
                    bbox = each_annotation["box"]
                    if bbox[2] < 10 or bbox[3] < 10 : # 
                        continue # don't process bboxes with height or width less than 10

                    for act in activity_label :
                        data_of_intrest = {
                            "file_name" : os.path.basename(f_name),
                            "person_id" : p_id,
                            "activity" : act,
                            "frame_no" : frame_no.replace(".jpg",""),
                            "bbox" : pybboxes.convert_bbox(bbox, from_type='coco', to_type='voc')
                        }
                        
                        if(p_groups.get(F"{p_id}_{act}",None) == None) :
                            p_groups[F"{p_id}_{act}"] = [data_of_intrest]
                        else :
                            p_groups[F"{p_id}_{act}"].append(data_of_intrest)
        all_p_ids = list(p_groups.keys())

        p_all_activities = []
        for _p_id in all_p_ids :
            c_activity = [p_groups[_p_id][0]]
            for act in p_groups[_p_id][1:] :
                # print(act["frame_no"]) 
                if act['activity'] == c_activity[-1]['activity'] and \
                        int(act["frame_no"]) - int(c_activity[-1]["frame_no"]) < 3:
                    c_activity.append(act)
                else :
                    p_all_activities.append(c_activity)
                    c_activity = [act]
            p_all_activities.append(c_activity)
        logger.info(F"total no of activities are {len(p_all_activities)}")

        # converting the activities suitable for getting the tubelets
        out = {}
        for each_act in p_all_activities :
            video_name = each_act[0]['file_name'].split(".")[0]
            each_act = sorted(each_act, key= lambda x: int(x['frame_no']))
            bbox_info = {}
            for _act in each_act :
                bbox_info[F"img_{int(_act['frame_no']):05d}"] = _act['bbox']

            act_info = {
                "start_f_no" : each_act[0]['frame_no'],
                "end_f_no" : each_act[-1]["frame_no"],
                "activity" : each_act[0]["activity"],
                "file_name" : video_name,
                "src_dir" : os.path.join(self.config['src_dir'],F"image_{video_name.split('_')[-1][-1]}","_".join(video_name.split("_")[:-1])),
                "bbox_info" : bbox_info
            }

            if out.get(video_name,None) != None :
                out[video_name].append(act_info)
            else :
                out[video_name] = [act_info]
        
        # with open("jrdb-act-ex.json","w") as fw :
        #     json.dump(out,fw)
        return out


    def get_train_test_split(self, dataset_dir) :
        """
        JRDB-Act doesn't have train and validation test.
        So we take the all the generated tubelets and split them into 75% (train) to 25% (test)

        parameters
        dataset_dir -> root dataset with jrbdact tubeletes
        """
        # get all the tubelets
        all_tubelets = [ x for x in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir,x))]

        # filter the jrdbact tubelets
        jrdbact_tubelets = [x for x in all_tubelets if x.split("-")[0]=="JRDBACT"]
        all_classes = list(set([x.split("-")[2] for x in jrdbact_tubelets]))
        
        logger.info(F"Total number of jrdbact tubelets are {len(jrdbact_tubelets)} with {len(all_classes)} => classes {all_classes}")
        
        train_files = []
        test_files = []

        for cls in all_classes :
            cls_tubelets = [x for x in jrdbact_tubelets if x.split("-")[2] == cls]
            no_of_test_samples = int(len(cls_tubelets) * 0.25) # get the 25% for test 
            random.shuffle(cls_tubelets)
            cls_test_files = cls_tubelets[:no_of_test_samples]
            cls_train_files = cls_tubelets[no_of_test_samples:]
            train_files.extend(cls_train_files)
            test_files.extend(cls_test_files)
        
        logger.info(F"train samples {len(train_files)}, test samples {len(test_files)}")

        return {
            "train" : train_files,
            "test" : test_files
        }


        
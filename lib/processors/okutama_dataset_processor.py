"""
Parsing the tubelets using Single Action labels (3840x2160)
Ref - http://okutama-action.org/
Original annotations consist of txt files 
Each line contains 10+ columns, separated by spaces
    track_id, xmin, ymin, xmax, ymax, frame, lost, occluded, generated, label, action

There are three label files for each videos: 
1. MultiActionLabels: labels for multi-action detection task. 
2. SingleActionLabels: labels for single-action detection task which has been created from the multi-action detection labels. In both of these files, all rows with the same “Track ID” belong to the same person for 180 frames. Then the person gets a new ID for the next 180 frames.
3. SingleActionTrackingLabels: same labels as 2, but here the ID’s are consistent. This means that each person has a unique ID in the video but will get a new one if he/she is absent for more than 90 frames.

This script considers the SingleActionLabels
"""

import os
import json
from loguru import logger
import random

class OkutamaDatasetProcessor() :
    def __init__(self, config):
        logger.info(F"initalized the okutama dataset processor")
        self.config = config
        keys_to_check = ['labels_dir', 'src_dir', 'bbox_info', 'data_format']
        for k in keys_to_check :
            assert(self.config.get(k,None) != None), F"{k} is missing in the OKUTAMA config"
        self.classes_to_include = config.get('classes_to_include',None)
        
    def __call__(self):
        return self.process_okumata_annotations()
    
    def process_okumata_annotations(self) :
        all_files = [ os.path.join(self.config['labels_dir'],x) for x in os.listdir(self.config['labels_dir'])]
        all_activity_data = {}

        for f_name in all_files :
            logger.info(F"processing {os.path.basename(f_name)}")
            with open(f_name) as fd :
                data = fd.readlines()
            
            unique_ids = list(set([x.split(" ")[0] for x in data]))

            for tid in unique_ids :
                tid_annotations = [x for x in data if x.split(" ")[0] == tid]
                unique_acts = list(set([x.split(" ")[-1] for x in tid_annotations]))
                #process each activity for given tid
                for act in unique_acts :
                    tid_act_annotations = [x for x in tid_annotations if x.split(" ")[-1] == act]
                    tid_act_frames = [int(x.split(" ")[5]) for x in tid_act_annotations]
                    start_f_no = min(tid_act_frames)
                    end_f_no = max(tid_act_frames)

                    bbox_info = {}
                    # get the data for single acitivty based on annotations
                    for ann in tid_act_annotations :
                        ann = ann.split(" ")
                        bbox_info[F"img_{int(ann[5]):05d}"] = [int(x) for x in ann[1:5]]

                    video_name = [ x for x in os.listdir(self.config["src_dir"]) if os.path.splitext(x)[0] == os.path.splitext(os.path.basename(f_name))[0]][0]
                    activity = act.strip().replace('"','').replace("/","_").replace("\\","")
                    if self.classes_to_include is not None and activity not in self.classes_to_include :
                        continue # skip if activity is not in classes_to_include
                    act_info = {
                        "start_f_no" : start_f_no,
                        "end_f_no" : end_f_no,
                        "activity" : activity,
                        "file_name" : video_name,
                        "bbox_info" : bbox_info
                    }

                    if all_activity_data.get(video_name,None) != None :
                        all_activity_data[video_name].append(act_info)
                    else :
                        all_activity_data[video_name] = [act_info]
        return all_activity_data

    def get_train_test_split(self, dataset_dir) :
        """
        Okutama doesn't have train and validation test.
        So we take the all the generated tubelets and split them into 75% (train) to 25% (test)

        parameters
        dataset_dir -> root dataset with okutama tubeletes
        """
        # get all the tubelets
        all_tubelets = [ x for x in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir,x))]

        # filter the jrdbact tubelets
        okutama_tubelets = [x for x in all_tubelets if x.split("-")[0]=="OKUTAMA"]
        all_classes = list(set([x.split("-")[2] for x in okutama_tubelets]))
        
        logger.info(F"Total number of okutama tubelets are {len(okutama_tubelets)} with {len(all_classes)} => classes {all_classes}")
        
        train_files = []
        test_files = []

        for cls in all_classes :
            cls_tubelets = [x for x in okutama_tubelets if x.split("-")[2] == cls]
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
                    



            

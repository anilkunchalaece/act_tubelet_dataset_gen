import re
import json
import os
from typing import Any
import cv2
import shutil
import concurrent.futures
from functools import partial
import copy

from loguru import logger

from lib.utils import utils

class ViratDatasetProcessor() :
    def __init__(self,config):
        logger.info("initialised virat data processor")
        assert(config.get('processed_annotations_dir',None) != None), "processed_annotations is missing in VIRAT config"
        assert(config.get('src_dir',None) != None), "src_dir is missing in VIRAT config"
        assert(config.get('bbox_info',None) != None), "bbox_info is missing in VIRAT config"
        assert(config.get('data_format',None) != None), "data_format is missing in VIRAT config"
        self.config = config
        self.annotation_data = {}
    
    def __call__(self) :

        self.src_dir = self.config.get('src_dir')
        processed_annotations_dir = self.config.get('processed_annotations_dir')
        assert(utils.check_if_dir_exists(processed_annotations_dir)), F"{processed_annotations_dir} not found for VIRAT"
        return self.process_virat_annotation_files(processed_annotations_dir)

    def update_annotations(self, key, value) :
        self.annotation_data[key] = value

    def process_virat_annotation_files(self, annotation_dir) :
        train_files_dir = os.path.join(annotation_dir,"train")
        valid_files_dir = os.path.join(annotation_dir,"validate")
        assert(utils.check_if_dir_exists(train_files_dir)), "VIRAT train files dir {train_files_dir} not found"
        assert(utils.check_if_dir_exists(valid_files_dir)), "VIRAT valid files dir {valid_files_dir} not found"

        train_files = [ os.path.join(train_files_dir,x) for x in os.listdir(train_files_dir) if x.split(".")[-1] == "json"]
        valid_files = [ os.path.join(valid_files_dir,x) for x in os.listdir(valid_files_dir) if x.split(".")[-1] == "json"]
        
        all_files = train_files + valid_files # combine both train and validation files

        for f_name in all_files :
            self.process_each_file(f_name)
        # multi-processing is not working - USING multi-threading instead
        # with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        #     executor.map(self.process_each_file, all_files)
        return self.annotation_data

    def get_train_test_split(self) :
        annotation_dir = self.config.get('processed_annotations_dir')
        train_files_dir = os.path.join(annotation_dir,"train")
        valid_files_dir = os.path.join(annotation_dir,"validate")

        train_files = [ x.split(".")[0] for x in os.listdir(train_files_dir) if x.split(".")[-1] == "json"]
        valid_files = [ x.split(".")[0] for x in os.listdir(valid_files_dir) if x.split(".")[-1] == "json"]
        return {
            "train" : train_files,
            "test" : valid_files
        }
        
    
    def process_each_file(self,file_path) :
        if not utils.check_if_file_exists(file_path) :
            logger.info(F"{file_path} not exists, skipping")
            return
        
        logger.info(F"processing {os.path.basename(file_path)} ..")

        with open(file_path) as fd :
            data = json.load(fd)
        
        # acitivites are sorted with track_id
        # each track_id my contains multiple activities
        activities_in_file = []
        for track_id , track_act_info in data['activity'].items() :
            for each_act in track_act_info :
                frame_range = each_act["tsr0"]
                activity = each_act["activity"]
                file_name = os.path.basename(file_path).split(".")[0]
                bbox_info = {}
                for frame_idx in range(min(frame_range), max(frame_range)) :
                    try :
                        bbox_idx = data["bbox"][track_id]["ts0"].index(frame_idx) # get index from ts0
                        bbox = data["bbox"][track_id]["g0"][bbox_idx]
                        bbox_info[F"img_{frame_idx:05d}"] = [int(x) for x in bbox.split(" ")]
                    except :
                        logger.info(F"Unable to get bbox_info for {frame_idx} of {activity} in {file_name}")
            activities_in_file.append({
                    "start_f_no": min(frame_range),
                    "end_f_no": max(frame_range),
                    "activity": activity,
                    "file_name": file_name,
                    "bbox_info": bbox_info
                    })
        self.update_annotations(os.path.basename(file_path).split(".")[0]+".mp4", activities_in_file)



        

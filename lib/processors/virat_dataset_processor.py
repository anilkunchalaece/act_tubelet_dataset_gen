"""
Parse Virat annotations 

VIRAT annotations has 3 files
1. activities
    Contains activities info based on track_id
2. types
    Contains the detection types i.e persons, vehicles with track_id
3. geom 
    Contains the bbox information based on each detection

Steps to parse VIRAT annotations
1. Get track_id of the person detections using *.types.yml file
2. Based on track_id, sort out the activities of each person from *.activities.yml file
    Filter the activities of intrest in this step. Get the track_id, respective
    start_frame_no and end_frame_no of each activity
3. based on track_id and activities start and end frame, extract 
    the bbox information of each person from *.geom.yml file

outfile will be in the following format
{
    "activity" : {
        "track_id" : [
            {
                "activity" : "Activity Name",
                "tsr0" :  [start_frame_no, end_frame_no]
            }
        ]
    },
    "bbox" : {
        "track_id" : {
            "ts0" : [frame_no_1, frame_no_2],
            "g0" : ["f1_x0 f1_y0 f1_x1 f1_y1", "f1_x0 f1_y0 f1_x1 f1_y1"]
        }
    }
}
"""

import re
import json
import os
from typing import Any
import cv2
import shutil
import concurrent.futures
from functools import partial
import copy
import yaml
import multiprocessing
from functools import partial

from loguru import logger
try :
    from lib.utils import utils
except Exception as e :
    import utils

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
    
    def process_virat_original_annotations_in_dir(self,dir_name):
        all_files = os.listdir(dir_name)
        files_without_ext = list(set([x.split(".")[0] for x in all_files ]))
        # print(files_without_ext)

        # for f_name in files_without_ext :
        #     self.get_actions_per_person(f_name)

        func = partial(self.get_actions_per_person,dir_name)
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        pool.map(func, files_without_ext)
        pool.close()
        pool.join()
        

    def process_virat_annotation_files(self, annotation_dir) :
        train_files_dir = os.path.join(annotation_dir,"train")
        valid_files_dir = os.path.join(annotation_dir,"validate")
        assert(utils.check_if_dir_exists(train_files_dir)), "VIRAT train files dir {train_files_dir} not found"
        assert(utils.check_if_dir_exists(valid_files_dir)), "VIRAT valid files dir {valid_files_dir} not found"

        # process both training and validation files and generate the json files
        self.process_virat_original_annotations_in_dir(train_files_dir)
        self.process_virat_original_annotations_in_dir(valid_files_dir)

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

        # virat train and valid sets are based on video's, so we should pass the video name as it is
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
        print(data["bbox"].keys())
        for track_id , track_act_info in data['activity'].items() :
            for each_act in track_act_info :
                frame_range = each_act["tsr0"]
                activity = each_act["activity"]
                # print(frame_range)
                # print(len(data["bbox"][track_id]["ts0"]))
                file_name = os.path.basename(file_path).split(".")[0]
                bbox_info = {}
                for frame_idx in range(min(frame_range), max(frame_range)) :
                    try :
                        bbox_idx = data["bbox"][F"{track_id}"]["ts0"].index(frame_idx) # get index from ts0
                        bbox = data["bbox"][F"{track_id}"]["g0"][bbox_idx]
                        bbox_info[F"img_{frame_idx:05d}"] = [int(x) for x in bbox.split(" ")]
                    except Exception as e:
                        logger.info(F"Unable to get bbox_info  tracker id {track_id} for {frame_idx} of {activity} in {file_name} failed with {e}")
                        # raise
                activities_in_file.append({
                        "start_f_no": min(frame_range),
                        "end_f_no": max(frame_range),
                        "activity": activity,
                        "file_name": file_name,
                        "bbox_info": bbox_info
                        })
        self.update_annotations(os.path.basename(file_path).split(".")[0]+".mp4", activities_in_file)
        return activities_in_file

    def get_persons_from_types_file(self, fName) :
        with open(fName) as fd :
            data = yaml.safe_load(fd)
        
        person_track_ids = []
        for d in data :
            if 'types' in d :
                # if Person is in cset3
                if 'Person' in d['types']['cset3'] :
                    person_track_ids.append(d['types']['id1'])
        
        return person_track_ids

    def get_bboxes_for_persons(self, geom_file, person_track_ids) :
        person_bboxs = {}
        with open(geom_file) as fd :
            data = yaml.safe_load(fd)
            for d in data :
                if 'geom' in d :
                    if d['geom']['id1'] in person_track_ids :
                        if person_bboxs.get(d['geom']['id1'],None) == None :
                            person_bboxs[d['geom']['id1']] = {
                                'ts0' : [d['geom']['ts0']],
                                'g0' : [d['geom']['g0']]
                            }
                        else :
                            person_bboxs[d['geom']['id1']]['ts0'].append(d['geom']['ts0'])
                            person_bboxs[d['geom']['id1']]['g0'].append(d['geom']['g0'])                 
        return person_bboxs

    def get_actions_per_person(self,dir_name,filename_without_ext) :


        activities_file = os.path.join(dir_name, F"{filename_without_ext}.activities.yml")
        types_file = os.path.join(dir_name,F"{filename_without_ext}.types.yml")
        geom_file = os.path.join(dir_name, F"{filename_without_ext}.geom.yml")
        out_file_name = os.path.join(dir_name, F"{filename_without_ext}.json")

        persons_track_ids = self.get_persons_from_types_file(types_file)
        person_bboxes = self.get_bboxes_for_persons(geom_file,persons_track_ids)
        
        with open(activities_file) as fd :
            data = yaml.safe_load(fd)
        
        person_activities = {}
        for d in data :
            if 'act' in d :
                # check if act2 contains ids
                for actor in d['act']['actors'] :
                    if actor['id1'] in persons_track_ids :
                        if person_activities.get(actor['id1'],None) == None :
                            person_activities[actor['id1']] = [
                                {
                                    'tsr0' : actor['timespan'][0]['tsr0'],
                                    'activity' :  list(d['act']['act2'].keys())[0]
                                }
                            ]
                        else :
                            person_activities[actor['id1']].append(
                                {
                                    'tsr0' : actor['timespan'][0]['tsr0'],
                                    'activity' :  list(d['act']['act2'].keys())[0]
                                }
                            )
        
        out_data = {
            "activity" : person_activities,
            "bbox" : person_bboxes
        }
        
        with open(out_file_name,'w') as fw :
            json.dump(out_data, fw)    



if __name__ == "__main__" :
    with open("/home/akunchala/Documents/PhDStuff/act_tubelet_dataset_gen/generator_config.json") as fd :
        data = json.dump(fd)
    virat_data = ViratDatasetProcessor(data["each_dataset_config"]["VIRAT"])
    act_data = virat_data.process_each_file("/home/akunchala/Documents/z_Datasets/VIRAT/viratannotations/train/VIRAT_S_000002.json")
    print(act_data)


        

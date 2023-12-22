"""
Parsing KTH Dataset
Url - https://www.csc.kth.se/cvap/actions/

Sequences are of the format

<filename> frames <a1_f_start>-<a1_f_end>,<a2_f_start>-<a2_f_end>, ...
<filename> frames <a1_f_start>-<a1_f_end>,<a2_f_start>-<a2_f_end>, ...

Steps to parse the tracklets
1. Parse the sequence file and extract the frames info
2. Conver the videos into frames
3. Run object detector on frames and extract the pedestrian bbox info
    1. Note - We only have one person per frame
4. Using the sequence file info and bbox info generate the tracklets

The output format of the processed file is 
{
    "video_1" : {
        "file_name" : "",
        "frames" : [
            "start_f_no" : 1,
            "end_f_no" : 10,
            "activity" : "boxing"
        ]
    }
}

"""
import re
import json
import os
from typing import Any
import cv2
import shutil

from loguru import logger

from lib.utils import utils


class KTHDatasetProcessor() :
    def __init__(self, config):
        logger.info("Initialised the KTHDatasetProcessor")
        assert config.get('sequence_file',None) != None, "sequence_file is missing in KTH dataset config"
        assert config.get('src_dir',None) != None, "src_dir is missing in KTH dataset config"
        assert config.get('data_format',None) != None, "data_type is missing in KTH dataset config"
        self.src_dir = config.get('src_dir')
        self.config_file = config.get('sequence_file')
        assert utils.check_if_file_exists(self.config_file), F"{self.config_file} not found for KTH"


    def __call__(self):
        return self.process_kth_annotation_file()
    
    def process_kth_annotation_file(self) :
        with open(self.config_file) as fd :
            data = fd.readlines()
        out = {}
        for line in data :
            if len(line) < 3 : # check if line has any data
                continue
            d = line.replace("\t\t"," ").replace("\t"," ").split(" ")
            activity = d[0].split("_")[1]
            file_name = F"{d[0]}_uncomp.avi"
            activities = []

            for f in d[2:] :
                if len(f.split("-")) < 2 :
                    continue
                f_nos = f.split("-")
                start_f_no = re.findall(r'\d+',f_nos[0])[0]
                end_f_no = re.findall(r'\d+',f_nos[1])[0]
                # print(start_f_no, end_f_no)
                activities.append(
                    {
                        "start_f_no" : int(start_f_no),
                        "end_f_no" : int(end_f_no),
                        "activity" : activity,
                        "src_path" : os.path.join(self.src_dir,file_name),
                        "file_name" : file_name
                    }
                )
            # activity_data = {
            #         # "activity" : activity,
            #         "src_path" : os.path.join(self.src_dir,file_name),
            #         "activities" : activities,
            #         # "activity" : activity
            #     }

            out[file_name] = activities
        
        # with open("kth_sequence_data.json","w") as fd :
        #     json.dump(out,fd)
        
        return out

    def get_all_file_list(self) :
        all_file_names = []
        with open(self.config_file) as fd :
            data = fd.readlines()

        for line in data :
            if len(line) < 3 : # check if line has any data
                continue
            d = line.replace("\t\t"," ").replace("\t"," ").split(" ")
            activity = d[0].split("_")[1]
            file_name = F"{d[0]}_uncomp.avi"
            all_file_names.append(file_name)
        return all_file_names  


    def get_train_test_split(self) :

        # partitions are taken from sequence00.txt file in dataset
        train_person_identifiers = "11,12,13,14,15,16,17,18".split(",")
        test_person_identifiers = "19,22,02,03,05,06,07,08,09,10".split(",")
        val_person_identifiers = "22,20,21,23,24,25,01,04".split(",")
        
        train_files = []
        test_files = []
        
        all_files = self.get_all_file_list()
        
        for f_name in all_files :
            person_id = f_name.split("_")[0].replace("person","")
            if person_id in train_person_identifiers or person_id in val_person_identifiers :
                train_files.append(f_name.split(".")[0])
            else :
                test_files.append(f_name.split(".")[0])
        
        return {
            "train" : train_files,
            "test" : test_files
        }


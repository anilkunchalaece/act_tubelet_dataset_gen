import os
import random
from typing import Any

from loguru import logger


class MMActDatasetProcessor() :
    def __init__(self, config) :
        logger.info(F"initialized the MMAct dataset processor")
        self.config = config
        keys_to_check = ["src_dir","data_format","bbox_info"]
        for k in keys_to_check :
            assert(self.config.get(k,None) != None), F"{k} is missing in the MMAct config"
        
    def __call__(self):
        all_activities = {}
        all_subject_dirs = os.listdir(self.config["src_dir"])
        for each_subject_dir in all_subject_dirs :
            cam_dirs = os.listdir(os.path.join(self.config["src_dir"],each_subject_dir))
            for each_cam_dir in cam_dirs :
                scene_dirs = os.listdir(os.path.join(self.config["src_dir"],each_subject_dir,each_cam_dir))
                for each_scene_dir in scene_dirs :
                    all_sessions = os.listdir(os.path.join(self.config["src_dir"],each_subject_dir, each_cam_dir, each_scene_dir))
                    for each_session in all_sessions :
                        all_videos = os.listdir(os.path.join(self.config["src_dir"],each_subject_dir, each_cam_dir, each_scene_dir,each_session))
                        for each_video in all_videos :
                            video_full_name = os.path.join(self.config["src_dir"],each_subject_dir, each_cam_dir, each_scene_dir,each_session, each_video)
                            video_subpath = os.path.join(each_subject_dir, each_cam_dir, each_scene_dir,each_session, each_video)
                            all_activities[video_subpath] = [
                                {
                                    "activity" : os.path.splitext(each_video)[0],
                                    "src_path" : video_full_name,
                                    "file_name" : video_full_name
                                }
                            ]
        return all_activities

    def get_train_test_split(self, dataset_dir) :
        """
        using cross subject evaluation split from mmact
        """

        all_tubelets = [ x for x in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir,x))]
        #filter the mmact tubelets
        ucfarg_tubelets = [x for x in all_tubelets if x.split("-")[0]=="MMACT"]
        training_subject_ids = list(range(1,17))
        train_files = [] 
        test_files = []

        for x in ucfarg_tubelets :
            if int(x.split("-")[1].split("_")[0].replace("subject","")) in training_subject_ids :
                train_files.append(x)
            else :
                test_files.append(x)
        logger.info(F" no of train files {len(train_files)}, test files {len(test_files)}")
        return {
            "train" : train_files,
            "test" : test_files
        }
        




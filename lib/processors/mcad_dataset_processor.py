import os
import json
import random

from loguru import logger

class MCADDatasetProcessor() :
    def __init__(self,config):
        logger.info("Initalized the MCAD dataset processor")
        self.config = config
        keys_to_check = ["src_dir","data_format", "bbox_info"]
        for k in keys_to_check :
            assert(self.config.get(k,None) != None), F"{k} not found in the MCAD dataset config"
        self.classes_to_include = config.get('classes_to_include',None)
        self.LABEL_MATCHER = {
                    "A01" : "Point",
                    "A02" : "Wave",
                    "A03" : "Jump",
                    "A04" : "Crouch",
                    "A05" : "Sneeze",
                    "A06" : "SitDown",
                    "A07" : "StandUp",
                    "A08" : "Walk",
                    "A09" : "PersonRun",
                    "A10" : "CellToEar",
                    "A11" : "UseCellPhone",
                    "A12" : "DrinkingWater",
                    "A13" : "TakePicture",
                    "A14" : "ObjectGet",
                    "A15" : "ObjectPut",
                    "A16" : "ObjectLeft",
                    "A17" : "ObjectCarry",
                    "A18" : "ObjectThrow"
                }


    def __call__(self):
        all_activites = {}
        all_id_dirs = os.listdir(self.config["src_dir"])
        # filter files if any and get me only dirs
        all_id_dirs = [x for x in all_id_dirs if os.path.isdir(os.path.join(self.config["src_dir"],x))]
        for each_id_dir in all_id_dirs :
            all_videos = os.listdir(os.path.join(self.config["src_dir"], each_id_dir))
            for each_video in all_videos :
                video_full_path = os.path.join(self.config["src_dir"],each_id_dir,each_video)
                activity = self.LABEL_MATCHER.get(each_video.split("_")[-2])
                if self.classes_to_include is not None and activity not in self.classes_to_include :
                    continue # skip if activity is not in classes_to_include
                all_activites[os.path.join(each_id_dir,each_video)] = [
                    {
                        "activity" : activity,
                        "src_path" : video_full_path,
                        "file_name" : video_full_path
                    }
                ]
        return all_activites

    def get_train_test_split(self, dataset_dir) :
        """
        MCAD doesn't have train and validation test.
        So we take the all the generated tubelets and split them into 75% (train) to 25% (test)

        parameters
        dataset_dir -> root dataset with mcad tubeletes
        """
        # get all the tubelets
        all_tubelets = [ x for x in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir,x))]

        # filter the jrdbact tubelets
        mcad_tubelets = [x for x in all_tubelets if x.split("-")[0]=="MCAD"]
        all_classes = list(set([x.split("-")[2] for x in mcad_tubelets]))
        
        logger.info(F"Total number of mcad tubelets are {len(mcad_tubelets)} with {len(all_classes)} => classes {all_classes}")
        
        train_files = []
        test_files = []

        for cls in all_classes :
            cls_tubelets = [x for x in mcad_tubelets if x.split("-")[2] == cls]
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
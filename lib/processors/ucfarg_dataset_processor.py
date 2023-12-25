import os
import json
import random
from loguru import logger


class UCFARGDatasetProcessor() :
    def __init__(self, config):
        logger.info(F"Initialized UCFARG dataset processor")
        self.config = config
        keys_to_check = ["src_dir","data_format","bbox_info"]
        for k in keys_to_check :
            assert(self.config.get(k,None) != None), F"{k} is missing in the UCFARG config"

    def __call__(self) :
        all_dirs = [os.path.join(self.config["src_dir"],x) for x in os.listdir(self.config["src_dir"])]
        # get only directories
        all_dirs = [x for x in all_dirs if os.path.isdir(x)]

        all_activity_data = {}

        for dir_name in all_dirs :
            class_dirs = [os.path.join(dir_name,x) for x in os.listdir(dir_name)]
            # filter the dirs 
            class_dirs = [x for x in class_dirs if os.path.isdir(x)]

            for each_class_dir in class_dirs :
                all_videos = os.listdir(each_class_dir)
                for each_video in all_videos :
                    each_video = os.path.join(each_class_dir,each_video)
                    all_activity_data[each_video] = [{
                        "activity" : os.path.basename(each_class_dir),
                        "src_path" : each_video,
                        "file_name" : each_video
                    }]
        return all_activity_data

    def get_train_test_split(self, dataset_dir) :
        """
        UCFARG doesn't have train and validation test.
        So we take the all the generated tubelets and split them into 75% (train) to 25% (test)

        parameters
        dataset_dir -> root dataset with okutama tubeletes        
        """
        # get all the tubelets
        all_tubelets = [ x for x in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir,x))]

        # filter the jrdbact tubelets
        ucfarg_tubelets = [x for x in all_tubelets if x.split("-")[0]=="UCFARG"]
        all_classes = list(set([x.split("-")[2] for x in ucfarg_tubelets]))
        
        logger.info(F"Total number of okutama tubelets are {len(ucfarg_tubelets)} with {len(all_classes)} => classes {all_classes}")
        
        train_files = []
        test_files = []

        for cls in all_classes :
            cls_tubelets = [x for x in ucfarg_tubelets if x.split("-")[2] == cls]
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
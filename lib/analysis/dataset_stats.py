import os
from loguru import logger

from lib.utils import utils

class DatasetStats() :
    def __init__(self, config=dict()):
        logger.info(F"Initialized Dataset Stats")
        if len(config) == 0 :
            config["DATASETS_TO_CONSIDER"] = "ALL"
        self.config = config
    
    def __call__(self, tubelet_dataset_dir, class_to_include=[], key_word="filtered"):
        train_file = os.path.join(tubelet_dataset_dir,"train.txt")
        test_file = os.path.join(tubelet_dataset_dir,"test.txt")
        class_list_file = os.path.join(tubelet_dataset_dir,"class_list.txt")

        assert(utils.check_if_file_exists(train_file)),F"{train_file} not found"
        assert(utils.check_if_file_exists(test_file)),F"{test_file} not found"
        assert(utils.check_if_file_exists(class_list_file)),F"{class_list_file} not found"

        logger.info(F"classes to include {class_to_include}, key_word {key_word}")
        
        with open(train_file) as fd :
            train_data = fd.readlines()
            train_data = [x.strip() for x in train_data]
        
        with open(test_file) as fd :
            test_data = fd.readlines()
            test_data = [x.strip() for x in test_data]
        
        with open(class_list_file) as fd :
            class_list_data = fd.readlines()
            class_list_data = [x.strip() for x in class_list_data]
        
        logger.info(F"{len(train_data)} train samples, {len(test_data)} test samples, {len(class_list_data)} classes")

        classes_to_remove = []

        if self.config["DATASETS_TO_CONSIDER"] != "ALL" :
            # pass # TODO - NEED TO FIGURE IT OUT
            raise NotImplementedError("Dataset specific stats are yet to be implemented")
        else :
            samples_per_class = {
                "activities" : [],
                "train" : [],
                "test" : [],
                "total" : []
            }
            
            for each_class in class_list_data :
                cl_train_samples = [x for x in train_data if x.split(" ")[0].split("-")[-3] == each_class]
                cl_test_samples = [x for x in test_data if x.split(" ")[0].split("-")[-3] == each_class]
                if len(cl_train_samples) < 10 or len(cl_test_samples) < 10 :
                    logger.warning(F"no samples for {each_class.strip()}, train {len(cl_train_samples)}, test {len(cl_test_samples)}")
                    classes_to_remove.append(each_class)
                    continue # not adding these classes to statistics     
        
            # removing the classes with low samples
            if len(class_to_include) == 0:
                train_data_filtered = [x for x in train_data if x.split(" ")[0].split("-")[-3] not in classes_to_remove]
                test_data_filtered = [x for x in test_data if x.split(" ")[0].split("-")[-3] not in classes_to_remove]
                class_list_data_filtered = [ x for x in class_list_data if x not in classes_to_remove]
            else :
                train_data_filtered = [x for x in train_data if x.split(" ")[0].split("-")[-3] in class_to_include]
                test_data_filtered = [x for x in test_data if  x.split(" ")[0].split("-")[-3] in class_to_include]
                class_list_data_filtered = [ x for x in class_list_data if x in class_to_include]                

            logger.info(F"train_data_filtered {len(train_data_filtered)} \
                test_data_filtered {len(test_data_filtered)} \
                class_list_data_filtered {len(class_list_data_filtered)}")
            
            for each_class in class_list_data_filtered :
                cl_train_samples = [x for x in train_data_filtered if x.split(" ")[0].split("-")[-3] == each_class]
                cl_test_samples = [x for x in test_data_filtered if x.split(" ")[0].split("-")[-3] == each_class]

                samples_per_class["train"].append(len(cl_train_samples))
                samples_per_class["test"].append(len(cl_test_samples))
                samples_per_class["total"].append(len(cl_train_samples) + len(cl_test_samples))
                samples_per_class["activities"].append(each_class)
            
                logger.info(F"each_class {each_class} , train {len(cl_train_samples)} test {len(cl_test_samples)}")
            
            # replacing the class numbers to filtered ones
            def new_class_idx (old_cls_idx) :
                cls_name = class_list_data[int(old_cls_idx)]
                return class_list_data_filtered.index(cls_name)
            
            train_data_filtered_updated_idx = []
            test_data_filtered_updated_idx = []

            for each_line in train_data_filtered : # filtered data still has old idx
                old_cls_idx = each_line.split(" ")[-1]
                new_line = F"{each_line.split(' ')[0]} {int(each_line.split(' ')[1]) - 1} {new_class_idx(old_cls_idx)}"
                train_data_filtered_updated_idx.append(new_line)

            for each_line in test_data_filtered :
                old_cls_idx = each_line.split(" ")[-1]
                new_line = F"{each_line.split(' ')[0]} {int(each_line.split(' ')[1]) - 1} {new_class_idx(old_cls_idx)}"
                test_data_filtered_updated_idx.append(new_line)                

                    
            
            with open(os.path.join(tubelet_dataset_dir,F"train_{key_word}.txt"),"w") as fw :
                fw.writelines([F"{x}\n" for x in train_data_filtered_updated_idx])

            with open(os.path.join(tubelet_dataset_dir,F"test_{key_word}.txt"),"w") as fw :
                fw.writelines([ F"{x}\n" for x in test_data_filtered_updated_idx])
            
            with open(os.path.join(tubelet_dataset_dir,F"class_list_{key_word}.txt"),"w") as fw :
                fw.writelines([F"{x}\n" for x in class_list_data_filtered])
            
            logger.info(F"filtered data is written to {tubelet_dataset_dir}")

        return samples_per_class
"""
This is used to generate the final train and test annotations.
It is used to combine the action classes
"""

import os

LABEL_MATCHER = {
    # KTH Labels
    "walking" : "WALKING",
    "running" : "RUNNING",
    "jogging" : "RUNNING",
    "handwaving" : "GESTURING",
    
    # VIRAT Labels
    "activity_carrying" : "CARRYING",
    "activity_sitting" : "SITTING",
    "activity_gesturing" : "GESTURING",
    "activity_standing" : "STANDING",
    "activity_running" : "RUNNING",
    "activity_walking" : "WALKING",
    "specialized_talking_phone" : "USING_PHONE",
    "specialized_texting_phone" : "USING_PHONE",
    
    # JRDBAct Labels
    "greeting_gestures" : "GESTURING",
    "running" : "RUNNING",
    "standing" : "STANDING",
    "walking" : "WALKING",
    "holding_sth" : "CARRYING",
    "sitting" : "SITTING",
    "talking_on_the_phone" : "USING_PHONE",
    
    # OKUTAMA
    "Calling" : "USING_PHONE",
    "Running" : "RUNNING",
    "Sitting" : "SITTING",
    "Walking" : "WALKING",
    "Standing" : "STANDING",
    "Carrying" : "CARRYING",
    
    # UCFARG
    "carrying" : "CARRYING",
    "jogging" : "RUNNING",
    "running" : "RUNNING",
    "walking" : "WALKING",
    "waving" : "GESTURING",
    
    # MMACT
    "carrying" : "CARRYING",
    "running" : "RUNNING",
    "standing" : "STANDING",
    "talking_on_phone" : "USING_PHONE",
    "using_phone" : "USING_PHONE",
    "walking" : "WALKING",
    "waving_hand" : "GESTURING",
    
    # MCAD
    "CellToEar" : "USING_PHONE",
    "PersonRun" : "RUNNING",
    "SitDown" : "SITTING",
    "StandUp" : "STANDING",
    "TakePicture" : "USING_PHONE",
    "UseCellPhone" : "USING_PHONE",
    "Walk" : "WALKING",
    "Wave" : "GESTURING"
    }

TUBELET_LABELS =["WALKING", "RUNNING", "SITTING", 
                 "STANDING", "GESTURING", "CARRYING", 
                 "USING_PHONE"]


def main(root_dir):
    train_file = os.path.join(root_dir,"train.txt")
    test_file = os.path.join(root_dir,"test.txt")
    class_list_file = os.path.join(root_dir,"class_list.txt")
    
    for f_name in [train_file, test_file] :
        with open(f_name) as fd :
            data = fd.readlines()
            data_modified = []
            for l in data :

                l = l.split(" ")
                org_class = l[0].split("-")[2]
                tubelet_label = LABEL_MATCHER[org_class]
                l_modified = F"{l[0]} {l[1]} {TUBELET_LABELS.index(tubelet_label)}\n"
                
                data_modified.append(l_modified)
            
            fToSave = os.path.join(root_dir, F"tubelet_{os.path.basename(f_name)}")
            
            with open(fToSave,'w') as fw :
                fw.writelines(data_modified)

    with open(os.path.join(root_dir,F"tubelet_class_list.txt"),'w') as fw :
        fw.writelines([F"{x}\n" for x in TUBELET_LABELS])

if __name__ == "__main__" :
    ROOT_DIR = "TUBELET_DATASET_FINAL"
    main(ROOT_DIR)
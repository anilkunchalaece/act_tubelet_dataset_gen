import json
from loguru import logger
import ffmpeg
import os
import shutil
import cv2
import itertools 
import subprocess
import sys
import multiprocessing
from functools import partial

from .processors.kth_dataset_processor import KTHDatasetProcessor
from .processors.virat_dataset_processor import ViratDatasetProcessor
from .processors.jrdbact_dataset_processor import JRDBActDatasetProcessor
from .processors.okutama_dataset_processor import OkutamaDatasetProcessor
from .processors.ucfarg_dataset_processor import UCFARGDatasetProcessor
from .processors.mmact_dataset_processor import MMActDatasetProcessor
from .processors.mcad_dataset_processor import MCADDatasetProcessor

from .utils import utils, person_detector


class ActTubeletGenerator():
    
    def __init__(self, config_file) :
        print("Initialized Act Tubelet Dataset Generator")
        self.config = self.get_config(config_file)
        self.current_processing_dataset = None
        self.current_activity_info = None
        self.current_data = dict()

    def get_config(self, config_file) :
        """get the config of the all the datasets"""
        try :
            with open(config_file) as fd :
                return json.load(fd)
        except Exception as e :
            print(F"Unable to load file {config_file}")
            raise
    
    def set_current_dataset_name(self,k) :
        self.current_processing_dataset = k
    
    def get_current_dataset_name(self) :
        return self.current_processing_dataset

    def set_current_activity_info(self,activity_info) :
        self.current_activity_info = activity_info
    
    def get_current_activity_info(self):
        return self.current_activity_info

    def get_current_bbox_info(self) :
        return self.current_activity_info['bbox_info']

    def set_frames_per_dataset(self) :
        self.MAX_FRAMES_IN_SAMPLE = int(self.config["global_settings"]["max_duration"] * \
                                self.config["each_dataset_config"][self.get_current_dataset_name()]["fps"])
        self.MIN_FRAMES_IN_SAMPLES = int(self.config["global_settings"]["min_duration"] * \
                                self.config["each_dataset_config"][self.get_current_dataset_name()]["fps"])    


    def generate_dataset(self) :
        logger.info("Generate the Act Tubelet Dataset")
        for k in self.config['each_dataset_config'].keys() :
            logger.info(F"running for dataset {k} ")
            logger.info(F"removing tmp dir {self.config['global_settings'].get('tmp_dir','tmp')}")
            
            shutil.rmtree(self.config['global_settings'].get('tmp_dir','tmp'),ignore_errors=True)
            utils.create_dir_if_not_exists(self.config['global_settings'].get('tmp_dir','tmp'))
            self.set_current_dataset_name(k)

            if k in self.config["global_settings"]["datasets_to_consider"] :
                if k == "KTH" :
                    kth_data = KTHDatasetProcessor(self.config["each_dataset_config"]["KTH"])
                    self.current_data = kth_data()
                elif k == "VIRAT" :
                    virat_data = ViratDatasetProcessor(self.config["each_dataset_config"]["VIRAT"])
                    self.current_data = virat_data()
                elif k == "JRDBACT" :
                    jrdbact_data = JRDBActDatasetProcessor(self.config['each_dataset_config']["JRDBACT"])
                    self.current_data = jrdbact_data()
                elif k == "OKUTAMA" :
                    okutama_data = OkutamaDatasetProcessor(self.config['each_dataset_config']["OKUTAMA"])
                    self.current_data = okutama_data()
                elif k == "UCFARG" :
                    ucfarg_data = UCFARGDatasetProcessor(self.config['each_dataset_config']["UCFARG"])
                    self.current_data = ucfarg_data()
                elif k == "MMACT" :
                    mmact_data = MMActDatasetProcessor(self.config['each_dataset_config']["MMACT"])
                    self.current_data = mmact_data()
                elif k == "MCAD" :
                    mcad_data = MCADDatasetProcessor(self.config['each_dataset_config']["MCAD"])
                    self.current_data = mcad_data()
                else :
                    logger.info(F"data processor not implemented for {k}")
                    sys.exit()
                
                self.current_data = dict(itertools.islice(self.current_data.items(), 20)) # FOR TESTING
                self.save_current_data()
                self.extract_tubelets()
            else :
                logger.info(F"skipping {k}")

    def get_train_test_split(self) :
        logger.info(F"Generating the train and test splits")
        train_data = []
        test_data = []
        train_test_split = {}
        # print(self.config['each_dataset_config'].keys())
        for k in self.config['each_dataset_config'].keys() :
            logger.info(F"Processing {k}")
            self.set_current_dataset_name(k)
            self.set_frames_per_dataset()
            # print(self.config["global_settings"]["datasets_to_consider"])
            # print(k in self.config["global_settings"]["datasets_to_consider"])
            if k in self.config["global_settings"]["datasets_to_consider"] :
                if k == "KTH" :
                    kth_data = KTHDatasetProcessor(self.config["each_dataset_config"]["KTH"])
                    train_test_split[k] = kth_data.get_train_test_split()
                elif k == "VIRAT" :
                    virat_data = ViratDatasetProcessor(self.config["each_dataset_config"]["VIRAT"])
                    train_test_split[k] = virat_data.get_train_test_split()
                    # print(train_test_split)
                elif k == "JRDBACT" :
                    jrdbact_data = JRDBActDatasetProcessor(self.config["each_dataset_config"]["JRDBACT"])
                    train_test_split[k] = jrdbact_data.get_train_test_split(self.config["global_settings"]["output_dir"])
                elif k == "OKUTAMA" :
                    okutama_data = OkutamaDatasetProcessor(self.config["each_dataset_config"]["OKUTAMA"])
                    train_test_split[k] = okutama_data.get_train_test_split(self.config["global_settings"]["output_dir"])
                elif k == "UCFARG" :
                    ucfarg_data = UCFARGDatasetProcessor(self.config["each_dataset_config"]["UCFARG"])
                    train_test_split[k] = ucfarg_data.get_train_test_split(self.config["global_settings"]["output_dir"])
                elif k == "MMACT" :
                    mmact_data = MMActDatasetProcessor(self.config["each_dataset_config"]["MMACT"])
                    train_test_split[k] = mmact_data.get_train_test_split(self.config["global_settings"]["output_dir"])
                elif k == "MCAD" :
                    mcad_data = MCADDatasetProcessor(self.config["each_dataset_config"]["MCAD"])
                    train_test_split[k] = mcad_data.get_train_test_split(self.config["global_settings"]["output_dir"])
                else :
                    logger.info(F"not implemted for {k}")
            else :
                logger.info(F"NOT PROCESSING FOR {k}")
                continue

        all_dataset_samples = [ x for x in os.listdir(self.config['global_settings']['output_dir']) \
                                if os.path.isdir(os.path.join(self.config['global_settings']['output_dir'],x))]
        
        classes_list = list(set([ x.split("-")[2] for x in all_dataset_samples]))

        for each_sample in all_dataset_samples :
            sample_length =len( os.listdir(os.path.join(self.config['global_settings']['output_dir'], each_sample)))
            k = each_sample.split("-")[0] # get dataset name
            
            # for JRDBACT consider entire dir name instead of video name -> since JRDB collects the data in single burst i.e it doesn't have any individual vidoes / all the images are part of single video ? 
            each_dir_name = each_sample.split("-")[1] if k not in ["JRDBACT","OKUTAMA","UCFARG","MMACT","MCAD"] else each_sample

            if sample_length > self.MIN_FRAMES_IN_SAMPLES :
                if each_dir_name in train_test_split[k]["train"] :
                    train_data.append(F"{os.path.basename(each_sample)} {sample_length} {classes_list.index(os.path.basename(each_sample).split('-')[2])}\n")
                elif each_dir_name in train_test_split[k]["test"] :
                    test_data.append(F"{os.path.basename(each_sample)} {sample_length} {classes_list.index(os.path.basename(each_sample).split('-')[2])}\n")
                else :
                    logger.info(F"{os.path.basename(each_sample)} is not part of partition")
            else :
                logger.info(F"Skipping {os.path.basename(each_sample)}, it contains only {sample_length} samples")

            
        with open(os.path.join(self.config['global_settings']['output_dir'],"train.txt"),'w') as fw:
            fw.writelines(train_data)
        with open(os.path.join(self.config['global_settings']['output_dir'],"test.txt"),'w') as fw :
            fw.writelines(test_data)
        with open(os.path.join(self.config['global_settings']['output_dir'],"class_list.txt"),'w') as fw :
            fw.writelines([ F"{x}\n" for x in classes_list])

        logger.info(F"Dataset annotations are saved to {self.config['global_settings']['output_dir']}")                      

    def extract_tubelets(self) :
        """ This function will do post processing (such as get person detections) if needed and extract the tubelets"""
        logger.info(F"extracting the processed data")
        
        dataset_name = self.get_current_dataset_name()
        self.set_frames_per_dataset()
        # print(self.current_data.keys())

        # modifying this to process all videos first (i.e) convert them to frames
        # and add the frames directory to the src_dir in each activity
        
        if self.config['each_dataset_config'][dataset_name].get('data_format','frames') == "video" :
            logger.info(F"converting the videos into the frames")
            all_videos = self.current_data.keys()
            all_videos = [os.path.join(self.config['each_dataset_config'][dataset_name]['src_dir'],x) \
                          for x in all_videos]
            
            pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
            pool.map(self.get_frames_from_video, all_videos)
            pool.close()
            pool.join()

            # add the frames dir as the src_dir for each activity
            for video_name in self.current_data.keys() :
                for idx, act in enumerate(self.current_data[video_name]) :
                    if self.get_current_dataset_name() != "MMACT" : 
                        self.current_data[video_name][idx]['src_dir'] = os.path.join(self.config["global_settings"]["tmp_dir"],\
                                                                     os.path.splitext(os.path.basename(video_name))[0] )
                    else :
                        self.current_data[video_name][idx]['src_dir'] = os.path.join(self.config["global_settings"]["tmp_dir"],\
                                                                    F"{'_'.join(video_name.split(os.sep)[-5:-1])}_{os.path.splitext(os.path.basename(video_name))[0]}")
        # if bbox info not available in the dataset, run the pedestrian detector and get the detections
        # for all the cases, we only have one person in frame i.e one person per frame
        if self.config['each_dataset_config'][dataset_name].get('bbox_info', False) == False :
            
            # we have to run this each video, cuda won't support multiprocessing (or does it ?)
            for each_video in self.current_data.keys() :
                detections = self.get_person_detections(self.current_data[each_video][0]['src_dir'])
                if len(detections) == 0 :
                    continue
                # check for start_f_no and end_f_no
                # we are assuming the each video has only class
                if self.current_data[each_video][0].get("start_f_no",None) == None :
                    all_frame_ids = [int(x.split("_")[-1]) for x in detections.keys()]
                    self.current_data[each_video][0]["start_f_no"] = min(all_frame_ids)
                    self.current_data[each_video][0]["end_f_no"] = max(all_frame_ids)
        
                # using '0' since all the activities in a single video has single frame
                for act_idx, act in enumerate(self.current_data[each_video]) :
                    bbox_info = {}
                    ## add bounding box information using the detections
                    for frame_idx in range(act["start_f_no"], act["end_f_no"]) :
                        bbox_info[F"img_{frame_idx:05d}"] = detections.get(F"img_{frame_idx:05d}")

                    self.current_data[each_video][act_idx]["bbox_info"] = bbox_info

        out_dir = os.path.join(self.config['global_settings']['output_dir'])
        utils.create_dir_if_not_exists(out_dir) # out root dir for dataset

        self.save_current_data()

        all_activities = []
        for each_video in self.current_data.keys() :
            for each_act in self.current_data[each_video] :
                all_activities.append(each_act)
        # For testing
        # for act in all_activities :
        #     self.process_each_activity(act)
        #     break
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        pool.map(self.process_each_activity, all_activities)
        pool.close()
        pool.join()

    def process_each_activity(self,activity_info) :
        img_src_dir_path = activity_info['src_dir']
        all_src_imgs = os.listdir(img_src_dir_path)
        act_start_frame_no = activity_info.get('start_f_no',None)
        act_end_frame_no = activity_info.get('end_f_no',None)
        if act_start_frame_no == None or act_end_frame_no == None :
            logger.warning(F"Skipping {img_src_dir_path}, since we are unable to find any detections")
            return
        act_start_frame_no = int(act_start_frame_no)
        act_end_frame_no = int(act_end_frame_no)
        activity_name = activity_info['activity']
        for p_idx, idx_org in enumerate(range(act_start_frame_no, act_end_frame_no, self.MAX_FRAMES_IN_SAMPLE)) :
            start_idx = idx_org
            end_idx = start_idx + self.MAX_FRAMES_IN_SAMPLE
            end_idx = end_idx if end_idx <= len(all_src_imgs) else len(all_src_imgs)
            # logger.info(F"processing {start_idx} to {end_idx}")
            f_name_idx = 0
            out_dir = os.path.join(self.config['global_settings']['output_dir'],
                                    # self.get_current_dataset_name(), -> skipping this as well store all of these in the same dir
                                    # current_activity, -> skipping this, may be we don't need it
                                    # replacing '-' in src_dir name with '_' for consistency in tubelet dir name 
                                    # replacing the spaces ' ' in activity aka class names with _ -> I dont know how its gonna pan out
                                    F"{self.get_current_dataset_name()}-{os.path.basename(activity_info['src_dir']).replace('-','_')}-{activity_name.replace(' ','_')}-id{act_start_frame_no}_{act_end_frame_no}-p{p_idx}"
                                    )
            utils.create_dir_if_not_exists(out_dir)

            for idx in range(start_idx, end_idx) :
                if self.get_current_dataset_name() != "JRDBACT" :
                    img_path = os.path.join(img_src_dir_path,F"img_{idx:05d}.jpg")
                else : # image names are having different notation for 
                    img_path = os.path.join(img_src_dir_path,F"{idx:06d}.jpg")

                if not os.path.isfile(img_path) :
                    logger.info(F"{img_path} not found! skipping")
                    # return
                
                try :
                    # logger.info(F"range {[start_idx, end_idx]} , idx {idx}, img_{f_name_idx:05d}.jpg")
                    img = cv2.imread(img_path)
                    bbox = self.get_bbox_for_idx(idx, [start_idx,end_idx], activity_info)
                    crop_img = img[bbox[1]:bbox[3],bbox[0]:bbox[2]]
                    # logger.info(F"range {[start_idx, end_idx]} , idx {idx}, img_{f_name_idx:05d}.jpg")
                    out_img_path = os.path.join(out_dir, F"img_{f_name_idx:05d}.png")
                    cv2.imwrite(out_img_path,crop_img)
                    f_name_idx = f_name_idx + 1
                except Exception as e:
                    pass
                    # logger.info(F"unable to write for {img_path}, failed with {e} , bbox {bbox}, {img.shape} , {crop_img.shape}")
                    # raise
                    # return


    
    def save_current_data(self) :
        utils.create_dir_if_not_exists(self.config['global_settings']['output_dir'])
        path_to_save = os.path.join(self.config['global_settings']['output_dir'],
                                    F"{self.get_current_dataset_name()}_data.json")
        
        with open(path_to_save,'w') as fw :
            json.dump(self.current_data, fw)

    
    def get_bbox_for_idx(self, frame_idx, tubelet_idx_range, activity_info) :
        """ 
        This function is used to return the bbox for given frame.
        It will also support bbox variations based on the config.
        currently it supports the following
        1. org -> don't modify the bbox and return as it is
        2. union -> regardless of currend idx, return the union of bbox'es based on tubelet_idx_range
        3. uniform -> make all the boundingboxes of uniform height and width - TODO
        """
        def get_bbox(idx) :
            img_key = F"img_{idx:05d}"
            return activity_info['bbox_info'].get(img_key,None)
        bbox_variation = self.config['global_settings']['bbox_variation']
        # for OKUTAMA only consider the 'org' bounding boxes, since it has moving camera, union is only applicable for static camera
        bbox_variation = bbox_variation if self.get_current_dataset_name() != "OKUTAMA" else "org"
        assert bbox_variation in ["org", "union"], F"unknown bbox variaion in config {bbox_variation}"

        if bbox_variation == "org" :
            return get_bbox(frame_idx) # TODO -> need a way to skip the frame
        elif bbox_variation == "union" :
            bboxes_for_range = [get_bbox(i) for i in range(tubelet_idx_range[0],tubelet_idx_range[1]) if get_bbox(i) != None]
            return self.union_of_bounding_boxes(bboxes_for_range)



    def union_of_bounding_boxes(self, bounding_boxes):
        # Parsing the bounding boxes into tuples of (x_min, y_min, x_max, y_max)
        bounding_boxes = [tuple(map(int, box.split() if type(box) == str else box)) for box in bounding_boxes]

        # Finding the minimum and maximum coordinates for the union
        x_min = min(box[0] for box in bounding_boxes)
        y_min = min(box[1] for box in bounding_boxes)
        x_max = max(box[2] for box in bounding_boxes)
        y_max = max(box[3] for box in bounding_boxes)

        return x_min, y_min, x_max, y_max            
    

    def get_frames_from_video(self,video_name):
        """
        Extract frames from given video and return the dir where the frames are stored
        """
        fps = self.config['global_settings'].get('src_data_fps','org')
        tmp_dir = os.path.join(self.config['global_settings'].get('tmp_dir','tmp'))
        if self.get_current_dataset_name() != "MMACT" :
            output_dir = os.path.join(tmp_dir,os.path.splitext(os.path.basename(video_name))[0])
        else :
            output_dir = os.path.join(tmp_dir,F"{'_'.join(video_name.split(os.sep)[-5:-1])}_{os.path.splitext(os.path.basename(video_name))[0]}")
        utils.create_dir_if_not_exists(output_dir)

        logger.info(F"Converting {os.path.basename(video_name)} and storing frames in {output_dir}")
        out_format = f"{output_dir}/img_%05d.jpg"

        try :
            if fps == "org" :
                cmd = ffmpeg.input(video_name).output(out_format, loglevel='quiet').run()
            else :
                cmd = ffmpeg.input(video_name, r=fps).output(out_format,loglevel='quiet').run()

            # subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            return output_dir
        except subprocess.CalledProcessError as e:
            logger.error(F"unable to extract from video {video_name} to {output_dir} failed with {e.output.decode()}")
            raise
    
    def get_person_detections(self, src_path, format="images") :
        """ Get the person detection from given a"""
        logger.info(F"currnet data of format {format} doesn't have any bounding box info, getting the bounding box info from {src_path}")

        frames_dir = self.get_frames_from_video(src_path) if format == "video" else src_path
        detections = person_detector.get_person_bboxes_from_dir(frames_dir)
        return detections
import json
from loguru import logger
import ffmpeg
import os
import shutil
import cv2
import itertools 
import subprocess

from .processors.kth_dataset_processor import KTHDatasetProcessor
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
    
    def get_current_acivity_info(self):
        return self.current_activity_info

    def get_current_bbox_info(self) :
        return self.current_activity_info['bbox_info']


    def generate_dataset(self) :
        logger.info("Generate the Act Tubelet Dataset")
        for k in self.config['each_dataset_config'].keys() :
            logger.info(F"running for dataset {k} ")
            logger.info(F"removing tmp dir {self.config['global_settings'].get('tmp_dir','tmp')}")
            
            shutil.rmtree(self.config['global_settings'].get('tmp_dir','tmp'),ignore_errors=True)
            utils.create_dir_if_not_exists(self.config['global_settings'].get('tmp_dir','tmp'))
            self.set_current_dataset_name(k)

            if k == "KTH" :
                kth_data = KTHDatasetProcessor()
                self.current_data = kth_data(self.config["each_dataset_config"]["KTH"])
                self.current_data = dict(itertools.islice(self.current_data.items(), 2)) 

            self.extract_tubelets()


    def extract_tubelets(self) :
        """ This function will do post processing (such as get person detections) if needed and extract the tubelets"""
        logger.info(F"extracting the processed data")
        
        dataset_name = self.get_current_dataset_name()

        # process by each video
        for each_video in self.current_data.keys() : # 'k' denotes the video / src dir name
            logger.info(F"processing {each_video}")

            if self.config['each_dataset_config'][dataset_name].get('bbox_info',False) == False:
                data_format = self.config['each_dataset_config'][dataset_name]['data_format']
                detections, frames_dir = self.get_person_detections(self.current_data[each_video]['src_path'], data_format)
                self.current_data[each_video]['src_path'] = frames_dir
                self.current_data[each_video]['bbox_info'] = detections
        
        MAX_FRAMES_IN_SAMPLE = int(self.config["global_settings"]["max_duration"] * \
                                self.config["each_dataset_config"][self.get_current_dataset_name()]["fps"])
        MIN_FRAMES_IN_SAMPLES = int(self.config["global_settings"]["min_duration"] * \
                                self.config["each_dataset_config"][self.get_current_dataset_name()]["fps"])        
        
        out_dir = os.path.join(self.config['global_settings']['output_dir'])
        utils.create_dir_if_not_exists(out_dir) # out root dir for dataset

        with open("current_data.json","w") as fw :
            json.dump(self.current_data,fw)

        for video_name, activities_in_sample in self.current_data.items() :
            logger.info(F"processing {video_name}")
            self.set_current_activity_info(activities_in_sample)
            img_src_dir_path = activities_in_sample['src_path']
            all_src_imgs = os.listdir(img_src_dir_path)

            for act_idx, act_info in enumerate(activities_in_sample['activities']) :
                act_start_frame_no = int(act_info['start_f_no'])
                act_end_frame_no = int(act_info['end_f_no'])
                current_activity = act_info['activity']
                logger.info(F"processing {current_activity} with frame range [{act_start_frame_no} to {act_end_frame_no}]")

                for p_idx, idx_org in enumerate(range(act_start_frame_no, act_end_frame_no, MAX_FRAMES_IN_SAMPLE)) :
                    start_idx = idx_org
                    end_idx = start_idx + MAX_FRAMES_IN_SAMPLE
                    end_idx = end_idx if end_idx <= len(all_src_imgs) else len(all_src_imgs)
                    logger.info(F"processing {start_idx} to {end_idx}")
                    f_name_idx = 0
                    out_dir = os.path.join(self.config['global_settings']['output_dir'],self.get_current_dataset_name(),
                                            current_activity,
                                            F"{os.path.basename(self.get_current_acivity_info()['src_path'])}_act_{act_idx}_p{p_idx}"
                                            )
                    utils.create_dir_if_not_exists(out_dir)                    
                    for idx in range(start_idx, end_idx) : 
                        img_path = os.path.join(img_src_dir_path, F"img_{idx:05d}.jpg") # get image path based on current index
                        # logger.info(F"image path is {img_path}")
                        if not os.path.isfile(img_path) :
                            logger.info(F"{img_path} not found skipping")
                            continue
                        
                        bbox = self.get_bbox_for_idx(idx, [start_idx, end_idx])
                        
                        try :
                            img = cv2.imread(img_path)
                            crop_img = img[bbox[1]:bbox[3],bbox[0]:bbox[2]]

                            out_img_path = os.path.join(out_dir,F"img_{f_name_idx:05d}.jpg")
                            cv2.imwrite(out_img_path,crop_img)
                            f_name_idx = f_name_idx + 1
                        except Exception as e:
                            logger.info(F"unable to write image! failed with exception {e}")
                            raise


    
    def get_bbox_for_idx(self, frame_idx, tubelet_idx_range) :
        """ 
        This function is used to return the bbox for given frame.
        It will also support bbox variations based on the config.
        currently it supports the following
        1. org -> don't modify the bbox and return as it is
        2. union -> regardless of currend idx, return the union of bbox'es based on tubelet_idx_range
        3. uniform -> make all the boundingboxes of uniform height and width - TODO
        """
        bbox_variation = self.config['global_settings']['bbox_variation']
        assert bbox_variation in ["org", "union"], F"unknown bbox variaion in config {bbox_variation}"

        if bbox_variation == "org" :
            return self.get_bbox(frame_idx) # TODO -> need a way to skip the frame
        elif bbox_variation == "union" :
            bboxes_for_range = [self.get_bbox(i) for i in range(tubelet_idx_range[0],tubelet_idx_range[1]) if self.get_bbox(i) != None]
            return self.union_of_bounding_boxes(bboxes_for_range)

    def get_bbox(self,idx) :
        img_key = F"img_{idx:05d}"
        return self.get_current_bbox_info().get(img_key,None)
        


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
        output_dir = os.path.join(tmp_dir,os.path.basename(video_name).split('.')[0])
        utils.create_dir_if_not_exists(output_dir)

        logger.info(F"Converting {os.path.basename(video_name)} and storing frames in {output_dir}")
        out_format = f"{output_dir}/img_%05d.jpg"

        try :
            if fps == "org" :
                cmd = ffmpeg.input(video_name).output(out_format).run()
            else :
                cmd = ffmpeg.input(video_name, r=fps).output(out_format).run()

            # subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            return output_dir
        except subprocess.CalledProcessError as e:
            logger.error(F"unable to extract from video {video_name} to {output_dir} failed with {e.output.decode()}")
            raise
    
    def get_person_detections(self, src_path, format) :
        """ Get the person detection from given a"""
        logger.info(F"currnet data of format {format} doesn't have any bounding box info, getting the bounding box info from given data")

        frames_dir = self.get_frames_from_video(src_path) if format == "video" else src_path
        
        detections = person_detector.get_person_bboxes_from_dir(frames_dir)
        return detections, frames_dir
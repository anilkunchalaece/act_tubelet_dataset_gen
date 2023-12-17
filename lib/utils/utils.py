import os
import shutil
from loguru import logger

def create_dir_if_not_exists(dir_to_check) :
    logger.info(F"creating dir(s) {dir_to_check}")
    if type(dir_to_check) == list :
        for d in dir_to_check:
            os.makedirs(d,exist_ok=True)
    else :
        os.makedirs(dir_to_check, exist_ok=True)

def check_if_file_exists(file_to_check) :
    return os.path.isfile(file_to_check)

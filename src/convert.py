import glob
import os

# http://shuoyang1213.me/WIDERFACE/
import shutil
from collections import defaultdict
from urllib.parse import unquote, urlparse

import numpy as np
import supervisely as sly
from dataset_tools.convert import unpack_if_archive
from dotenv import load_dotenv
from supervisely.io.fs import (
    dir_exists,
    file_exists,
    get_file_ext,
    get_file_name,
    get_file_name_with_ext,
    get_file_size,
)
from tqdm import tqdm

import src.settings as s


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    # project_name = "FACE"
    dataset_path = "APP_DATA/WIDER FACE/"
    batch_size = 30
    anns_folder = "APP_DATA/WIDER FACE/wider_face_split"

    def create_ann(image_path):
        labels = []

        image_np = sly.imaging.image.read(image_path)[:, :, 0]
        img_height = image_np.shape[0]
        img_wight = image_np.shape[1]

        bboxes = im_path_to_bboxes[image_path]
        for bbox in bboxes:
            left = int(bbox[0])
            right = left + int(bbox[2])
            top = int(bbox[1])
            bottom = top + int(bbox[3])
            rectangle = sly.Rectangle(top=top, left=left, bottom=bottom, right=right)
            label = sly.Label(rectangle, obj_class)
            labels.append(label)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels)

    obj_class = sly.ObjClass("face", sly.Rectangle)

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
    meta = sly.ProjectMeta(obj_classes=[obj_class])
    api.project.update_meta(project.id, meta.to_json())

    ds_to_anns = {
        "train": "wider_face_train_bbx_gt.txt",
        "val": "wider_face_val_bbx_gt.txt",
        "test": "wider_face_test_filelist.txt",
    }

    for ds_name in os.listdir(dataset_path):
        if ds_name == "wider_face_split":
            continue
        ds_path = os.path.join(dataset_path, ds_name)
        if dir_exists(ds_path):
            ds_name = ds_name.split("_")[1]
            dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)
            if ds_name != "test":
                anns_path = os.path.join(anns_folder, ds_to_anns[ds_name])
                im_path_to_bboxes = defaultdict(list)
                with open(anns_path) as f:
                    content = f.read().split("\n")

                    for curr_data in content:
                        if curr_data.find("/") != -1:
                            curr_im_path = os.path.join(ds_path, "images", curr_data)
                            continue
                        if curr_data.find(" ") != -1:
                            im_path_to_bboxes[curr_im_path].append(curr_data.split(" ")[:4])

                images_pathes = list(im_path_to_bboxes.keys())

            else:
                images_pathes = glob.glob(ds_path + "/*/*/*.jpg")

            progress = sly.Progress("Create dataset {}".format(ds_name), len(images_pathes))

            for img_pathes_batch in sly.batched(images_pathes, batch_size=batch_size):
                img_names_batch = [get_file_name_with_ext(im_path) for im_path in img_pathes_batch]

                img_infos = api.image.upload_paths(dataset.id, img_names_batch, img_pathes_batch)
                img_ids = [im_info.id for im_info in img_infos]

                if ds_name != "test":
                    anns = [create_ann(image_path) for image_path in img_pathes_batch]
                    api.annotation.upload_anns(img_ids, anns)

                progress.iters_done_report(len(img_names_batch))
    return project

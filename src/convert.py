# http://shuoyang1213.me/WIDERFACE/
import glob
import os
import shutil
from collections import defaultdict
from urllib.parse import unquote, urlparse

import numpy as np
import supervisely as sly
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
from dataset_tools.convert import unpack_if_archive


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    # project_name = "FACE"
    dataset_path = "APP_DATA/WIDER FACE/"
    batch_size = 30
    anns_folder = "APP_DATA/WIDER FACE/wider_face_split"

    def create_ann(image_path, meta):
        # global meta
        labels = []

        # tag_scene_meta = meta.get_tag_meta(tag_scene_value)
        # if tag_scene_meta is None:
        #     tag_scene_meta = sly.TagMeta(tag_scene_value, sly.TagValueType.NONE)
        #     meta = meta.add_tag_meta(tag_scene_meta)
        #     api.project.update_meta(project.id, meta.to_json())
        # tag_scene = sly.Tag(tag_scene_meta)

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
            blur = sly.Tag(tag_blur, value=tag_to_blur[bbox[4]])
            expression = sly.Tag(tag_expression, value=tag_to_expression[bbox[5]])
            illumination = sly.Tag(tag_illumination, value=tag_to_illumination[bbox[6]])
            occlusion = sly.Tag(tag_occlusion, value=tag_to_occlusion[bbox[8]])
            pose = sly.Tag(tag_pose, value=tag_to_pose[bbox[9]])
            invalid = sly.Tag(tag_invalid, value=tag_to_invalid[bbox[7]])
            label = sly.Label(
                rectangle,
                obj_class,
                tags=[blur, expression, illumination, occlusion, pose, invalid],
            )
            labels.append(label)

        tag_name = image_path.split("/")[-2].replace("--", "-").lower()
        if "people-driving-car" in tag_name:
            tag_name = "59-people_driving_car"
        tags = [sly.Tag(tag_meta) for tag_meta in tag_metas if tag_meta.name == tag_name]

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels, img_tags=tags)

    obj_class = sly.ObjClass("face", sly.Rectangle)

    tag_blur = sly.TagMeta(
        "blur",
        sly.TagValueType.ONEOF_STRING,
        possible_values=["clear", "normal blur", "heavy blur"],
    )
    tag_expression = sly.TagMeta(
        "expression",
        sly.TagValueType.ONEOF_STRING,
        possible_values=["typical expression", "exaggerate expression"],
    )
    tag_illumination = sly.TagMeta(
        "illumination",
        sly.TagValueType.ONEOF_STRING,
        possible_values=["normal illumination", "extreme illumination"],
    )
    tag_occlusion = sly.TagMeta(
        "occlusion",
        sly.TagValueType.ONEOF_STRING,
        possible_values=["no occlusion", "partial occlusion", "heavy occlusion"],
    )
    tag_pose = sly.TagMeta(
        "pose", sly.TagValueType.ONEOF_STRING, possible_values=["typical pose", "atypical pose"]
    )
    tag_invalid = sly.TagMeta(
        "invalid", sly.TagValueType.ONEOF_STRING, possible_values=["false", "true"]
    )

    tag_to_blur = {"0": "clear", "1": "normal blur", "2": "heavy blur"}
    tag_to_expression = {"0": "typical expression", "1": "exaggerate expression"}
    tag_to_illumination = {"0": "normal illumination", "1": "extreme illumination"}
    tag_to_occlusion = {"0": "no occlusion", "1": "partial occlusion", "2": "heavy occlusion"}
    tag_to_pose = {"0": "typical pose", "1": "atypical pose"}
    tag_to_invalid = {"0": "false", "1": "true"}

    tag_names = [
        "0-parade",
        "1-handshaking",
        "2-demonstration",
        "3-riot",
        "4-dancing",
        "5-car_accident",
        "6-funeral",
        "7-cheering",
        "8-election_campain",
        "9-press_conference",
        "10-people_marching",
        "11-meeting",
        "12-group",
        "13-interview",
        "14-traffic",
        "15-stock_market",
        "16-award_ceremony",
        "17-ceremony",
        "18-concerts",
        "19-couple",
        "20-family_group",
        "21-festival",
        "22-picnic",
        "23-shoppers",
        "24-soldier_firing",
        "25-soldier_patrol",
        "26-soldier_drilling",
        "27-spa",
        "28-sports_fan",
        "29-students_schoolkids",
        "30-surgeons",
        "31-waiter_waitress",
        "32-worker_laborer",
        "33-running",
        "34-baseball",
        "35-basketball",
        "36-football",
        "37-soccer",
        "38-tennis",
        "39-ice_skating",
        "40-gymnastics",
        "41-swimming",
        "42-car_racing",
        "43-row_boat",
        "44-aerobics",
        "45-balloonist",
        "46-jockey",
        "47-matador_bullfighter",
        "48-parachutist_paratrooper",
        "49-greeting",
        "50-celebration_or_party",
        "51-dresses",
        "52-photographers",
        "53-raid",
        "54-rescue",
        "55-sports_coach_trainer",
        "56-voter",
        "57-angler",
        "58-hockey",
        "59-people_driving_car",
        "61-street_battle",
    ]
    tag_metas = [sly.TagMeta(name, sly.TagValueType.NONE) for name in tag_names]

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
    meta = sly.ProjectMeta(
        obj_classes=[obj_class],
        tag_metas=[
            tag_blur,
            tag_expression,
            tag_illumination,
            tag_occlusion,
            tag_pose,
            tag_invalid,
        ]
        + tag_metas,
    )
    api.project.update_meta(project.id, meta.to_json())

    ds_to_anns = {
        "WIDER_train": "wider_face_train_bbx_gt.txt",
        "WIDER_val": "wider_face_val_bbx_gt.txt",
        "WIDER_test": "wider_face_test_filelist.txt",
    }

    for folder_name in sorted(os.listdir(dataset_path)):
        if folder_name == "wider_face_split":
            continue
        ds_path = os.path.join(dataset_path, folder_name)
        if dir_exists(ds_path):
            ds_name = folder_name.split("_")[1]
            dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)
            if folder_name != "WIDER_test":
                anns_path = os.path.join(anns_folder, ds_to_anns[folder_name])
                im_path_to_bboxes = defaultdict(list)
                with open(anns_path) as f:
                    content = f.read().split("\n")

                    for curr_data in content:
                        if curr_data.find("/") != -1:
                            curr_im_path = os.path.join(ds_path, "images", curr_data)
                            continue
                        if curr_data.find(" ") != -1:
                            im_path_to_bboxes[curr_im_path].append(curr_data.split(" ")[:10])

                images_pathes = list(im_path_to_bboxes.keys())

            else:
                images_pathes = glob.glob(ds_path + "/*/*/*.jpg")

            progress = sly.Progress("Create dataset {}".format(ds_name), len(images_pathes))

            for img_pathes_batch in sly.batched(images_pathes, batch_size=batch_size):
                img_names_batch = [get_file_name_with_ext(im_path) for im_path in img_pathes_batch]

                img_infos = api.image.upload_paths(dataset.id, img_names_batch, img_pathes_batch)
                img_ids = [im_info.id for im_info in img_infos]

                if folder_name != "WIDER_test":
                    anns = [create_ann(image_path, meta) for image_path in img_pathes_batch]
                    api.annotation.upload_anns(img_ids, anns)

                progress.iters_done_report(len(img_names_batch))
    return project

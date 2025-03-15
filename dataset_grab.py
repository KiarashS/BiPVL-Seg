import os
import numpy as np
from monai.transforms import (
    AsDiscrete, Compose, LoadImaged, ToTensord, ScaleIntensityRanged, CropForegroundd, 
    Orientationd, RandShiftIntensityd, RandSpatialCropd, Spacingd, EnsureTyped, 
    RandAffined,  RandCropByPosNegLabeld, NormalizeIntensityd, RandScaleIntensityd,
    SpatialPadd, MapTransform, RandFlipd, EnsureChannelFirstd
)
from utils import RemapLabels
import torch
from monai.apps import DecathlonDataset
import pandas as pd
from sklearn.model_selection import train_test_split
class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 2 and label 3 to construct TC
            result.append(torch.logical_or(d[key] == 2, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(torch.logical_or(torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1))
            # label 2 is ET
            result.append(d[key] == 2)
            d[key] = torch.stack(result, axis=0).float()
        return d

def whs_ct(seed, train_num_samples):
    # %%
    class_definitions = './semantic_aware/classes_definitions_whs_iccv.csv'
    directory = "./WHS_CT"
    # if version == "v1":
    #     class_definitions = '/home/avisionguy/Multi_Modality_Seg_v3/semantic_aware/classes_definitions_whs.csv'
    # elif version == "v2":
    #     class_definitions  = '/home/avisionguy/Multi_Modality_Seg_v3/semantic_aware/classes_definitions_whs_v2.csv' #non-fixed, ambiguous details
    # elif version == "v3":
    #     class_definitions = '/home/avisionguy/Multi_Modality_Seg_v3/semantic_aware/classes_definitions_whs_v3.csv' #only class names
    # elif version == "v4":
    #     class_definitions = '/home/avisionguy/Multi_Modality_Seg_v3/semantic_aware/classes_definitions_whs_v4.csv' #misaligned features details

    # Get all files in the directory
    files = os.listdir(directory)

    # Separate images and labels
    images = sorted([f for f in files if 'image' in f])
    labels = sorted([f for f in files if 'label' in f])

    # Check if the number of images and labels match
    if len(images) != len(labels):
        print("Mismatch between images and labels count!")
    else:
        # Create the list of dictionaries with image-label pairs
        dataset = []
        for img, lbl in zip(images, labels):
            dataset.append({
                'image': os.path.join(directory, img),
                'label': os.path.join(directory, lbl)
            })

    # %%
    


    train, test = train_test_split(dataset, test_size=0.2, random_state=seed)

    train, val = train_test_split(train, test_size=0.1, random_state=seed)
    
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"], ensure_channel_first=True, image_only=True), #0
            Spacingd(keys=["image", "label"], pixdim=(
                1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"], a_min=0, a_max=400,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            SpatialPadd(keys=["image", "label"], spatial_size=(96, 96, 96), mode='constant'),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=train_num_samples,
                image_key="image",
                image_threshold=0,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.50,
            ),
            RandAffined(
                keys=['image', 'label'],
                mode=('bilinear', 'nearest'),
                prob=1.0, spatial_size=(96, 96, 96),
                rotate_range=(0, 0, np.pi / 30),
                scale_range=(0.1, 0.1, 0.1)),
            RemapLabels(keys=["label"]),
            ToTensord(keys=["image", "label"]),
        ]
    )
    val_transforms =  Compose(
            [
                LoadImaged(keys=["image", "label"], ensure_channel_first=True, image_only=True), #0
                Spacingd(keys=["image", "label"], pixdim=(
                    1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=0, a_max=400,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                SpatialPadd(keys=["image", "label"], spatial_size=(96, 96, 96), mode='constant'),
                RemapLabels(keys=["label"]),
                ToTensord(keys=["image", "label"]),
            ]
        )
    return train, val, test, class_definitions, train_transforms, val_transforms

def whs_mri(seed, train_num_samples):
    # %%
    class_definitions = './semantic_aware/classes_definitions_whs_iccv_mr.csv'
    directory = "/home/avisionguy/Multi_Modality_Seg_v3/dataset/data2/WHS_MR/"

    # Get all files in the directory
    files = os.listdir(directory)

    # Separate images and labels
    images = sorted([f for f in files if 'image' in f])
    labels = sorted([f for f in files if 'label' in f])

    # Check if the number of images and labels match
    if len(images) != len(labels):
        print("Mismatch between images and labels count!")
    else:
        # Create the list of dictionaries with image-label pairs
        dataset = []
        for img, lbl in zip(images, labels):
            dataset.append({
                'image': os.path.join(directory, img),
                'label': os.path.join(directory, lbl)
            })

    # %%
    import pandas as pd
    from sklearn.model_selection import train_test_split


    train, test = train_test_split(dataset, test_size=0.2, random_state=seed)

    train, val = train_test_split(train, test_size=0.1, random_state=seed)
    
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"], ensure_channel_first=True, image_only=True), #0
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            SpatialPadd(keys=["image", "label"], spatial_size=(96, 96, 96), mode='constant'),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 96), #192, 192, 64
                pos=1,
                neg=1,
                num_samples=train_num_samples,
                image_key="image",
                image_threshold=0,
            ),
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
            RemapLabels(keys=["label"]),
            ToTensord(keys=["image", "label"]),
        ]
    )
    val_transforms =  Compose(
            [
                LoadImaged(keys=["image", "label"], ensure_channel_first=True, image_only=True),
                Spacingd(keys=["image", "label"], pixdim=(
                    1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                SpatialPadd(keys=["image", "label"], spatial_size=(96, 96, 96), mode='constant'),
                RemapLabels(keys=["label"]),
                ToTensord(keys=["image", "label"]),
            ]
        )
    return train, val, test, class_definitions, train_transforms, val_transforms

def amos(seed, train_num_samples):
    import json
    class_definitions = '/home/avisionguy/UniViLa/semantic_aware/classes_definitions_amos_iccv.csv'
    base_dir = '/home/avisionguy/Multi_Modality_Seg_v3/amos22'
    os.chdir(base_dir)
    
    f = open('/home/avisionguy/Multi_Modality_Seg_v3/amos22/dataset.json')      #JSON file is provided
    dataJson = json.load(f)
    train_list = dataJson['training']
    train_list = train_list[0:200]

    # print(len(train_list))
    test_list = dataJson['validation']
    test = test_list[0:100]

    # %%
    import pandas as pd
    from sklearn.model_selection import train_test_split


    train, val = train_test_split(train_list, test_size=0.1)

    
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"], ensure_channel_first=True, image_only=True), #0
            Spacingd(keys=["image", "label"], pixdim=( 
                1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            SpatialPadd(keys=["image", "label"], spatial_size=(96, 96, 96), mode='constant'),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=train_num_samples,
                image_key="image",
                image_threshold=0,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.50,
            ),
            RandAffined(
                keys=['image', 'label'],
                mode=('bilinear', 'nearest'),
                prob=1.0, spatial_size=(96, 96, 96),
                rotate_range=(0, 0, np.pi / 30),
                scale_range=(0.1, 0.1, 0.1)),
            ToTensord(keys=["image", "label"]),
        ]
    )
    val_transforms =  Compose(
            [
                LoadImaged(keys=["image", "label"], ensure_channel_first=True, image_only=True), #0
                Spacingd(keys=["image", "label"], pixdim=(
                    1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-175, a_max=250,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                SpatialPadd(keys=["image", "label"], spatial_size=(96, 96, 96), mode='constant'),
                ToTensord(keys=["image", "label"]),
            ]
        )
    return train, val, test, class_definitions, train_transforms, val_transforms


def brain(seed, train_num_samples):
    root_dir = "/home/avisionguy/UniViLa/Task01_BrainTumour"

    class_definitions = '/home/avisionguy/UniViLa/semantic_aware/classes_definitions_brain_iccv.csv'
    train_transforms = Compose(
        [
            # load 4 Nifti images and stack them together
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            SpatialPadd(keys=["image", "label"], spatial_size=(96,96,96), mode='constant'), # (192,192,64)
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96,96,96),
                pos=1,
                neg=1,
                num_samples=train_num_samples,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )
    train = DecathlonDataset(
        root_dir=root_dir,
        task="Task01_BrainTumour",
        transform=train_transforms,
        section="training",
        download=False,
        cache_rate=0.0,
        num_workers=4,
    )
    from torch.utils.data import Subset
    train = Subset(train, range(5))
    test = DecathlonDataset(
        root_dir=root_dir,
        task="Task01_BrainTumour",
        transform=val_transforms,
        section="validation",
        download=False,
        cache_rate=0.0,
        num_workers=4,
    
    )
    train, val = train_test_split(train, test_size=0.1, random_state=seed)
    
    return train, val, test, class_definitions
# %%
model_type = "mulmodseg" # natunetr, dinatunetr, unet, unet++, segresnet, nnunet, attunet, unetr, swinunetr, nnformer, medformer, mednext, universal, zept, mulmodseg
dataset = "amos" # whs_ct, whs_mri, amos, brain
seed = 42
natunetr = False

import os
import time
import sys
sys.path.append(os.path.dirname(__file__))
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import multiprocessing
import monai
from monai.transforms import (
    AsDiscrete, Compose,
)
import torch
from monai.data import Dataset, CacheDataset, PersistentDataset, DataLoader, decollate_batch
from monai.utils import first
num_workers = multiprocessing.cpu_count()

import random

from monai.networks.layers import Norm
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from utils import info_nce_loss, CE_loss
from dataset_grab import whs_ct, whs_mri, amos, brain
from model_grab import def_model
from pathlib import Path

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)



dataset_dir = '/home/avisionguy/UniViLa/weights'
cache_dir = '/home/avisionguy/UniViLa/cache'
device = torch.device("cuda:0")
train_num_samples = 1

if dataset == "whs_ct":
    train, val, test, class_definitions, train_transforms, val_transforms = whs_ct(seed, train_num_samples)
    classes_dict = {
    'background': 0, 'LV': 1., 'RV': 2., 'LA': 3., 'RA': 4., 'Myo': 5., 'AA': 6., 'PA': 7.,}
elif dataset == "whs_mri":
    train, val, test, class_definitions, train_transforms, val_transforms = whs_mri(seed, train_num_samples)
    classes_dict = {
    'background': 0, 'LV': 1., 'RV': 2., 'LA': 3., 'RA': 4., 'Myo': 5., 'AA': 6., 'PA': 7.,}    
elif dataset == "amos":
    train, val, test, class_definitions, train_transforms, val_transforms = amos(seed, train_num_samples)
    classes_dict = {
    'background': 0., 'spleen': 1., 'right kidney': 2., 'left kidney': 3., 'gallbladder': 4., 'esophagus': 5., 'liver': 6., 'stomach': 7., 'aorta': 8., 'inferior vena cava': 9., 'pancreas': 10., 'right adrenal gland': 11., 'left adrenal gland': 12., 'duodenum': 13., 'bladder': 14., 'prostate/uterus': 15.,
     }
elif dataset == "brain":
    train, val, test, class_definitions, train_transforms, val_transforms = brain(seed, train_num_samples)
    classes_dict = {
    'background': 0., 'Tumor core': 1, 'Whole tumor': 2, 'Enhancing tumor': 3,
     }
cache_dir = cache_dir + "/" + dataset   



num_channels = len(classes_dict)



train_ds = PersistentDataset(train, transform=train_transforms, cache_dir=cache_dir)
val_ds = PersistentDataset(data=val, transform=val_transforms, cache_dir=cache_dir)

train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers)

# %%
max_epochs = 300
val_interval = 5
        
VAL_AMP = True







    

in_channels = 1
seg_loss = monai.losses.DiceCELoss(to_onehot_y=True, softmax=True)

    
model = def_model(model_type, in_channels, num_channels, device)
# Initialize the optimizer with all the parameters, including from the enhanced embedding module
optimizer = torch.optim.AdamW(
    list(model.parameters()),
    lr=1e-4,
    weight_decay=1e-5
)


lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

# %%
# use amp to accelerate training
scaler = torch.GradScaler(device="cuda")
  
# enable cuDNN benchmark
torch.backends.cudnn.benchmark = True


best_metric = -1
best_metric_epoch = -1




post_pred = Compose([AsDiscrete(argmax=True, to_onehot=num_channels)])
post_label = Compose([AsDiscrete(to_onehot=num_channels)])        



# %%
total_start = time.time()


from tqdm import tqdm


for epoch in range(max_epochs):
    epoch_start = time.time()
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step_start = time.time()
        step += 1
        inputs, labels = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
        )
        
        #optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logit_map = model(inputs)
            # logit_map = model(inputs, "CT")
            if model_type == "unet++" or model_type == "segresnet":
                logit_map = logit_map[0]
            loss = seg_loss(logit_map, labels)
        scaler.scale(loss).backward()
        epoch_loss += loss.item()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    epoch_loss /= step
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():

            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model)
                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                dice_metric(y_pred=val_outputs, y=val_labels)


                

            metric = dice_metric.aggregate().item()
            dice_metric.reset()
            

            if metric > best_metric:        # change the sign to > for dice
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(
                    model.state_dict(),
                    os.path.join(dataset_dir, str(model_type)+"_"+str(dataset)+".pth"),
                )
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f"\nbest mean dice: {best_metric:.4f}"
                f" at epoch: {best_metric_epoch}"
            )
    print(f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")
total_time = time.time() - total_start


# %%
print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")


# %%
# loss_name = None
# model.load_state_dict(torch.load(os.path.join(dataset_dir, str(model_type)+"_whs_"+str(loss_name)+".pth")))
model.load_state_dict(torch.load(os.path.join(dataset_dir, str(model_type)+"_"+str(dataset)+".pth")))
model.eval()

from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDiceMetric
dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
hd95 = HausdorffDistanceMetric(include_background=False, percentile=95)        #95 HD
ASD = SurfaceDiceMetric(include_background=False, class_thresholds=[2]*(num_channels-1), use_subvoxels=True)       #Average surface distance

post_pred = Compose([AsDiscrete(argmax=True, to_onehot=num_channels)])
post_label = Compose([AsDiscrete(to_onehot=num_channels)])        
test_ds = PersistentDataset(data=test, transform=val_transforms, cache_dir=cache_dir)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)


model.eval()
dice_scores = []
hd95_scores = []
asd_scores = []

# with torch.no_grad():
#     for val_data in tqdm(test_loader):
#         val_inputs, val_labels = (
#             val_data["image"].to(device),
#             val_data["label"].to(device),
#         )

#         val_outputs = sliding_window_inference(
#             inputs=val_inputs, roi_size=(96, 96, 96), sw_batch_size=1,
#             predictor=model, device=device, overlap=0.25
#         )
#         val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
#         val_labels = [post_label(i) for i in decollate_batch(val_labels)]
        
#         dice_metric(y_pred=val_outputs, y=val_labels)
#         print(f"Dice: {dice_metric.aggregate().item()}")
#         # hd95(y_pred=val_outputs, y=val_labels)
#         # ASD(y_pred=val_outputs, y=val_labels)
        
#         # Append batch-wise results for mean and std calculation
#         dice_scores.extend(metric.cpu().tolist() for metric in dice_metric.get_buffer())  # Convert each tensor to a list
#         # hd95_scores.extend(metric.cpu().tolist() for metric in hd95.get_buffer())
#         # asd_scores.extend(metric.cpu().tolist() for metric in ASD.get_buffer())

#     # Convert to tensors for mean and std calculation
#     dice_scores = torch.tensor(dice_scores)  # Shape: [num_samples, num_classes]
#     # hd95_scores = torch.tensor(hd95_scores)
#     # asd_scores = torch.tensor(asd_scores)

#     # Calculate mean and std across samples for each class
#     dice_mean, dice_std = dice_scores.mean(dim=0), dice_scores.std(dim=0)
#     # hd95_mean, hd95_std = hd95_scores.mean(dim=0), hd95_scores.std(dim=0)
#     # asd_mean, asd_std = asd_scores.mean(dim=0), asd_scores.std(dim=0)

#     # Calculate overall mean and std across all classes
#     dice_mean_overall, dice_std_overall = dice_scores.mean(), dice_scores.std()
#     # hd95_mean_overall, hd95_std_overall = hd95_scores.mean(), hd95_scores.std()
#     # asd_mean_overall, asd_std_overall = asd_scores.mean(), asd_scores.std()


#     print(
#     f"Overall ({model_type}): Dice = {dice_mean_overall.item() * 100:.4f} ± {dice_std_overall.item() * 100:.4f}%, "
#     # f"HD95 = {hd95_mean_overall.item():.4f} ± {hd95_std_overall.item():.4f}mm, "
#     # f"ASD = {asd_mean_overall.item() * 100:.4f} ± {asd_std_overall.item() * 100:.4f}%")
#     )


with torch.no_grad():
    for val_data in tqdm(test_loader):
        val_inputs, val_labels = (
            val_data["image"].to(device),
            val_data["label"].to(device),
        )

        val_outputs = sliding_window_inference(
            inputs=val_inputs, roi_size=(96, 96, 96), sw_batch_size=1,
            predictor=model, device=device, overlap=0.25
        )
        val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
        val_labels = [post_label(i) for i in decollate_batch(val_labels)]
        

        dice_metric(y_pred=[i.cpu() for i in val_outputs], y=[i.cpu() for i in val_labels])
        hd95(y_pred=[i.cpu() for i in val_outputs], y=[i.cpu() for i in val_labels])
        ASD(y_pred=[i.cpu() for i in val_outputs], y=[i.cpu() for i in val_labels])

        
        torch.cuda.empty_cache()  # Clear memory
        
    metric = dice_metric.aggregate().item()
    print(f"Dice metric on test dataset {metric*100:.2f}")
    metrichd = hd95.aggregate().item()
    print(f"Hausdorff distance 95 on test dataset {metrichd:.2f}")
    metricASD = ASD.aggregate().item()
    print(f"NSD on test dataset {metricASD*100:.2f}")

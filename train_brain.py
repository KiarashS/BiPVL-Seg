# %%

model_type = "univila"   #unet++_with_text, dinatUNETR_with_text or swinunetr_with_text or swinunetrv2_with_text or univila
dataset = "brain"      # "whs_ct", whs_mri, brain or "amos"
seed = 42
natunetr = True
roi_size = (96,96,96)

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
    AsDiscrete, Compose, Activations
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
elif dataset == "amos":
    train, val, test, class_definitions, train_transforms, val_transforms = amos(seed, train_num_samples)
    classes_dict = {
    'background': 0., 'spleen': 1., 'right kidney': 2., 'left kidney': 3., 'gallbladder': 4., 'esophagus': 5., 'liver': 6., 'stomach': 7., 'aorta': 8., 'inferior vena cava': 9., 'pancreas': 10., 'right adrenal gland': 11., 'left adrenal gland': 12., 'duodenum': 13., 'bladder': 14., 'prostate/uterus': 15.,   
}
elif dataset == "whs_mri":
    train, val, test, class_definitions, train_transforms, val_transforms = whs_mri(seed, train_num_samples)
    classes_dict = {
    'background': 0, 'LV': 1., 'RV': 2., 'LA': 3., 'RA': 4., 'Myo': 5., 'AA': 6., 'PA': 7.,}   
elif dataset == "brain":
    train, val, test, class_definitions = brain(seed, train_num_samples)
    classes_dict = {
    'Tumor core': 1, 'Whole tumor': 2, 'Enhancing tumor': 3,
     }
    

cache_dir = cache_dir + "/" + dataset 
# train = train[:1]
# val = val[:1]


num_channels = len(classes_dict)

# %%
train_loader = DataLoader(train, batch_size=1, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val, batch_size=1, shuffle=False, num_workers=num_workers)

# %%
max_epochs = 300
val_interval = 5
        
VAL_AMP = True

# %%


# %%
from transformers import AutoTokenizer, AutoModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"



seg_loss = monai.losses.DiceCELoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)


# Define the Linear Projection Layer
projection_dim = 768  # This should match the embedding dimension of text_embeddings

# checkpoint = torch.load(os.path.join(dataset_dir, str(model_type)+str(dataset)+".pth"), weights_only=True)

# Initialize the module for both visual and text embeddings
if model_type == "unet++":
    cp_feats = nn.Linear(in_features=projection_dim, out_features=216, device=device) 
else:    
    cp_feats_visual = nn.Linear(in_features=27, out_features=128, device=device)
    cp_feats_text = nn.Linear(in_features=projection_dim, out_features=128, device=device)
    
    # Load the pre-trained model
    # cp_feats_visual.load_state_dict(checkpoint['cp_feats_visual'])
    # cp_feats_text.load_state_dict(checkpoint['cp_feats_text'])

in_channels = 4

model = def_model(model_type, in_channels, num_channels, device, text_encoder="pubmedbert")    ##bert, biobert, clinicalbert, pubmedbert
# model.load_state_dict(checkpoint['model'])

# Initialize the optimizer with all the parameters, including from the enhanced embedding module
optimizer = torch.optim.AdamW(
    list(model.parameters()) + 
    list(cp_feats_visual.parameters()) +
    list(cp_feats_text.parameters()),  
    lr=1e-4,
    weight_decay=1e-5
)


lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
# dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")     # to get the dice values of all classes

# %%
# use amp to accelerate training
scaler = torch.GradScaler(device="cuda")
  
# enable cuDNN benchmark
torch.backends.cudnn.benchmark = True


best_metric = -1
best_metric_epoch = -1




post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
# post_label = Compose([AsDiscrete(to_onehot=num_channels)])        





    

# %%
total_start = time.time()

# cls__semantic_txt_amos_encoding = torch.load("/home/avisionguy/Multi_Modality_Seg_v3/semantic_aware/cls__semantic_txt_amos_encoding.pth").to(device)
# class_definitions = pd.read_csv('/home/avisionguy/Multi_Modality_Seg_v3/semantic_aware/classes_definitions_whs.csv')


from tqdm import tqdm
# Define learnable task uncertainties
log_sigma1 = torch.nn.Parameter(torch.tensor(0.0, device=device))  # Segmentation loss uncertainty
log_sigma2 = torch.nn.Parameter(torch.tensor(0.0, device=device))  # Feature consistency loss uncertainty
log_sigma3 = torch.nn.Parameter(torch.tensor(0.0, device=device))  # InfoNCE loss uncertainty


# log_sigma1 = torch.nn.Parameter(checkpoint['log_sigma1'].to(device))
# log_sigma2 = torch.nn.Parameter(checkpoint['log_sigma2'].to(device))
# log_sigma3 = torch.nn.Parameter(checkpoint['log_sigma3'].to(device))

# Register parameters in optimizer
optimizer.add_param_group({'params': [log_sigma1, log_sigma2, log_sigma3]})

for epoch in range(max_epochs):
    epoch_start = time.time()
    print("-" * 10)
    # print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0.0
    epoch_loss1 = 0.0  # Initialize to accumulate loss1 over the epoch
    epoch_loss2 = 0.0  # Initialize to accumulate loss2 over the epoch
    epoch_loss3 = 0.0  # Initialize to accumulate loss3 over the epoch
    step = 0
    for batch_data in train_loader:
        step_start = time.time()
        step += 1
        inputs, labels = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
        )
        
        optimizer.zero_grad()
        
        with torch.autocast(device_type="cuda"):
           
            if model_type == "unet++":
                visual_embeddings, text_embeddings, logit_map = model(
                    x=inputs,  
                    class_definitions=class_definitions,
                    device=device
                )
                logit_map = logit_map[0]
            else:
                visual_embeddings, text_embeddings, logit_map = model(
                    x=inputs,  
                    class_definitions=class_definitions,
                    device=device
                )
            
            
            
            h, w, d = visual_embeddings.shape[-3:]
            # print(visual_embeddings.shape, logit_map.shape)
            #visual_embeddings torch.Size([train_num_samples, num_channels, 3, 3, 3]), logits torch.Size([train_num_samples, num_channels, 96, 96, 96])
            logits_downsampled = F.interpolate(logit_map, size=(h, w, d), mode='trilinear', align_corners=False)
            visual_embeddings = visual_embeddings.view(train_num_samples, num_channels, -1)
            logits_downsampled = logits_downsampled.view(train_num_samples, num_channels, -1)
            # Calculate segmentation loss (loss1)
            loss1 = seg_loss(logit_map, labels)
            
            loss2 = CE_loss(visual_embeddings, logits_downsampled)

            # if epoch < 20:
            #     loss3 = 0.0  # Ignore InfoNCE for first 50 epochs
            #     loss = loss1 + 0.1 * loss2
            # elif 20 <= epoch < 30:
            #     loss3 = info_nce_loss(visual_embeddings, text_embeddings, cp_feats_visual, cp_feats_text, device)
            #     loss = loss1 + 0.1 * loss2 + 0.05 * loss3  # Small weight to ease alignment learning
            # elif 30 <= epoch < 40:
            #     loss3 = info_nce_loss(visual_embeddings, text_embeddings, cp_feats_visual, cp_feats_text, device)
            #     loss = loss1 + 0.1 * loss2 + 0.2 * loss3  # Gradually increasing weight
            # else:
            #     loss3 = info_nce_loss(visual_embeddings, text_embeddings, cp_feats_visual, cp_feats_text, device)
            #     loss = loss1 + 0.1 * loss2 + 0.3 * loss3  # Full weight after epoch 70
            
            loss3 = info_nce_loss(visual_embeddings, text_embeddings, cp_feats_visual, cp_feats_text, device, 0.2)
            # Compute total loss using homoscedastic uncertainty weighting
            loss = (
                (loss1 / (2 * torch.exp(log_sigma1))) + log_sigma1 +
                (loss2 / (2 * torch.exp(log_sigma2))) + log_sigma2 +
                (loss3 / (2 * torch.exp(log_sigma3))) + log_sigma3
            )


        # Backpropagation
        scaler.scale(loss).backward()
        
        
        epoch_loss1 += loss1.item()
        epoch_loss2 += loss2.item()
        epoch_loss3 += loss3 if isinstance(loss3, float) else loss3.item()
        epoch_loss += loss.item()
        
        # Optimizer steps
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        

        lr_scheduler.step()

    # After the epoch ends
    epoch_loss /= step
    
    epoch_loss1 /= step
    epoch_loss2 /= step
    epoch_loss3 /= step


    # Print average loss per epoch
    # print(f"Epoch {epoch + 1} - Average Loss1 (Segmentation): {epoch_loss1:.4f}, "
    #     f"Average Loss2 (InfoNCE): {epoch_loss2:.4f}, "
    #     f"Average Total Loss: {epoch_loss:.4f}")
 

  
    print(f"Epoch {epoch + 1}, "
        f"Average Loss1 (DiceCE): {epoch_loss1:.4f}, "
        f"Average Loss2 (CE): {epoch_loss2:.4f}, "
        f"Average Loss3 (InfoNCE): {epoch_loss3:.4f}, "
        f"Average Total Loss: {epoch_loss:.4f}")


    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            for val_data in tqdm(val_loader):
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )

                
                
                val_outputs = sliding_window_inference(inputs = val_inputs, roi_size = roi_size, sw_batch_size = 1, predictor = model, 
                                                        natunetr = natunetr, class_definitions = class_definitions, device = device)
                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                # val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                dice_metric(y_pred=val_outputs, y=val_labels)
          

            metric = dice_metric.aggregate().item()
         

            dice_metric.reset()

            
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save({
                    'model': model.state_dict(),
                    'cp_feats_visual': cp_feats_visual.state_dict(),
                    'cp_feats_text': cp_feats_text.state_dict(),
                    'log_sigma1': log_sigma1.detach().cpu(),  # Save as a tensor
                    'log_sigma2': log_sigma2.detach().cpu(),
                    'log_sigma3': log_sigma3.detach().cpu(),
                # }, os.path.join(dataset_dir, "best_metric_model"+str(model_type)+"_whs_v4_1"+str(version)+".pth"))
                }, os.path.join(dataset_dir,str(model_type)+str(dataset)+".pth"))
                # print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f"\nbest mean dice: {best_metric:.4f}"
                f" at epoch: {best_metric_epoch}"
            )
    print(f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")
total_time = time.time() - total_start

print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")


#inference
# Load the saved checkpoint
checkpoint = torch.load(os.path.join(dataset_dir, str(model_type)+str(dataset)+".pth"), weights_only=True)
model.load_state_dict(checkpoint['model'])


from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDiceMetric
dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
hd95 = HausdorffDistanceMetric(include_background=False, percentile=95)        #95 HD
ASD = SurfaceDiceMetric(include_background=False, class_thresholds=[2]*(num_channels-1), use_subvoxels=True)       #Average surface distance
       

test_loader = DataLoader(test, batch_size=1, shuffle=False, num_workers=num_workers)


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

        # val_outputs = sliding_window_inference(
        #     inputs=val_inputs, roi_size=(96, 96, 96), sw_batch_size=1,
        #     predictor=model, device=device, overlap=0.25
        # )
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

        val_outputs = sliding_window_inference(inputs = val_inputs, roi_size = roi_size, sw_batch_size = 1, predictor = model, 
                                                        natunetr = natunetr, class_definitions = class_definitions, device = device, overlap=0.75)
        val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
        # val_labels = [post_label(i) for i in decollate_batch(val_labels)]
        

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


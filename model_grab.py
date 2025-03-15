import torch
from monai.networks.nets import UNet, BasicUNetPlusPlus_text, AttentionUnet, UNETR, SwinUNETR, SwinUNETR_text, NATUNETR_TEXT, UniViLa, NATUNETR, BasicUNetPlusPlus, DynUNet, SegResNetVAE
from networks.nnFormer.nnFormer_seg import nnFormer
from monai.networks.layers import Norm
from networks.medformer import MedFormer
from networks.MedNeXt.nnunet_mednext.network_architecture.mednextv1.MedNextV1 import MedNeXt
from networks.universal.Universal_model import Universal_model
from networks.zept.ZePT_pred import ZePT
from networks.CAT.CAT_pred import CAT
from networks.mulmodseg.Universal_model import Universal_model as mulmodseg  

def def_model(model_type, in_channels, num_channels, device, text_encoder = None):
    if model_type == "dinatUNETR_with_text":
        model = NATUNETR_TEXT(
            in_channels=in_channels,
            out_channels=num_channels,
            kernel_size=(7, 7, 7, 3, 3),
            dilations=[[1, 6], [1, 2, 1], [1, 1, 1, 1], [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2], [1, 1, 1, 1, 1]],     #based on depths (2, 3, 4, 18, 5),
            feature_size=48,
            device=device,
            use_v2=True,
        ).to(device)
        # state_dicts = torch.load("/home/avisionguy/Multi_Modality_Seg_v3/amos22/best_metric_modeldinatUNETRv4_1.pth")
        # model.load_state_dict(state_dicts['model'])

    elif model_type == "swinunetr_with_text":
        model = SwinUNETR_text(
            img_size=(96, 96, 96),
            in_channels=in_channels,
            out_channels=num_channels,
            feature_size=48,
            use_checkpoint=True,
            use_v2 = False,
        ).to(device)
    elif model_type == "swinunetrv2_with_text":
        model = SwinUNETR_text(
            img_size=(96, 96, 96),
            in_channels=in_channels,
            out_channels=num_channels,
            feature_size=48,
            use_checkpoint=True,
            use_v2 = True,
        ).to(device)
    elif model_type == "unet++_with_text":
        model = BasicUNetPlusPlus_text(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=num_channels,
            features=(32, 32, 64, 128, 256, 32),
        ).to(device)
    elif model_type == "univila":
        model = UniViLa(
            img_size=(96, 96, 96),
            in_channels=in_channels,
            out_channels=num_channels,
            feature_size=48,
            device=device,
            text_encoder=text_encoder,      #bert, biobert, clinicalbert, pubmedbert
            use_checkpoint=True,
            use_v2 = True,
        ).to(device)
    elif model_type == "natunetr":
        model = NATUNETR(
            in_channels=in_channels,
            out_channels=num_channels,
            kernel_size=3,
            dilations=None,
            feature_size=48,
        ).to(device)
    elif model_type == "dinatunetr":
        model = NATUNETR(
            in_channels=in_channels,
            out_channels=num_channels,
            kernel_size=3,
            dilations=[[1, 16], [1, 8], [1, 2, 1, 3, 1, 4], [1, 2], [1,1]],    #settings 2
            feature_size=48,
        ).to(device)
    elif model_type == "unet":
        model = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=num_channels,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        ).to(device)
    elif model_type == "segresnet":
        model = SegResNetVAE(
            input_image_size =(96, 96, 96),
            spatial_dims= 3, 
            in_channels=in_channels,
            out_channels=num_channels,
        ).to(device)
    elif model_type == "unet++":
        model = BasicUNetPlusPlus(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=num_channels,
            features=(32, 32, 64, 128, 256, 32),
        ).to(device)
    elif model_type == "attunet":
        model = AttentionUnet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=num_channels,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
        ).to(device)    
    elif model_type == "nnunet":
        model = DynUNet(spatial_dims=3, 
                        in_channels=in_channels, 
                        out_channels=num_channels,
                        kernel_size = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]], 
                        strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]], 
                        upsample_kernel_size =[[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
                        ).to(device) 
    elif model_type == "unetr":
        model = UNETR(
            img_size=(96, 96, 96),
            in_channels=in_channels,
            out_channels=num_channels,
            feature_size=48,
            norm_name='batch',
        ).to(device)
    elif model_type == "swinunetr":
        model = SwinUNETR(
            img_size=(96, 96, 96),
            in_channels=in_channels,
            out_channels=num_channels,
            feature_size=48,
            use_checkpoint=True,
            # use_v2=True,
        ).to(device)
    elif model_type == "swinunetrv2":
        model = SwinUNETR(
            img_size=(96, 96, 96),
            in_channels=in_channels,
            out_channels=num_channels,
            feature_size=48,
            use_checkpoint=True,
            use_v2=True,
        ).to(device)
    elif model_type == "nnformer":
        model = nnFormer(input_channels=in_channels, 
                         num_classes=num_channels,
                         ).to(device)
    elif model_type == 'medformer':
        model = MedFormer(in_chan=in_channels, num_classes=num_channels, base_chan=32, conv_block='BasicBlock',
                         map_size=[2, 6, 6], 
                         scale=[[1,2,2], [1,2,2], [2,2,2], [2,2,2]], 
                         conv_num=[2,0,0,0, 0,0,2,2], trans_num=[0,2,2,2, 2,2,0,0], 
                         num_heads=[1,4,4,4, 4,4,1,1], fusion_depth=2, fusion_dim=256, 
                         fusion_heads=4, expansion=4, attn_drop=0., proj_drop=0., 
                         proj_type='depthwise', norm='in', act='gelu', 
                         kernel_size=[[1,3,3], [1,3,3], [3,3,3], [3,3,3], [3,3,3]]).to(device)
    elif model_type == 'mednext':
        model = MedNeXt(
          in_channels= in_channels,
          n_channels= 32,
          n_classes= num_channels,
          exp_r=4,                           # Expansion ratio in Expansion Layer
          kernel_size=5,                     # Kernel Size in Depthwise Conv. Layer
          enc_kernel_size=None,              # (Separate) Kernel Size in Encoder
          dec_kernel_size=None,              # (Separate) Kernel Size in Decoder
          deep_supervision=False,           # Enable Deep Supervision
          do_res=True,                     # Residual connection in MedNeXt block
          do_res_up_down=True,             # Residual conn. in Resampling blocks
          checkpoint_style=None,            # Enable Gradient Checkpointing
          block_counts=[2,2,2,2,2,2,2,2,2], # Depth-first no. of blocks per layer 
          norm_type = 'group',                      # Type of Norm: 'group' or 'layer'
          dim = '3d',                                # Supports `3d', '2d' arguments
        ).to(device)
    elif model_type == 'universal':
        model = Universal_model(
            in_channels=in_channels,
            out_channels=num_channels,
            img_size = (96, 96, 96),
            backbone="swinunetr",
            encoding="word_embedding",
        ).to(device)
        word_embedding = torch.load("/home/avisionguy/CLIP-Driven-Universal-Model/pretrained_weights/txt_encoding_brain.pth")
        model.organ_embedding.data = word_embedding.float()
    elif model_type == 'mulmodseg':
        model = mulmodseg(
            in_channels=in_channels,
            out_channels=num_channels,
            img_size = (96, 96, 96),
            backbone="swinunetr",
            encoding="word_embedding",
        ).to(device)
        word_embedding = torch.load("/home/avisionguy/CLIP-Driven-Universal-Model/pretrained_weights/txt_encoding_brain.pth")
        model.organ_embedding.data = word_embedding.float()
    elif model_type == 'zept':
        model = ZePT(
            in_channels=in_channels,
            out_channels=num_channels,
            img_size = (96, 96, 96),
            backbone="swinunetr",
        ).to(device)
    elif model_type == 'cat':
        anatomical_prompts_paths = "/home/avisionguy/CLIP-Driven-Universal-Model/data/anatomical_prompts_paths.json"
        textual_prompts_paths = "/home/avisionguy/CLIP-Driven-Universal-Model/data/text_prompt_path.json"
        model = CAT(img_size=(96, 96, 96),
                    in_channels=1,
                    out_channels=num_channels,
                    backbone="swinunetr",
                    anatomical_prompts_paths=anatomical_prompts_paths,
                    text_prompt_path=textual_prompts_paths,
                    ).to(device)   
    return model
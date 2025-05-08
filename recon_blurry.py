import os
import sys
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from accelerate import Accelerator
import numpy as np 
# SDXL unCLIP requires code from https://github.com/Stability-AI/generative-models/tree/main
sys.path.append('generative_models/')
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder # bigG embedder
from sklearn.preprocessing import StandardScaler
from einops import rearrange
# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True

# custom functions #
import utils
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"



parser = argparse.ArgumentParser(description="Model Training Configuration")
parser.add_argument(
    "--model_name", type=str, default="testing",
    help="name of model, used for ckpt saving and wandb logging (if enabled)",
)
parser.add_argument(
    "--subj",type=int, default=1, choices=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
    help="Validate on which subject?",
)
parser.add_argument(
    "--use_prior",action=argparse.BooleanOptionalAction,default=False,
    help="whether to train diffusion prior (True) or just rely on retrieval part of the pipeline (False)",
)
parser.add_argument(
    "--batch_size", type=int, default=10,
    help="Batch size can be increased by 10x if only training retreival submodule and not diffusion prior",
)
parser.add_argument(
    "--mixup_pct",type=float,default=.33,
    help="proportion of way through training when to switch from BiMixCo to SoftCLIP",
)
parser.add_argument(
    "--blurry_recon",action=argparse.BooleanOptionalAction,default=True,
    help="whether to output blurry reconstructions",
)
parser.add_argument(
    "--blur_scale",type=float,default=.5,
    help="multiply loss from blurry recons by this number",
)
parser.add_argument(
    "--clip_scale",type=float,default=1.,
    help="multiply contrastive loss by this number",
)
parser.add_argument(
    "--prior_scale",type=float,default=30,
    help="multiply diffusion prior loss by this",
)
parser.add_argument(
    "--use_image_aug",action=argparse.BooleanOptionalAction,default=False,
    help="whether to use image augmentation",
)
parser.add_argument(
    "--num_epochs",type=int,default=150,
    help="number of epochs of training",
)
parser.add_argument(
    "--n_blocks",type=int,default=4,
)
parser.add_argument(
    "--hidden_dim",type=int,default=4096,
)
parser.add_argument(
    "--lr_scheduler_type",type=str,default='cycle',choices=['cycle','linear'],
)
parser.add_argument(
    "--ckpt_saving",action=argparse.BooleanOptionalAction,default=True,
)
parser.add_argument(
    "--seed",type=int,default=42,
)
parser.add_argument(
    "--max_lr",type=float,default=3e-4,
)
parser.add_argument(
    "--use_text",action=argparse.BooleanOptionalAction,default=False,
)
parser.add_argument(
    "--fps",type=int,default=6,
)
parser.add_argument(
    "--cudas",type=int
    ,default=0,
)
parser.add_argument(
    "--test_id",type=int,default=0,
)
parser.add_argument(
    "--clip_seq_dim",type=int,default=32,
)
parser.add_argument(
    "--backbone",type=str,default='mlp',
)

args = parser.parse_args()

# create global variables without the args prefix
for attribute_name in vars(args).keys():
    globals()[attribute_name] = getattr(args, attribute_name)
    
# seed all random functions
utils.seed_everything(seed)
model_name =f'video_subj0{subj}_lPR_{test_id}_{backbone}'

outdir = os.path.abspath(f'neuroclips/outputs/blurry/')
if not os.path.exists(outdir) and ckpt_saving:
    os.makedirs(outdir,exist_ok=True)
    
### Multi-GPU config ###
local_rank = os.getenv('RANK')
if local_rank is None: 
    local_rank = 0
else:
    local_rank = int(local_rank)
print("LOCAL RANK ", local_rank)  

data_type = torch.float16 # change depending on your mixed_precision
num_devices = torch.cuda.device_count()
if num_devices==0: num_devices = 1

# First use "accelerate config" in terminal and setup using deepspeed stage 2 with CPU offloading!
torch.cuda.set_device(cudas)
accelerator = Accelerator(split_batches=False, mixed_precision="fp16")

print("PID of this process =",os.getpid())
device = accelerator.device
#device = 'cuda:0'
print("device:",device)
world_size = accelerator.state.num_processes
distributed = not accelerator.state.distributed_type == 'NO'
num_devices = torch.cuda.device_count()
if num_devices==0 or not distributed: num_devices = 1
num_workers = num_devices
print(accelerator.state)

print("distributed =",distributed, "num_devices =", num_devices, "local rank =", local_rank, "world size =", world_size, "data_type =", data_type)
print = accelerator.print # only print if local_rank=0



class NeuroclipDataset(torch.utils.data.Dataset):
    def __init__(self,eeg):
        self.eeg = eeg
        
        
    def __len__(self):
        return len(self.eeg)

    def __getitem__(self, index):
        return self.eeg[index]
    

subj_list = [subj]
seq_len = 62
clip_seq_dim = 32
clip_emb_dim = 768
hidden_dim =400



if blurry_recon:
    from diffusers import AutoencoderKL    
    autoenc = AutoencoderKL(
        down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
        up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
        block_out_channels=[128, 256, 512, 512],
        layers_per_block=2,
        sample_size=256,
    )
    ckpt = torch.load(f'/userhome/zhoutianyi/Zhoutianyi/Mutilmodel/huggingface/pscotti-neuroclips/sd_image_var_autoenc.pth')
    autoenc.load_state_dict(ckpt)
    
    autoenc.eval()
    autoenc.requires_grad_(False)
    autoenc.to(device)
    utils.count_params(autoenc)

class Neuroclips(nn.Module):
    def __init__(self):
        super(Neuroclips, self).__init__()
    def forward(self, x):
        return x
        
model = Neuroclips()



from Perception import Perception_Reconstruction, Inception_Extension
model.backbone = Perception_Reconstruction(h=clip_emb_dim, in_dim=hidden_dim, seq_len=clip_seq_dim, n_blocks=n_blocks,
                          clip_size=clip_emb_dim, out_dim=clip_emb_dim, 
                          blurry_recon=blurry_recon, clip_scale=clip_scale)
model.eeg = Inception_Extension(h=256, in_dim=hidden_dim, out_dim=hidden_dim, expand=fps*2, seq_len=seq_len)
utils.count_params(model.backbone)
utils.count_params(model.eeg)
utils.count_params(model)


print("\n---resuming from last.pth ckpt---\n")
checkpoint = torch.load(f'/userhome/zhoutianyi/Zhoutianyi/Mutilmodel/Models/EEG/EEG2Video/EEG2Video2/neuroclips/outputs/PR/checkpoints/mlp/video_subj0{subj}_low_levelsubj{subj}_{test_id}_last_epoch.pth', map_location='cpu')['model_state_dict']
model.load_state_dict(checkpoint, strict=True)
del checkpoint

###################### data structures ################# 
load_npy = np.load(f'data/SEED-DV/Segmented_Rawf_200Hz_2s/sub{subj}.npy')
All_train = rearrange(load_npy,'a b c d e -> a (b c) d e')
del load_npy
print(All_train.shape)

test_eeg =All_train[test_id]
test_eeg = test_eeg.reshape(test_eeg.shape[0],62*400)
normalize = StandardScaler()
normalize.fit(test_eeg)
test_eeg = normalize.transform(test_eeg)  
C=62
T=400
test_eeg = test_eeg.reshape(test_eeg.shape[0], C, T)
test_dataset = NeuroclipDataset(test_eeg)
test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False,pin_memory=True)


epoch = 0
torch.cuda.empty_cache()

model = accelerator.prepare(model)
print(f"{model_name} starting with epoch {epoch} / {num_epochs}")
progress_bar = tqdm(range(epoch,num_epochs), ncols=1200, disable=(local_rank!=0))
model.eval()
all_blurryrecons = None
all_blurryembedding = None
if local_rank==0:
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=data_type): 
        for test_i, eeg in enumerate(test_dl):  
            # all test samples should be loaded per batch such that test_i should never exceed 0
            ## Average same-image repeats ##
            
            eeg = eeg.half()

            loss=0.
                            
            eeg = eeg.to(device)

            eeg = model.eeg(eeg)
                
            blurry_image_enc_ = model.backbone(eeg, time = batch_size*fps*2)
            
            for j in range(int(batch_size)):
                embedding = blurry_image_enc_[0][7+j*12].unsqueeze(0)
                
                if all_blurryembedding is None:
                    
                    all_blurryembedding = torch.Tensor(embedding)
                else:
                    all_blurryembedding = torch.vstack((all_blurryembedding, torch.Tensor(embedding)))
                
            
            print(all_blurryembedding.shape)
            if blurry_recon:
                image_enc_pred, _ = blurry_image_enc_
                blurry_recon_images = (autoenc.decode(image_enc_pred/0.18215).sample / 2 + 0.5).clamp(0,1)

            for i in range(len(eeg)):
                im = torch.Tensor(blurry_recon_images[i])
                if all_blurryrecons is None:
                    all_blurryrecons = im[None].cpu()
                else:
                    all_blurryrecons = torch.vstack((all_blurryrecons, im[None].cpu()))
            
            print(all_blurryrecons.shape)
    print(all_blurryrecons.shape)
    print(all_blurryembedding.shape)
    torch.save(all_blurryrecons, outdir+f'/{model_name}_PR.pt')
    torch.save(all_blurryembedding, outdir+f'/{model_name}_PR_embeddings.pt')
'''
要解决的问题
1. images 怎么转换
2. reconstructor 怎么下载
3. loss 怎么改

'''


import torch.nn.functional as F
import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
import gc
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from accelerate import Accelerator
from einops import rearrange
# SDXL unCLIP requires code from https://github.com/Stability-AI/generative-models/tree/main
sys.path.append('generative_models/')
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder # bigG embedder
from sklearn.preprocessing import StandardScaler
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
    "--subj",type=int, default=1, choices=[1,2,3],
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
    "--max_lr",type=float,default=1e-3,
)
parser.add_argument(
    "--use_text",action=argparse.BooleanOptionalAction,default=False,
)
parser.add_argument(
    "--fps",type=int,default=6,
)
parser.add_argument(
    "--test_id",type=int,default=0,
)
parser.add_argument(
    "--clip_seq_dim",type=int,default=32,
)
parser.add_argument(
    "--cudas",type=int
    ,default=0,
)
parser.add_argument(
    "--backbone_model",type=str
    ,default='neuroclips',
)
args = parser.parse_args()

# seed all random functions
utils.seed_everything(args.seed)
model_name =f'video_subj0{args.subj}_low_level_{args.max_lr}_'

outdir=f"./outputs/PR/checkpoints/{args.backbone_model}/"
if not os.path.exists(outdir) and args.ckpt_saving:
    os.makedirs(outdir,exist_ok=True)

save_path = os.path.join(outdir,model_name+f'subj{args.subj}_{args.test_id}.pth')

if args.use_image_aug or args.blurry_recon:
    import kornia
    from kornia.augmentation.container import AugmentationSequential
if args.use_image_aug:
    img_augment = AugmentationSequential(
        kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.3),
        same_on_batch=False,
        data_keys=["input"],
    )

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

torch.cuda.set_device(args.cudas)
# First use "accelerate config" in terminal and setup using deepspeed stage 2 with CPU offloading!
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
    def __init__(self,eeg,image,text):
        self.eeg = eeg
        self.image=image
        self.text = text
    def __len__(self):
        return len(self.eeg)

    def __getitem__(self, index):
        return self.eeg[index],self.image[index],self.text[index]

subj_list = [args.subj]
seq_len = 62
clip_seq_dim = 32
clip_emb_dim = 768
hidden_dim =400

if args.blurry_recon:
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

    from autoencoder.convnext import ConvnextXL

    cnx = ConvnextXL(f'/userhome/zhoutianyi/Zhoutianyi/Mutilmodel/huggingface/pscotti-neuroclips/convnext_xlarge_alpha0.75_fullckpt.pth')
    cnx.requires_grad_(False)
    cnx.eval()
    cnx.to(device)


    mean = torch.tensor([0.485, 0.456, 0.406]).to(device).reshape(1,3,1,1)
    std = torch.tensor([0.228, 0.224, 0.225]).to(device).reshape(1,3,1,1)

    blur_augs = AugmentationSequential(
        kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
        kornia.augmentation.RandomGrayscale(p=0.1),
        kornia.augmentation.RandomSolarize(p=0.1),
        kornia.augmentation.RandomResizedCrop((224,224), scale=(.9,.9), ratio=(1,1), p=1.0),
        data_keys=["input"],
    )

class Neuroclips(nn.Module):
    def __init__(self):
        super(Neuroclips, self).__init__()
    def forward(self, x):
        return x

model = Neuroclips()



from Perception import Perception_Reconstruction, Inception_Extension
model.backbone = Perception_Reconstruction(h=clip_emb_dim, in_dim=hidden_dim, seq_len=clip_seq_dim, n_blocks=args.n_blocks,
                          clip_size=clip_emb_dim, out_dim=clip_emb_dim,
                          blurry_recon=args.blurry_recon, clip_scale=args.clip_scale)
model.eeg = Inception_Extension(h=256, in_dim=hidden_dim, out_dim=hidden_dim, expand=args.fps*2, seq_len=seq_len)
utils.count_params(model.backbone)
utils.count_params(model.eeg)
utils.count_params(model)

# test on subject 1 with fake data


for param in model.parameters():
    param.requires_grad_(True)
def prepare_optimizier(model,num_iterations_per_epoch):
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.max_lr)

    if args.lr_scheduler_type == 'linear':
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            total_iters=int(np.floor(args.num_epochs*num_iterations_per_epoch)),
            last_epoch=-1
        )
    elif args.lr_scheduler_type == 'cycle':
        total_steps=int(np.floor(args.num_epochs*num_iterations_per_epoch))
        print("total_steps", total_steps)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.max_lr,
            total_steps=total_steps,
            final_div_factor=1000,
            last_epoch=-1, pct_start=2/args.num_epochs
        )

    return optimizer, lr_scheduler

def save_ckpt(save_path,epoch,optimizer,lr_scheduler,losses,test_losses,lrs,last_epoch=False):
    if last_epoch is True:
        ckpt_path = os.path.join(outdir,model_name+f'subj{args.subj}_{args.test_id}_last_epoch.pth')
    else:
        ckpt_path = save_path
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save({
            'epoch': epoch,
            'model_state_dict': unwrapped_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'train_losses': losses,
            'test_losses': test_losses,
            'lrs': lrs,
            }, ckpt_path)
    print(f"\n---saved {save_path}\n")

def load_ckpt(tag,load_lr=True,load_optimizer=True,load_epoch=True,strict=True,outdir=outdir,multisubj_loading=False):
    print(f"\n---loading {outdir}/{tag}.pth ckpt---\n")
    checkpoint = torch.load(outdir+'/last.pth', map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    if multisubj_loading: # remove incompatible ridge layer that will otherwise error
        state_dict.pop('ridge.linears.0.weight',None)
    model.load_state_dict(state_dict, strict=strict)
    if load_epoch:
        globals()["epoch"] = checkpoint['epoch']
        print("Epoch",epoch)
    if load_optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if load_lr:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    del checkpoint

def write_scores_to_csv(csv_file_path ,epoch,pixcorr):
    import csv
    file_exists = False
    try:
        with open(csv_file_path, mode='r', newline='') as f:
            file_exists = True
    except FileNotFoundError:
        pass

    # 如果文件不存在或需要写入标题行
    if not file_exists:
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['epoch', 'pixcorr'])
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch, pixcorr])

print("\nDone with model preparations!")
num_params = utils.count_params(model)

########### perpare for training #######################
load_npy = np.load(f'data/SEED-DV/Segmented_Rawf_200Hz_2s/sub{args.subj}.npy')
images = torch.load('/userhome/zhoutianyi/Zhoutianyi/dataset/SEED-DV/video_torch_12.pt',map_location='cpu')
texts = torch.load('data/SEED-DV/text_emb_768.pt',map_location='cpu')
# eeg shape [7,40,5,62,200] images shape [7,200,6,768],text shape [7,200,768]
All_train = rearrange(load_npy,'a b c d e -> a (b c) d e')
images = rearrange(images,'a b e f c h w -> a (b e ) f c h w')
del load_npy
print(All_train.shape)
Top_1 = []
Top_K = []


train_eeg = np.empty((0,62,400))
train_images = torch.empty((0,12,3,512,512))
train_texts = torch.empty((0,768))
test_eeg =All_train[args.test_id]
test_images=images[args.test_id] #[6,200,768]
test_texts = texts[args.test_id]
for i in range(7):
    if i == args.test_id:
        continue 
    train_eeg = np.concatenate((train_eeg,All_train[i].reshape(200,62,400)))
    train_images = torch.cat((train_images,images[i].reshape(200,12,3,512,512)))
    train_texts = torch.cat((train_texts,texts[i].reshape(200,768)))

train_eeg = train_eeg.reshape(train_eeg.shape[0],62*400)
test_eeg = test_eeg.reshape(test_eeg.shape[0],62*400)
normalize = StandardScaler()
normalize.fit(train_eeg)
train_eeg = normalize.transform(train_eeg)  
normalize = StandardScaler()
normalize.fit(test_eeg)
test_eeg = normalize.transform(test_eeg)   
C=62
T=400
train_eeg = train_eeg.reshape(train_eeg.shape[0], C, T)
test_eeg = test_eeg.reshape(test_eeg.shape[0], C, T)

train_dataset = NeuroclipDataset(train_eeg, train_images, train_texts)
test_dataset = NeuroclipDataset(test_eeg, test_images, test_texts)
train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False,pin_memory=True)
test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False,pin_memory=True)

print("\nDone with model preparations!")
num_params = utils.count_params(model)
num_samples_per_epoch = len(train_eeg) // num_devices 
num_iterations_per_epoch = num_samples_per_epoch // (args.batch_size)
save_path = os.path.join(outdir,model_name+f'subj{args.subj}_{args.test_id}.pth')
name = f'subj{args.subj}_{args.test_id}'
optimizer,lr_scheduler = prepare_optimizier(model, num_iterations_per_epoch)

epoch = 0
losses, test_losses, lrs = [], [], []
best_test = 0
torch.cuda.empty_cache()
model, optimizer, train_dl, lr_scheduler = accelerator.prepare(model, optimizer, train_dl, lr_scheduler)
# leaving out test_dl since we will only have local_rank 0 device do evals



print(f"{model_name} starting with epoch {epoch} / {args.num_epochs}")
name = f'subj{args.subj}_{args.test_id}'
csv_file_path=f"neuroclips/outputs/PR/logs/{args.backbone_model}/evaluation_scores_{model_name}_{name}.csv"
progress_bar = tqdm(range(epoch,args.num_epochs), ncols=1200, disable=(local_rank!=0))
test_image, test_eeg = None, None
mse = nn.MSELoss()
l1 = nn.L1Loss()
soft_loss_temps = utils.cosine_anneal(0.004, 0.0075, args.num_epochs - int(args.mixup_pct * args.num_epochs))
text_scale = 0.3
if num_devices!=0 and distributed:
    model = model.module
del train_eeg,train_images,train_texts,test_eeg,test_images,test_texts,All_train,images

for epoch in progress_bar:
    model.train()

    fwd_percent_correct = 0.
    bwd_percent_correct = 0.
    
    loss_clip_total = 0.
    loss_blurry_total = 0.
    loss_blurry_cont_total = 0.
    test_loss_clip_total = 0.
    
    loss_prior_total = 0.
    test_loss_prior_total = 0.

    blurry_pixcorr = 0.
    test_blurry_pixcorr = 0. 
    
    # you now have eeg_iters and image_iters with num_iterations_per_epoch batches each
    step = 0
    for train_i, (eeg, image,texts) in enumerate(train_dl):
        with torch.amp.autocast('cuda',dtype=data_type):
            optimizer.zero_grad()
            loss=0.

            #text = text_iters[train_i].detach()
            #image = rearrange(image,'b f d - > (b f) d').to(device)
            image = image.reshape(len(image)*args.fps*2, 3, 512, 512).to(device)
            image = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False).to(device)
            eeg = eeg.half().to(device)

            eeg = model.eeg(eeg)
            
            blurry_image_enc_ = model.backbone(eeg, time = args.batch_size*args.fps*2)

            if args.blurry_recon:
                image_enc_pred, transformer_feats = blurry_image_enc_

                image_enc = autoenc.encode(2*image-1).latent_dist.mode() * 0.18215
                
                
                loss_blurry = l1(image_enc_pred, image_enc)
                loss_blurry_total += loss_blurry.item()

                if epoch < int(args.mixup_pct * args.num_epochs):
                    print("epoch < mixup_pct * num_epochs")
                    # image_enc_shuf = image_enc[perm]
                    # betas_shape = [-1] + [1]*(len(image_enc.shape)-1)
                    # image_enc[select] = image_enc[select] * betas[select].reshape(*betas_shape) + \
                    #     image_enc_shuf[select] * (1 - betas[select]).reshape(*betas_shape)

                image_norm = (image - mean)/std
                image_aug = (blur_augs(image) - mean)/std
                _, cnx_embeds = cnx(image_norm)
                _, cnx_aug_embeds = cnx(image_aug)

                cont_loss = utils.soft_cont_loss(
                    nn.functional.normalize(transformer_feats.reshape(-1, transformer_feats.shape[-1]), dim=-1),
                    nn.functional.normalize(cnx_embeds.reshape(-1, cnx_embeds.shape[-1]), dim=-1),
                    nn.functional.normalize(cnx_aug_embeds.reshape(-1, cnx_embeds.shape[-1]), dim=-1),
                    temp=0.2)
                loss_blurry_cont_total += cont_loss.item()

                loss += (loss_blurry + 0.1*cont_loss) * args.blur_scale #/.18215

            if args.blurry_recon:
                with torch.no_grad():
                    # only doing pixcorr eval on a subset of the samples per batch because its costly & slow to compute autoenc.decode()
                    random_samps = np.random.choice(np.arange(len(image)), size=len(image), replace=False)
                    blurry_recon_images = (autoenc.decode(image_enc_pred[random_samps]/0.18215).sample/ 2 + 0.5).clamp(0,1)
                    pixcorr = utils.pixcorr(image[random_samps], blurry_recon_images)
                    blurry_pixcorr += pixcorr.item()

            utils.check_loss(loss)
            accelerator.backward(loss)
            optimizer.step()

            losses.append(loss.item())
            lrs.append(optimizer.param_groups[0]['lr'])

            if args.lr_scheduler_type is not None:
                lr_scheduler.step()
            step += 1
            print(f'Training epoch: {epoch}, sample: {step*args.batch_size}, lr: {optimizer.param_groups[0]["lr"]}, loss: {loss.item():.4f}, loss_mean: {np.mean(losses[-(train_i+1):]):.4f}')
    
    model.eval()
    
    if local_rank==0:
        sum_pixcorr=0
        test_count=0
        with torch.no_grad(), torch.amp.autocast('cuda',dtype=data_type):
            for test_i, (eeg, image,texts) in enumerate(test_dl):  
                # all test samples should be loaded per batch such that test_i should never exceed 0

                if test_image is None:
                    eeg = eeg.half()
                    
                    image = image.reshape(len(image)*args.fps*2, 3, 512, 512).cpu()
                    image = F.interpolate(image, size=(256, 256), mode='bilinear', align_corners=False).to(device)

                loss=0.
                            
                eeg = eeg.to(device)
                image = image.to(device)

                #clip_target = clip_img_embedder(image.float())

                test_fwd_percent_correct = 0.
                test_bwd_percent_correct = 0.
                text_fwd_percent_correct = 0.

                eeg = model.eeg(eeg)
                
                blurry_image_enc_ = model.backbone(eeg, time = args.batch_size*args.fps*2)
               
                # for some evals, only doing a subset of the samples per batch because of computational cost
                #random_samps = np.random.choice(np.arange(len(image)), size=len(image)//6, replace=False)     
                
                if args.blurry_recon:
                    image_enc_pred, _ = blurry_image_enc_
                    blurry_recon_images = (autoenc.decode(image_enc_pred/0.18215).sample / 2 + 0.5).clamp(0,1)
                    pixcorr = utils.pixcorr(image, blurry_recon_images)
                    test_blurry_pixcorr += pixcorr.item() 

                    #print('PixCorr:', pixcorr.item())

                #utils.check_loss(loss)              
                    test_losses.append(pixcorr.item())
                    sum_pixcorr +=pixcorr.item()
                
                test_count +=1

            # if utils.is_interactive(): clear_output(wait=True)
            print("-------------------------")
            write_scores_to_csv(csv_file_path,epoch,sum_pixcorr/test_count)
    # Save model checkpoint and reconstruct
    if test_blurry_pixcorr/30 > best_test:
        best_test = test_blurry_pixcorr/30
        print('new best test loss:',best_test)
        save_ckpt(save_path,epoch,optimizer,lr_scheduler,losses,test_losses,lrs)
    else:
        print('not best',test_blurry_pixcorr/30,'best test loss is',best_test)

    # wait for other GPUs to catch up if needed
    accelerator.wait_for_everyone()
    torch.cuda.empty_cache()
    gc.collect()
    

print("\n===Finished!===\n")
save_ckpt(save_path,epoch,optimizer,lr_scheduler,losses,test_losses,lrs,True)
#if ckpt_saving:
    #save_ckpt(f'{model_name}')
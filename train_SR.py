'''
Description: 
Author: Zhou Tianyi
Date: 2025-01-20 07:37:07
LastEditTime: 2025-02-26 11:14:47
LastEditors:  
'''

'''
可以改进
1. 使用数据集的label 作为loss 且评测指标
2. 只用cliploss

'''



import os
import sys
# SDXL unCLIP requires code from https://github.com/Stability-AI/generative-models/tree/main
sys.path.append('generative_models/')
import argparse
import numpy as np
from tqdm import tqdm
import webdataset as wds
import gc
from Semantic import *
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from accelerate import Accelerator
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder # bigG embedder from OpenCLIP
from sklearn.preprocessing import StandardScaler
from einops import rearrange
# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True
torch.cuda.empty_cache()
# custom functions #
import utils
import os
import datetime
from modeling_pretrain import Labram,TemporalConv
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
from models import *



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
    "--batch_size", type=int, default=32,
    help="Batch size can be increased by 10x if only training retreival submodule and not diffusion prior",
)
parser.add_argument(
    "--mixup_pct",type=float,default=.33,
    help="proportion of way through training when to switch from BiMixCo to SoftCLIP",
)
parser.add_argument(
    "--blurry_recon",action=argparse.BooleanOptionalAction,default=False,
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
    "--hidden_dim",type=int,default=400,
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
    "--test_id",type=int,default=5,
)
parser.add_argument(
    "--clip_seq_dim",type=int,default=64,
)
parser.add_argument(
    "--use_labels",type=bool
    ,default=False,
)
parser.add_argument(
    "--cudas",type=int
    ,default=0,
)
parser.add_argument(
    "--backbone_model",type=str
    ,default='neuroclips',
)
parser.add_argument(
    "--W",type=int
    ,default=224,
)
parser.add_argument(
    "--use_fake",action=argparse.BooleanOptionalAction,default=False
)
parser.add_argument(
    "--use_mse",action=argparse.BooleanOptionalAction,default=False
)


args = parser.parse_args()

# create global variables without the args prefix
for attribute_name in vars(args).keys():
    globals()[attribute_name] = getattr(args, attribute_name)


### Multi-GPU config ###
local_rank = os.getenv('RANK')
if local_rank is None: 
    local_rank = cudas
else:
    local_rank = int(local_rank)
print("LOCAL RANK ", local_rank)  

data_type = torch.float16 # change depending on your mixed_precision
num_devices = torch.cuda.device_count()
if num_devices==0: num_devices = 1

# First use "accelerate config" in terminal and setup using deepspeed stage 2 with CPU offloading!



print("PID of this process =",os.getpid())
# device = 'cuda:0'
# #device = 'cuda:0'
torch.cuda.set_device(cudas)
accelerator = Accelerator(split_batches=False, mixed_precision="fp16")
device = accelerator.device
print("device:",device)
world_size = accelerator.state.num_processes
distributed = not accelerator.state.distributed_type == 'NO'
#distributed =False
num_devices = torch.cuda.device_count()
if num_devices==0 or not distributed: num_devices = 1
num_workers = num_devices
print(accelerator.state)

print("distributed =",distributed, "num_devices =", num_devices, "local rank =", local_rank, "world size =", world_size, "data_type =", data_type)
print = accelerator.print # only print if local_rank=0


def get_time():
    current_time = datetime.datetime.now()

    # 加8小时并取mod 24
    new_hour = (current_time.hour + 8) % 24

    # 构建新的时间，替换小时部分
    new_time = current_time.replace(hour=new_hour)

    # 格式化输出
    formatted_time = new_time.strftime("%m-%d_%H-%M")
    return formatted_time

# seed all random functions
utils.seed_everything(seed)
model_name='SR'

if use_labels and use_prior:
    model_name+='_use_labels_after_prior_'
elif use_labels:
    model_name += '_use_labels_'

if backbone_model =='conformer':
    model_name = 'conformer_'+model_name 
elif backbone_model =='glfnet':
    model_name = 'glfnet_'+model_name
elif backbone_model =='labram':
    model_name = 'labram_'+model_name
elif backbone_model =='conv':
    model_name = 'conv_'+model_name

outdir=f"neuroclips/outputs/SR/checkpoints/{backbone_model}/"
if use_prior:
    outdir +='prior/'
######################## Parameters ##############################
subj_list = [subj]
seq_len = 62
clip_seq_dim = 256
clip_emb_dim = 1664
hidden_dim =400
current_time = get_time()
outdir +=f'{current_time}/'
os.makedirs(outdir,exist_ok=True)
model_name = model_name + f'{clip_seq_dim}_{clip_emb_dim}_{max_lr}'
if use_fake:
    model_name +='_after_fake_'
if use_mse:
    model_name +='_mse_'
################################# help function #####################
def topk_accuracy(output, target, topk=(1, )):       
    # output.shape (bs, num_classes), target.shape (bs, )
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred)) # bool array

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size).item())
        return res




def save_ckpt(save_path,epoch,optimizer,lr_scheduler,losses,test_losses,lrs,last_epoch=False):
    if last_epoch is True:
        ckpt_path = os.path.join(outdir,model_name+f'subj{subj}_{test_id}_last_epoch.pth')
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


def prepare_optimizier(model,num_iterations_per_epoch):
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=max_lr,weight_decay=0.02)

    if lr_scheduler_type == 'linear':
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            total_iters=int(np.floor(num_epochs*num_iterations_per_epoch)),
            last_epoch=-1
        )
    elif lr_scheduler_type == 'cycle':
        total_steps=int(np.floor(num_epochs*num_iterations_per_epoch))
        print("total_steps", total_steps)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=max_lr,
            total_steps=total_steps,
            final_div_factor=1000,
            last_epoch=-1, pct_start=2/num_epochs
        )
    
    return optimizer, lr_scheduler
    

def write_scores_to_csv(csv_file_path ,a,b,c,d,epoch,e):
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
            writer.writerow(['epoch', 'test_fwd_percent_correct', 'test_bwd_percent_correct', 'text_fwd_percent_correct','loss_all','train_loss'])
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch, a, b, c,d,e])
    
clip_img_embedder = FrozenOpenCLIPImageEmbedder(
    arch="ViT-bigG-14",
    version="/userhome/zhoutianyi/Zhoutianyi/Mutilmodel/huggingface/CLIP-ViT-bigG-14-laion2B-39B-b160k/open_clip_pytorch_model.bin",
    output_tokens=True,
    only_tokens=True,
    )
clip_img_embedder.to(device)
def train_test(model, train_iter, test_dl, num_epochs, device, save_path,optimizer,lr_scheduler,name):
    epoch = 0
    losses, test_losses, lrs = [], [], []
    best_test_loss = 1e9
    torch.cuda.empty_cache()
    train_dls = [train_iter]
    if use_prior:
        csv_dir = f"neuroclips/outputs/SR/logs/{backbone_model}/prior/{current_time}/"
    else:
        csv_dir = f"neuroclips/outputs/SR/logs/{backbone_model}/{current_time}/"
    os.makedirs(csv_dir,exist_ok=True)
    csv_file_path=f"{csv_dir}/evaluation_scores_{model_name}_{name}.csv"
    model, optimizer, *train_dls, lr_scheduler = accelerator.prepare(model, optimizer, *train_dls, lr_scheduler)
    # leaving out test_dl since we will only have local_rank 0 device do evals



    print(f"{model_name} starting with epoch {epoch} / {num_epochs}")
    progress_bar = tqdm(range(epoch,num_epochs), ncols=1200, disable=(local_rank!=0))
   
    mse = nn.MSELoss()
    l1 = nn.L1Loss()
    l2 = nn.CrossEntropyLoss()
    #soft_loss_temps = utils.cosine_anneal(0.004, 0.0075, num_epochs - int(mixup_pct * num_epochs))
    soft_loss_temps = utils.cosine_anneal(0.004, 0.0075, num_epochs)
    text_scale = 1
    text_scale_prior = 1.0
    if num_devices!=0 and distributed:
        model = model.module

    for epoch in progress_bar:
        model.train()
        fwd_percent_correct = 0.
        bwd_percent_correct = 0.
        loss_mse=0.
        loss_clip_total = 0.
        loss_blurry_total = 0.
        loss_blurry_cont_total = 0.
        test_loss_clip_total = 0.
        loss_all = 0.
        
        loss_prior_total = 0.
        test_loss_prior_total = 0.

        blurry_pixcorr = 0.
        test_blurry_pixcorr = 0. 
        top_1_acc =0
        top_k_acc =0
        train_loss =0
        step = 0
        model.train()
        for iter, (eeg, image, text,eeg_labels) in enumerate(train_dl): 
            with torch.amp.autocast('cuda',dtype=data_type):
                optimizer.zero_grad()
                loss=0.

                eeg = eeg.detach().to(device,dtype=data_type)
                #eeg = [eeg[f"subj0{s}_iter{train_i}"].detach() for s in subj_list]
                image = image.detach().to(dtype=data_type)
                text = text.detach().to(dtype=data_type)
                eeg_labels = eeg_labels.detach().to(dtype=torch.long)
                eeg_labels = eeg_labels.to(device)
                image = image.to(device)
                # print(eeg.dtype)
                # print(image.dtype)
                # print(text.dtype)
                # exit()

                clip_target = clip_img_embedder(image)
                
                assert not torch.any(torch.isnan(clip_target))
                
                
                
                if backbone_model =='neuroclips':
                    _, clip_eegs, blurry_image_enc_ = model.backbone(eeg)
                    clip_eegs = model.trans(clip_eegs)
                else:
                    clip_eegs = model.backbone(eeg)
                assert not torch.any(torch.isnan(clip_eegs))
                # clip_eegs =clip_eegs/clip_eegs.norm(dim=-1,keepdim=True)
                # clip_target =clip_target/ clip_target.norm(dim=-1, keepdim=True)
                # tar_mean = clip_target.mean()
                # tar_std = clip_target.std()
                # clip_target = (clip_target-tar_mean)/tar_std 
                
                if clip_scale>0:
                    clip_eegs_norm = nn.functional.normalize(clip_eegs.flatten(1), dim=-1) #[b,64*768]
                    clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)
                    # clip_eegs_norm =  nn.functional.normalize(model.eeg(clip_eegs),dim=-1) #[b,77*1280]
                    # clip_target_norm =  nn.functional.normalize(model.image(clip_target),dim=-1)
                    
                loss_mse = mse(clip_eegs.flatten(1),clip_target.flatten(1))
                loss +=loss_mse*10
                if use_prior:
                    loss_prior, prior_out = model.diffusion_prior(text_embed=clip_eegs, image_embed=clip_target)
                    loss_prior_total += loss_prior.item()
                    loss_prior *= prior_scale
                    loss += loss_prior
                    
                    # prior_norm = nn.functional.normalize(prior_out.flatten(1), dim=-1)
                    # loss +=mse(prior_norm,clip_target_norm)*prior_scale

                if use_labels:
                    if use_prior:
                        predicts = model.predictor(prior_out)
                    else:
                        predicts = model.predictor(clip_eegs)
                    loss_labels = l2(predicts,eeg_labels)
                    loss +=loss_labels*0.05

                
                # if clip_scale>0:
                #     if epoch < int(mixup_pct * num_epochs):                
                #         loss_clip = utils.mixco_nce(
                #             clip_eegs_norm,
                #             clip_target_norm,
                #             temp=.007,
                #             perm=None, betas=None, select=None)
                #     else:
                #         epoch_temp = soft_loss_temps[epoch-int(mixup_pct*num_epochs)]
                #         loss_clip = utils.soft_clip_loss(
                #             clip_eegs_norm,
                #             clip_target_norm,
                #             temp=epoch_temp)
                #     # epoch_temp = soft_loss_temps[epoch]
                #     # loss_clip = utils.soft_clip_loss(
                #     #     clip_eegs_norm,
                #     #     clip_target_norm,
                #     #     temp=epoch_temp)
                #     if not use_prior:
                #         loss_clip_total += loss_clip.item() 
                #         #loss_clip *= 0.1
                #         loss += loss_clip
                        

                if use_text:
                    if use_prior:
                        text = text.to(device)
                        pred_text_norm=nn.functional.normalize(model.eeg(prior_out), dim=-1)
                        target_text_norm = nn.functional.normalize(text.flatten(1), dim=-1)
                        loss_text = utils.mixco_nce(pred_text_norm, target_text_norm, perm=None, betas=None, select=None)
                        loss += loss_text*text_scale
                        #loss +=mse(pred_text_norm,target_text_norm)*10
                    else:
                        text = text.to(device)
                        pred_text_norm=nn.functional.normalize(model.eeg(clip_eegs), dim=-1)
                        target_text_norm = nn.functional.normalize(text.flatten(1), dim=-1)
                        # loss_text = utils.mixco_nce(pred_text_norm, target_text_norm, perm=None, betas=None, select=None)
                        # loss += loss_text*text_scale
                        # loss += model.loss_func(pred_text_norm,target_text_norm,model.logit_scale)*text_scale
                        loss_mse = mse(pred_text_norm,target_text_norm)
                        loss +=loss_mse*10


                if clip_scale>0:
                    # forward and backward top 1 accuracy        
                    labels = torch.arange(len(clip_eegs_norm)).to(clip_eegs_norm.device) 
                    fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_eegs_norm, clip_target_norm), labels, k=1).item()
                    bwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_target_norm, clip_eegs_norm), labels, k=1).item()
                
                



                utils.check_loss(loss)
                accelerator.backward(loss)
                optimizer.step()

                losses.append(loss.item())
                lrs.append(optimizer.param_groups[0]['lr'])
                train_loss += loss.item()
                if lr_scheduler_type is not None:
                    lr_scheduler.step()
                step += 1
                # print(f'Training epoch: {epoch}, sample: {step*batch_size}, lr: {optimizer.param_groups[0]["lr"]}, loss_clip: {loss_clip.item():.4f}, loss: {loss.item():.4f}, loss_mse: {loss_mse.item():.4f}')
                #print(f'Training epoch: {epoch}, sample: {step*batch_size}, lr: {optimizer.param_groups[0]["lr"]}, loss_clip: {loss_clip.item():.4f}, loss_text: {loss_text.item():.4f},loss: {loss.item():.4f}')
                print(f'Training epoch: {epoch}, sample: {step*batch_size}, lr: {optimizer.param_groups[0]["lr"]}, loss: {loss.item():.4f}')

        model.eval()
    
        if local_rank==cudas:
            test_fwd_percent_correct = 0.
            test_bwd_percent_correct = 0.
            text_fwd_percent_correct = 0.
            total_iters =0
            block_top_1 = []
            block_top_k = []
            with torch.no_grad(), torch.amp.autocast('cuda',dtype=data_type):
                for test_i, (eeg, image, text,eeg_labels) in enumerate(test_dl):  
                    # all test samples should be loaded per batch such that test_i should never exceed 0

                    ## Average same-image repeats ##
                    # if test_image is None:
                    #     eeg = eeg.half()                    
                    #     image = image[:,2,:,:,:].cpu()

                    loss=0.
                                
                    eeg = eeg.detach().to(device,dtype=data_type)
                    eeg_labels = eeg_labels.detach().to(device,dtype=torch.long)
                    #eeg = [eeg[f"subj0{s}_iter{train_i}"].detach() for s in subj_list]
                    image = image.detach().to(dtype=data_type)
                    text = text.detach().to(dtype=data_type)
                    image = image.to(device)
                    clip_target = clip_img_embedder(image)
                    
                    if backbone_model =='neuroclips':
                        _, clip_eegs, blurry_image_enc_ = model.backbone(eeg)
                        clip_eegs = model.trans(clip_eegs)
                    else:
                        clip_eegs = model.backbone(eeg)

                    clip_eegs = clip_eegs.to(device)

                    if clip_scale>0:
                        clip_eegs_norm = nn.functional.normalize(clip_eegs.flatten(1), dim=-1)
                        clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)
                        # clip_eegs_norm =  nn.functional.normalize(model.eeg(clip_eegs),dim=-1) #[b,77*1280]
                        # clip_target_norm =  nn.functional.normalize(model.image(clip_target),dim=-1)
                    loss_mse = mse(clip_eegs.flatten(1),clip_target.flatten(1))
                    loss +=loss_mse*10
                    if use_labels:
                        predicts = model.predictor(clip_eegs)
                        top_K_results = topk_accuracy(predicts, eeg_labels, topk=(1,5))
                        block_top_1.append(top_K_results[0])
                        block_top_k.append(top_K_results[1])
                        loss_labels = l2(predicts,eeg_labels)
                        loss +=loss_labels *0.25
                    # for some evals, only doing a subset of the samples per batch because of computational cost
                    random_samps = np.random.choice(np.arange(len(image)), size=len(image)//6, replace=False)
                
                    if use_prior:
                        loss_prior, prior_out = model.diffusion_prior(text_embed=clip_eegs[random_samps], image_embed=clip_target[random_samps])
                        test_loss_prior_total += loss_prior.item()
                        loss_prior *= prior_scale
                        loss += loss_prior
                        # prior_norm = nn.functional.normalize(prior_out.flatten(1), dim=-1)
                        # loss +=mse(prior_norm,clip_target_norm)*prior_scale
                        
                    if use_text:
                        if not use_prior:
                            text = text.to(device)
                            pred_text_norm=nn.functional.normalize(model.eeg(clip_eegs), dim=-1)
                            target_text_norm = nn.functional.normalize(text.flatten(1), dim=-1)
                            # loss_text = utils.mixco_nce(pred_text_norm, target_text_norm, perm=None, betas=None, select=None)
                            # loss += loss_text* text_scale
                            # labels = torch.arange(len(pred_text_norm)).to(pred_text_norm.device) 
                            loss_mse = mse(pred_text_norm,target_text_norm)
                            loss +=loss_mse*10
                            text_fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(pred_text_norm, target_text_norm), labels, k=5).item()
                        else:
                            text = text[random_samps].to(device)
                            pred_text_norm = nn.functional.normalize(model.eeg(prior_out).flatten(1), dim=-1)
                            target_text_norm = nn.functional.normalize(text.flatten(1), dim=-1)
                            loss_text = utils.mixco_nce(pred_text_norm, target_text_norm, perm=None, betas=None, select=None)
                            loss += loss_text* text_scale
                            labels = torch.arange(len(pred_text_norm)).to(pred_text_norm.device) 
                            text_fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(pred_text_norm, target_text_norm), labels, k=5).item()
                        


                    '''
                    if blurry_recon:
                        image_enc_pred, _ = blurry_image_enc_
                        blurry_recon_images = (autoenc.decode(image_enc_pred[random_samps]/0.18215).sample / 2 + 0.5).clamp(0,1)
                        pixcorr = utils.pixcorr(image[random_samps], blurry_recon_images)
                        test_blurry_pixcorr += pixcorr.item() 
                    '''

                    # if clip_scale>0:
                    #     loss_clip = utils.soft_clip_loss(
                    #         clip_eegs_norm,
                    #         clip_target_norm,
                    #         temp=.007)

                    #     test_loss_clip_total += loss_clip.item()
                    #     loss_clip = loss_clip 
                    #     loss += loss_clip
                    #     # loss_mse = mse(clip_eegs_norm,clip_target_norm)
                    #     # loss +=loss_mse*0.85


                    if clip_scale>0:
                        # forward and backward top 1 accuracy        
                        labels = torch.arange(len(clip_eegs_norm)).to(clip_eegs_norm.device) 
                        test_fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_eegs_norm, clip_target_norm), labels, k=1).item()
                        test_bwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_target_norm, clip_eegs_norm), labels, k=1).item()
                        #print('fwd:',test_fwd_percent_correct, 'bwd:',test_bwd_percent_correct, 'text fwd:', text_fwd_percent_correct)

                    utils.check_loss(loss) 
                    loss_all += loss.item()             
                    test_losses.append(loss.item())
                    total_iters +=1

                if use_labels:
                    top_1_acc = np.mean(np.array(block_top_1))
                    top_k_acc = np.mean(np.array(block_top_k))
                else:
                    top_1_acc=0
                    top_k_acc=0
                # if utils.is_interactive(): clear_output(wait=True)
                print("-------------------------")
                print(f'Test epoch: {epoch},loss: {loss.item():.4f}')
                write_scores_to_csv(csv_file_path,test_fwd_percent_correct/total_iters,test_bwd_percent_correct/total_iters,text_fwd_percent_correct/total_iters,loss_all/len(test_dl),epoch,train_loss/len(train_dl))
                

    # Save model checkpoint and reconstruct
        if loss_all/len(test_dl) < best_test_loss:
            best_test_loss = loss_all/len(test_dl)
            print('new best test loss:',best_test_loss)
            if not use_prior:
                save_ckpt(save_path,epoch,optimizer,lr_scheduler,losses,test_losses,lrs)
        else:
            print('not best:',loss_all/len(test_dl), 'best test loss is',best_test_loss)
        
        #if epoch % 30 == 0:
            #save_ckpt(f'{model_name}')

        # wait for other GPUs to catch up if needed
        accelerator.wait_for_everyone()
        torch.cuda.empty_cache()
        gc.collect()

        
    print("\n===Finished!===\n")
    #if ckpt_saving and use_prior:
    save_ckpt(save_path,epoch,optimizer,lr_scheduler,losses,test_losses,lrs,last_epoch=True)



def prepare_model(config):
    
    class Neuroclips(nn.Module):
        def __init__(self):
            super(Neuroclips, self).__init__()
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            self.loss_func = utils.ClipLoss() 
        def forward(self, x):
            return x

    class Proj_eeg(nn.Module):
        def __init__(self, embedding_dim=clip_emb_dim, proj_dim=1280,in_chans=clip_seq_dim,out_chans=77, drop_proj=0.5):
            super().__init__()
            self.fc = nn.Sequential(
                Rearrange('b c t -> b t c'),
                nn.Linear(in_chans,out_chans),
                nn.Dropout(drop_proj),
                nn.GELU(),
                Rearrange('b t c -> b c t'),
                nn.Linear(embedding_dim, proj_dim),
                nn.Dropout(drop_proj),
                nn.GELU(),
                nn.Flatten())
            
        def forward(self,x):
            return self.fc(x)
    model = Neuroclips()
    print('1')
    if config.backbone_model =='glfnet':
        model.backbone = glfnet(emb_dim=clip_emb_dim,seq_len=clip_seq_dim,depth = 2)
        
    elif backbone_model =='labram':
        model.backbone = Labram()
    elif backbone_model =='conv':
        model.backbone = TemporalConv(in_dim=5)
    print('2')
    
    model.eeg = Proj_eeg()
    print('3')
    if use_labels:
        model.predictor=mlpnet(out_dim=40, input_dim=clip_emb_dim*clip_seq_dim)
        print('4')
    utils.count_params(model.backbone)
    utils.count_params(model)

    if use_prior:
        ck = torch.load('/userhome/zhoutianyi/Zhoutianyi/Mutilmodel/Models/EEG/EEG2Video/EEG2Video2/neuroclips/outputs/SR/checkpoints/conv/02-24_23-18/conv_SR256_1664subj1_0.pth',map_location='cpu')
        model.load_state_dict(ck['model_state_dict'])
        del ck
    if config.use_prior:
        # setup diffusion prior network
        out_dim = clip_emb_dim
        depth = 6
        dim_head = 52
        heads = clip_emb_dim//52 # heads * dim_head = clip_emb_dim
        timesteps = 100

        prior_network = PriorNetwork(
                dim=out_dim,
                depth=depth,
                dim_head=dim_head,
                heads=heads,
                causal=False,
                num_tokens = clip_seq_dim,
                learned_query_mode="pos_emb",
            )

        model.diffusion_prior = BrainDiffusionPrior(
            net=prior_network,
            
            image_embed_dim=out_dim,
            condition_on_text_encodings=False,
            timesteps=timesteps,
            cond_drop_prob=0.2,
            image_embed_scale=None,
        )
        
        utils.count_params(model.diffusion_prior)
        utils.count_params(model)

    # test on subject 1 with fake data
    if config.use_prior:
        for param in model.parameters():
            param.requires_grad_(False)
        for param in model.diffusion_prior.parameters():
            param.requires_grad_(True)
    else:
        for param in model.parameters():
            param.requires_grad_(True)
    if use_text:
        ck = torch.load("neuroclips/outputs/Proj/02-23_20-28/best_0.0001_0.1017.bin",map_location='cpu')
        model.eeg.load_state_dict(ck)
        for param in model.eeg.parameters():
            param.requires_grad_(False)
        del ck
    utils.count_params(model)
    return model
GT_label = np.array([[23, 22, 9, 6, 18,       14, 5, 36, 25, 19,      28, 35, 3, 16, 24,      40, 15, 27, 38, 33, 
             34, 4, 39, 17, 1,       26, 20, 29, 13, 32,     37, 2, 11, 12, 30,      31, 8, 21, 7, 10, ],
            [27, 33, 22, 28, 31,     12, 38, 4, 18, 17,      35, 39, 40, 5, 24,      32, 15, 13, 2, 16,
 	         34, 25, 19, 30, 23,     3, 8, 29, 7, 20,        11, 14, 37, 6, 21,      1, 10, 36, 26, 9, ],
            [15, 36, 31, 1, 34,      3, 37, 12, 4, 5,        21, 24, 14, 16, 39,     20, 28, 29, 18, 32, 
             2, 27, 8, 19, 13,       10, 30, 40, 17, 26,     11, 9, 33, 25, 35,      7, 38, 22, 23, 6,],
            [16, 28, 23, 1, 39,      10, 35, 14, 19, 27,     37, 31, 5, 18, 11,      25, 29, 13, 20, 24, 
            7, 34, 26, 4, 40 ,       12, 8, 22, 21, 30,      17, 2, 38, 9,  3 ,      36, 33, 6, 32, 15,],
            [18, 29, 7, 35, 22  ,    19, 12, 36, 8, 15,      28, 1, 34, 23, 20 ,     13, 37, 9, 16, 30  ,  
             2, 33, 27, 21, 14 ,     38, 10, 17, 31, 3,      24, 39, 11, 32, 4,      25, 40, 5, 26, 6 ,],
            [29, 16, 1, 22, 34,      39, 24, 10, 8, 35,      27, 31, 23, 17, 2,      15, 25, 40, 3, 36, 
             26, 6, 14, 37, 9,       12, 19, 30, 5, 28,      32, 4, 13, 18, 21,      20, 7, 11, 33, 38],
            [38, 34, 40, 10, 28,     7, 1, 37, 22, 9,        16, 5, 12, 36, 20,      30, 6, 15, 35, 2,      
             31, 26, 18, 24, 8,      3, 23, 19, 14, 13,      21, 4, 25, 11, 32,      17, 39, 29, 33, 27]
            ])
GT_label = GT_label - 1

class NeuroclipDataset(torch.utils.data.Dataset):
    def __init__(self,eeg,image,text,labels):
        self.eeg = eeg
        self.image=image
        self.text = text 
        self.labels = labels
    def __len__(self):
        return len(self.eeg)

    def __getitem__(self, index):
        return self.eeg[index],self.image[index],self.text[index],self.labels[index]

All_label = np.empty((0, 200))
for block_id in range(7):
    All_label = np.concatenate((All_label, GT_label[block_id].repeat(5).reshape(1, 200)))

for sub in subj_list:
    #load_npy = np.load(f'data/SEED-DV/Segmented_Rawf_200Hz_2s/sub1.npy')
    load_npy = np.load(f'data/SEED-DV/DE_1per2s/sub1.npy')
    images = torch.load('/userhome/zhoutianyi/Zhoutianyi/dataset/SEED-DV/video_torch_224_6.pt',map_location='cpu') 
    images = images[:,:,:,3]
    images = rearrange(images,'a b c d e f -> a (b c) d e f')
    
    #
    images = transforms.CenterCrop(224)(images)
    
    texts = torch.load('/userhome/zhoutianyi/Zhoutianyi/dataset/SEED-DV/text_emb_77*1280.pt',map_location='cpu')
    
    # eeg shape [7,40,5,62,200] images shape [7,200,6,768],text shape [7,200,768]
    All_train = rearrange(load_npy,'a b c e f-> a (b c )  e f')
    del load_npy
    print(All_train.shape)
    
    Top_1 = []
    Top_K = []

    
    train_eeg = np.empty((0,62,5))
    train_images = torch.empty((0,3,224,224))
    train_texts = torch.empty((0,77,1280))
    train_label = np.empty((0))
    test_eeg =All_train[test_id]
    test_images = images[test_id]
    test_texts = texts[test_id]
    test_labels =All_label[test_id]
    print(test_labels.shape)
    for i in range(7):
        if i == test_id:
            continue 
        train_eeg = np.concatenate((train_eeg,All_train[i].reshape(200,62,5)))
        train_images = torch.cat((train_images,images[i].reshape(200,3,224,224)))
        train_texts = torch.cat((train_texts,texts[i].reshape(200,77,1280)))
        train_label = np.concatenate((train_label, All_label[i]))
    print(train_label.shape)
    
    train_eeg = train_eeg.reshape(train_eeg.shape[0],62*5)
    test_eeg = test_eeg.reshape(test_eeg.shape[0],62*5)
    normalize = StandardScaler()
    normalize.fit(train_eeg)
    train_eeg = normalize.transform(train_eeg)  
    normalize = StandardScaler()
    normalize.fit(test_eeg)
    test_eeg = normalize.transform(test_eeg)   
    C=62
    T=5
    train_eeg = train_eeg.reshape(train_eeg.shape[0], C, T)
    test_eeg = test_eeg.reshape(test_eeg.shape[0], C, T)
    
    train_dataset = NeuroclipDataset(train_eeg, train_images, train_texts,train_label)
    test_dataset = NeuroclipDataset(test_eeg, test_images, test_texts,test_labels)
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False,pin_memory=True)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, drop_last=False,pin_memory=True)
    model = prepare_model(args)
    if use_fake:
        ck = torch.load('/userhome/zhoutianyi/Zhoutianyi/Mutilmodel/Models/EEG/EEG2Video/EEG2Video2/neuroclips/outputs/SR/checkpoints/conv/fake/02-24_23-41/conv_SR256_1664subj1_0.pth',map_location='cpu')
        model.load_state_dict(ck['model_state_dict'], strict=False)
        del ck
    print("\nDone with model preparations!")
    
    num_params = utils.count_params(model)
    num_samples_per_epoch = len(train_eeg) // num_devices 
    num_iterations_per_epoch = num_samples_per_epoch // (batch_size)
    optimizer,lr_scheduler = prepare_optimizier(model, num_iterations_per_epoch)
    save_path = os.path.join(outdir,model_name+f'subj{subj}_{test_id}.pth')
    name = f'subj{subj}_{test_id}'
    del train_eeg,test_eeg,train_images,train_texts,test_images,test_texts,train_label,test_labels
    train_test(model, train_dl, test_dl, num_epochs, device, save_path,optimizer,lr_scheduler,name)
    
    del model, train_dl, test_dl,optimizer,lr_scheduler
    torch.cuda.empty_cache()







    



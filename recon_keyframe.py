import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from accelerate import Accelerator
from models import *
from Semantic import *
# SDXL unCLIP requires code from https://github.com/Stability-AI/generative-models/tree/main
sys.path.append('generative_models/')
from generative_models.sgm.models.diffusion import DiffusionEngine
from generative_models.sgm.util import append_dims
from omegaconf import OmegaConf
from sklearn.preprocessing import StandardScaler
from einops import rearrange
# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder
# custom functions #
import utils
import os
import datetime
from modeling_pretrain import Labram,TemporalConv
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"



parser = argparse.ArgumentParser(description="Model Training Configuration")
parser.add_argument(
    "--model_name", type=str, default="testing",
    help="will load ckpt for model found in ../train_logs/model_name",
)
parser.add_argument(
    "--data_path", type=str, default=os.getcwd(),
    help="Path to where NSD data is stored / where to download it to",
)
parser.add_argument(
    "--subj",type=int, default=1, choices=[1,2,3],
    help="Validate on which subject?",
)
parser.add_argument(
    "--use_prior",action=argparse.BooleanOptionalAction,default=True,
    help="whether to train diffusion prior (True) or just rely on retrieval part of the pipeline (False)",
)
parser.add_argument(
    "--clip_scale",type=float,default=1.,
    help="multiply contrastive loss by this number",
)
parser.add_argument(
    "--blurry_recon",action=argparse.BooleanOptionalAction,default=False,
)
parser.add_argument(
    "--n_blocks",type=int,default=4,
)
parser.add_argument(
    "--prior_scale",type=float,default=30,
    help="multiply diffusion prior loss by this",
)
parser.add_argument(
    "--hidden_dim",type=int,default=400,
)
parser.add_argument(
    "--seed",type=int,default=42,
)
parser.add_argument(
    "--use_labels",type=bool
    ,default=False,
)
parser.add_argument(
    "--cudas",type=int
    ,default=3,
)
parser.add_argument(
    "--backbone_model",type=str
    ,default='glfnet',
)
parser.add_argument(
    "--test_id",type=int,default=5,
)
parser.add_argument(
    "--use_text",action=argparse.BooleanOptionalAction,default=False,
)
args = parser.parse_args()

# create global variables without the args prefix
for attribute_name in vars(args).keys():
    globals()[attribute_name] = getattr(args, attribute_name)
    

### Multi-GPU config ###
local_rank = os.getenv('RANK')
if local_rank is None: 
    local_rank = 0
else:
    local_rank = int(local_rank)
print("LOCAL RANK ", local_rank)  
torch.cuda.set_device(cudas)
accelerator = Accelerator(split_batches=False, mixed_precision="fp16")
device = accelerator.device


# seed all random functions
utils.seed_everything(seed)

# make output directory
os.makedirs("neuroclips/evals",exist_ok=True)
def get_time():
    current_time = datetime.datetime.now()

    # 加8小时并取mod 24
    new_hour = (current_time.hour + 8) % 24

    # 构建新的时间，替换小时部分
    new_time = current_time.replace(hour=new_hour)

    # 格式化输出
    formatted_time = new_time.strftime("%m-%d_%H-%M")
    return formatted_time
current_time=get_time()
eegs = {}
model_name =f'subj0{subj}_SR'

if use_labels:
    model_name+='_use_labels_'

if backbone_model =='conformer':
    model_name = 'conformer_'+model_name 
elif backbone_model =='glfnet':
    model_name = 'glfnet_'+model_name
elif backbone_model =='neuroclips':
    model_name = 'neuroclips_'+model_name

    
os.makedirs(f"neuroclips/frames_generated/{backbone_model}/{current_time}",exist_ok=True)
# Load hdf5 data for betas
outdirs = f"neuroclips/frames_generated/{backbone_model}/{current_time}"

class NeuroclipDataset(torch.utils.data.Dataset):
    def __init__(self,eeg,blurry):
        self.eeg = eeg
        self.blurry = blurry
    def __len__(self):
        return len(self.eeg)

    def __getitem__(self, index):
        return self.eeg[index],self.blurry[index]

subj_list = [subj]
seq_len = 62
clip_seq_dim = 256
clip_emb_dim = 1664
hidden_dim =5

# load_npy = np.load(f'data/SEED-DV/Segmented_Rawf_200Hz_2s/sub{subj}.npy')
load_npy = np.load(f'data/SEED-DV/DE_1per2s/sub1.npy')
All_train = rearrange(load_npy,'a b c e f-> a (b c )  e f')
del load_npy
print(All_train.shape)

test_eeg =All_train[test_id]
test_eeg = test_eeg.reshape(test_eeg.shape[0],62*5)
normalize = StandardScaler()
normalize.fit(test_eeg)
test_eeg = normalize.transform(test_eeg)  
C=62
T=5
test_eeg = test_eeg.reshape(test_eeg.shape[0], C, T)
# images = torch.load('/userhome/zhoutianyi/Zhoutianyi/dataset/SEED-DV/image_emb_1280_vit.pt',map_location='cpu')

blurry =  torch.load('/userhome/zhoutianyi/Zhoutianyi/Mutilmodel/Models/EEG/EEG2Video/EEG2Video2/neuroclips/outputs/blurry/video_subj01_lPR_0_mlp_PR_embeddings.pt',map_location='cpu')


test_dataset = NeuroclipDataset(test_eeg,blurry)
test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0, drop_last=False,pin_memory=True)
del test_eeg,test_dataset
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
    
def prepare_model(config):
    
    class Neuroclips(nn.Module):
        def __init__(self):
            super(Neuroclips, self).__init__()
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            self.loss_func = utils.ClipLoss() 
        def forward(self, x):
            return x
    class ResidualAdd(nn.Module):
        def __init__(self, fn):
            super().__init__()
            self.fn = fn

        def forward(self, x, **kwargs):
            res = x
            x = self.fn(x, **kwargs)
            x += res
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
    elif config.backbone_model =='conv':
        model.backbone = TemporalConv(in_dim=5)
    print('2')
    
    model.eeg = Proj_eeg()
    print('3')
    if use_labels:
        model.predictor=mlpnet(out_dim=40, input_dim=clip_emb_dim*clip_seq_dim)
        print('4')
    utils.count_params(model.backbone)
    utils.count_params(model)

    # if use_prior:
    #     ck = torch.load('/userhome/zhoutianyi/Zhoutianyi/Mutilmodel/Models/EEG/EEG2Video/EEG2Video2/neuroclips/outputs/SR/checkpoints/glfnet/02-23_22-15/glfnet_SR256_1664subj1_0_last_epoch.pth',map_location='cpu')
    #     model.load_state_dict(ck['model_state_dict'])
    #     del ck
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
        checkpoint = torch.load('/userhome/zhoutianyi/Zhoutianyi/Mutilmodel/Models/EEG/EEG2Video/EEG2Video2/neuroclips/outputs/SR/checkpoints/conv/prior/02-25_11-56/conv_SR256_1664_5e-05subj1_5_last_epoch.pth', map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        del checkpoint

    # test on subject 1 with fake data
    for param in model.parameters():
        param.requires_grad_(False)
    utils.count_params(model)
    return model

model = prepare_model(args)
model.eval()
print("\n---resuming from last.pth ckpt---\n")





# setup text caption networks
from transformers import AutoProcessor
from modeling_git import GitForCausalLMClipEmb
processor = AutoProcessor.from_pretrained("/userhome/zhoutianyi/Zhoutianyi/Mutilmodel/huggingface/git-large-coco")
clip_text_model = GitForCausalLMClipEmb.from_pretrained("/userhome/zhoutianyi/Zhoutianyi/Mutilmodel/huggingface/git-large-coco")
clip_text_model.to(device) # if you get OOM running this script, you can switch this to cpu and lower minibatch_size to 4
clip_text_model.eval().requires_grad_(False)
clip_text_seq_dim = 257
clip_text_emb_dim = 1024
class Adapter(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
        return x 
    
class EEGProj(nn.Sequential):
    def __init__(self,channel=256,in_dim = 1664,proj_dim=1024):
        super().__init__(
            Rearrange('b c l -> b l c'),
            nn.Linear(channel, 257),
            nn.LayerNorm(257),
            Rearrange('b l c -> b c l'),
            nn.Linear(in_dim,proj_dim),
            nn.LayerNorm(proj_dim)
        )
class ImageProj(nn.Module):
    def __init__(self,in_dim=1280,proj=1024):
        super().__init__()
        
        self.lin1 = nn.Linear(1,257)
        self.lin2 = nn.Linear(in_dim,proj)
        self.norm1 = nn.LayerNorm(257)
        self.norm2 = nn.LayerNorm(proj)
        self.rearrange1 = Rearrange('b c l -> b l c')
        self.rearrange2 = Rearrange('b l c -> b c l')
    
    def forward(self,x):
        x = x.unsqueeze(1)
        x = self.rearrange1(x)
        x = self.norm1(self.lin1(x))
        x = self.rearrange2(x)
        x = self.norm2(self.lin2(x))
        return x

clip_convert = Adapter()
clip_convert.eeg = EEGProj()
clip_convert.image = ImageProj()
state_dict = torch.load(f"neuroclips/outputs/image_adapter/PixelProjector_best.bin", map_location='cpu')
clip_convert.load_state_dict(state_dict, strict=True)
clip_convert.to(device) # if you get OOM running this script, you can switch this to cpu and lower minibatch_size to 4
del state_dict

# prep unCLIP
config = OmegaConf.load("neuroclips/models/generative_models/configs/unclip6.yaml")
config = OmegaConf.to_container(config, resolve=True)
unclip_params = config["model"]["params"]
network_config = unclip_params["network_config"]
denoiser_config = unclip_params["denoiser_config"]
first_stage_config = unclip_params["first_stage_config"]
conditioner_config = unclip_params["conditioner_config"]
sampler_config = unclip_params["sampler_config"]
scale_factor = unclip_params["scale_factor"]
disable_first_stage_autocast = unclip_params["disable_first_stage_autocast"]
offset_noise_level = unclip_params["loss_fn_config"]["params"]["offset_noise_level"]

first_stage_config['target'] = 'sgm.models.autoencoder.AutoencoderKL'
sampler_config['params']['num_steps'] = 50

diffusion_engine = DiffusionEngine(network_config=network_config,
                       denoiser_config=denoiser_config,
                       first_stage_config=first_stage_config,
                       conditioner_config=conditioner_config,
                       sampler_config=sampler_config,
                       scale_factor=scale_factor,
                       disable_first_stage_autocast=disable_first_stage_autocast)
# set to inference
diffusion_engine.eval().requires_grad_(False)
diffusion_engine.to(device)
ckpt_path = f'/userhome/zhoutianyi/Zhoutianyi/Mutilmodel/huggingface/pscotti-neuroclips/unclip6_epoch0_step110000.ckpt'
ckpt = torch.load(ckpt_path, map_location='cpu')
diffusion_engine.load_state_dict(ckpt['state_dict'])
del ckpt

batch={"jpg": torch.randn(1,3,1,1).to(device), # jpg doesnt get used, it's just a placeholder
      "original_size_as_tuple": torch.ones(1, 2).to(device) * 768,
      "crop_coords_top_left": torch.zeros(1, 2).to(device)}
out = diffusion_engine.conditioner(batch)
vector_suffix = out["vector"].to(device)
print("vector_suffix", vector_suffix.shape)

# get all reconstructions
model.to(device)
model.eval().requires_grad_(False)

# all_images = None
all_blurryrecons = None
all_recons = None
all_predcaptions = []
all_clipeegs = None
all_texteegs = None

minibatch_size = 10
num_samples_per_image = 1
assert num_samples_per_image == 1
plotting = False
index = 0
clip_img_embedder = FrozenOpenCLIPImageEmbedder(
    arch="ViT-bigG-14",
    version="/userhome/zhoutianyi/Zhoutianyi/Mutilmodel/huggingface/CLIP-ViT-bigG-14-laion2B-39B-b160k/open_clip_pytorch_model.bin",
    output_tokens=True,
    only_tokens=True,
    )
clip_img_embedder.to(device)
with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
    for (eeg,blurry) in test_dl:
        
        
        eeg = eeg.detach().to(device,dtype=torch.float16)
        
        
        if backbone_model =='neuroclips':
            backbone, clip_eegs, blurry_image_enc_ = model.backbone(eeg)
            clip_eegs = model.trans(clip_eegs)
        else:
            clip_eegs = model.backbone(eeg)

        clip_eegs = clip_eegs.to(device)
        
                
        # Save retrieval submodule outputs
        if all_clipeegs is None:
            all_clipeegs = clip_eegs.cpu()
        else:
            all_clipeegs = torch.vstack((all_clipeegs, clip_eegs.cpu()))
        
        #Feed eegs through OpenCLIP-bigG diffusion prior
        prior_out = model.diffusion_prior.p_sample_loop(clip_eegs.shape, 
                        text_cond = dict(text_embed = clip_eegs), 
                        cond_scale = 1., timesteps = 20)
        # loss_prior, prior_out = model.diffusion_prior(text_embed=clip_eegs, image_embed=clip_target)

        prior_out = prior_out.to(device)
        
        pred_caption_emb = clip_convert.eeg(prior_out)
        generated_ids = clip_text_model.generate(pixel_values=pred_caption_emb, max_length=20)
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)
        all_predcaptions = np.hstack((all_predcaptions, generated_caption))
        
        
        # Feed diffusion prior outputs through unCLIP
        for i in range(len(eeg)):
            index += 1
            print(index)
            samples = utils.unclip_recon(prior_out[[i]],blurry[i],
                                diffusion_engine,
                                vector_suffix,
                                num_samples=num_samples_per_image,
                                device = device)
            if all_recons is None:
                all_recons = samples.cpu()
            else:
                all_recons = torch.vstack((all_recons, samples.cpu()))

            #transforms.ToPILImage()(samples[0]).save(f'/home/students/gzx_4090_1/Video/frames_generated/video_subj01_skiplora_text_40sess_10bs/images/{all_recons.shape[0]-1}.png')
            if plotting:
                for s in range(num_samples_per_image):
                    plt.figure(figsize=(2,2))
                    plt.imshow(transforms.ToPILImage()(samples[s]))
                    plt.axis('off')
                    plt.show()

            if blurry_recon:
                blurred_image = (autoenc.decode(blurry_image_enc[0]/0.18215).sample/ 2 + 0.5).clamp(0,1)
                
                for i in range(len(eeg)):
                    im = torch.Tensor(blurred_image[i])
                    if all_blurryrecons is None:
                        all_blurryrecons = im[None].cpu()
                    else:
                        all_blurryrecons = torch.vstack((all_blurryrecons, im[None].cpu()))
                    if plotting:
                        plt.figure(figsize=(2,2))
                        plt.imshow(transforms.ToPILImage()(im))
                        plt.axis('off')
                        plt.show()

            
                 # dont actually want to run the whole thing with plotting=True
        

# resize outputs before saving
imsize = 224
all_recons = transforms.Resize((imsize,imsize))(all_recons).float()
if blurry_recon: 
    all_blurryrecons = transforms.Resize((imsize,imsize))(all_blurryrecons).float()
        
# saving
print(all_recons.shape)
# torch.save(all_images,"evals/all_images.pt")
if blurry_recon:
    torch.save(all_blurryrecons,f"{outdirs}/all_blurryrecons.pt")
torch.save(all_recons,f"{outdirs}/all_recons.pt")
torch.save(all_predcaptions,f"{outdirs}/all_predcaptions.pt")
torch.save(all_clipeegs,f"{outdirs}/all_clipeegs.pt")
print(f"saved {model_name} outputs!")


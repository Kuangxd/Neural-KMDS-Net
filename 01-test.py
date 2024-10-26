import dataloaders_test
import torch
import numpy as np
from tqdm import tqdm
import argparse
import os
from skimage.restoration import  denoise_nl_means,estimate_sigma
import scipy.io as scio
from model_loader import init_model,load_model
from ops.utils_blocks import block_module
from ops.utils import show_mem, generate_key, save_checkpoint, str2bool, step_lr, get_lr,MSIQA
import pdb
from model.KMDSNet import KMDSNetParams
from model.KMDSNet import KMDSNet4D

parser = argparse.ArgumentParser()
parser.add_argument("--gpus", '--list',action='append', type=int, help='GPU')
parser.add_argument("--model_name", type=str, dest="model_name", help="The name of the model to be saved.", default=None)
parser.add_argument("--test_path", type=str, help="Path to the dir containing the testing datasets.", default="../00-dataset/02-zubal-4D/01-4D-test/noise")
parser.add_argument("--tqdm", type=str2bool, default=False)
parser.add_argument("--gt_path", type=str, help="Path to the dir containing the ground truth datasets.", default="gt/")
parser.add_argument("--test_batch", type=int, default=1, help='batch size of testing')
parser.add_argument("--inner_num", type=int, default=32)
parser.add_argument("--iter_num", type=int, default=10)
parser.add_argument("--use_kernel", type=int, default=0)
parser.add_argument("--kernel_depth", type=int, default=3)
parser.add_argument("--frame_num", type=int, default=28)
parser.add_argument("--which_model", type=str, default='KMDSNet4D')
parser.add_argument("--use_ConvTranspose2d", type=int, default=0)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
capability = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else os.cpu_count()

gpus=args.gpus
test_path = [args.test_path]

loaders = dataloaders_test.get_dataloaders(test_path, drop_last=True, verbose=True, frame_num=args.frame_num)

params = KMDSNetParams(inner_num=args.inner_num, iter_num=args.iter_num, use_kernel=args.use_kernel, 
                     kernel_depth=args.kernel_depth, frame_num=args.frame_num, use_ConvTranspose2d=args.use_ConvTranspose2d)

model = KMDSNet4D(params).to(device=device)

pytorch_total_params = sum(p.numel() for p in model.parameters())
print('Nb tensors: ',len(list(model.named_parameters())), "; Trainable Params: ", pytorch_total_params)

load_model(model_name=args.model_name, model=model)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
if device.type == 'cuda':
    torch.cuda.set_device('cuda:{}'.format(gpus[0]))
if device.type == 'cuda':
    model = torch.nn.DataParallel(model.to(device=device), device_ids=gpus)
model.eval()  # Set model to evaluate mode

loader = loaders['test']

for batch,fname in tqdm(loader, disable=not args.tqdm):
    batch = batch.to(device=device)
    fname=fname[0]
    with torch.set_grad_enabled(False):
        output = model(batch)
        output[output<0] = 0
        output = output.squeeze(0).detach().cpu().numpy()

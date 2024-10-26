import dataloaders_train
import torch
import numpy as np
from tqdm import tqdm
import argparse
import os
import time
from ops.utils import show_mem, generate_key, save_checkpoint, str2bool, step_lr, get_lr
import pdb
from model.KMDSNet import KMDSNetParams
from model.KMDSNet import KMDSNet4D

parser = argparse.ArgumentParser()
parser.add_argument("--gpus", '--list', action='append', type=int, help='GPU')
parser.add_argument("--lr", type=float, dest="lr", help="ADAM Learning rate", default=1e-4)
parser.add_argument("--lr_step", type=int, dest="lr_step", help="ADAM Learning rate step for decay", default=80)
parser.add_argument("--lr_decay", type=float, dest="lr_decay", help="ADAM Learning rate decay (on step)", default=0.35)
parser.add_argument("--backtrack_decay", type=float, help='decay when backtracking',default=0.8)
parser.add_argument("--eps", type=float, dest="eps", help="ADAM epsilon parameter", default=1e-3)
parser.add_argument("--validation_every", type=int, default=4000, help='validation frequency on training set (if using backtracking)')
parser.add_argument("--backtrack", type=str2bool, default=1, help='use backtrack to prevent model divergence')
parser.add_argument("--num_epochs", type=int, dest="num_epochs", help="Total number of epochs to train", default=300)
parser.add_argument("--train_batch", type=int, default=1, help='batch size during training')
parser.add_argument("--test_batch", type=int, default=1, help='batch size during eval')
parser.add_argument("--out_dir", type=str, dest="out_dir", help="Results' dir path", default='checkpoints')
parser.add_argument("--model_name", type=str, dest="model_name", help="The name of the model to be saved.", default=None)
parser.add_argument("--train_path", type=str, help="Path to the dir containing the training datasets.", default="")
parser.add_argument("--tqdm", type=str2bool, default=True)
parser.add_argument("--resume", type=str2bool, dest="resume", help='Resume training of the model',default=True)
parser.add_argument("--inner_num", type=int, default=24)
parser.add_argument("--iter_num", type=int, default=20)
parser.add_argument("--use_kernel", type=int, default=1)
parser.add_argument("--kernel_depth", type=int, default=2)
parser.add_argument("--which_model", type=str, default='KMDSNet4D')
parser.add_argument("--use_ConvTranspose2d", type=int, default=0)
parser.add_argument("--frame_num", type=int, default=24)


args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
capability = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else os.cpu_count()
gpus=args.gpus
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
if device.type=='cuda':
    torch.cuda.set_device('cuda:{}'.format(gpus[0]))

train_path = [f'{args.train_path}']
val_path = train_path

loaders = dataloaders_train.get_dataloaders(train_path, val_path, batch_size=args.train_batch, concat=0)

params = KMDSNetParams(inner_num=args.inner_num, iter_num=args.iter_num, use_kernel=args.use_kernel, kernel_depth=args.kernel_depth, 
                       frame_num=args.frame_num, use_ConvTranspose2d=args.use_ConvTranspose2d)

model = KMDSNet4D(params).to(device=device)


if device.type=='cuda':
    model = torch.nn.DataParallel(model.to(device=device), device_ids=gpus)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=args.eps)
if args.backtrack:
    reload_counter = 0
pytorch_total_params = sum(p.numel() for p in model.parameters())
print(f'Arguments: {vars(args)}')
print('Nb tensors: ',len(list(model.named_parameters())), "; Trainable Params: ", pytorch_total_params, "; device: ", device,
      "; name : ", device_name)
psnr = {x: np.zeros(args.num_epochs) for x in ['train', 'test', 'val']}

model_name = args.model_name if args.model_name is not None else generate_key()

out_dir = os.path.join(args.out_dir, model_name)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
log_file_name = os.path.join(out_dir, 'log.txt')

ckpt_path = os.path.join(out_dir+'/ckpt')
config_dict = vars(args)
if args.resume:
    if os.path.isfile(ckpt_path):
        try:
            print('\n existing ckpt detected')
            checkpoint = torch.load(ckpt_path)
            start_epoch = checkpoint['epoch']
            psnr_validation = checkpoint['psnr_validation']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> loaded checkpoint '{ckpt_path}' (epoch {start_epoch})")
        except Exception as e:
            print(e)
            print(f'ckpt loading failed @{ckpt_path}, exit ...')
            exit()
    else:
        print(f'\nno ckpt found @{ckpt_path}')
        start_epoch = 0
        psnr_validation = 22.0
        if args.backtrack:
            state = {'psnr_validation': psnr_validation,
                     'epoch': 0,
                     'config': config_dict,
                     'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict(), }
            torch.save(state, ckpt_path + '_lasteval')
print(f'... starting training ...\n')
epoch = start_epoch

while epoch < args.num_epochs:
    tic = time.time()
    phases = ['train',  'val']
    for phase in phases:
        if phase == 'train':
            if (epoch % args.lr_step) == 0 and (epoch != 0) :
                step_lr(optimizer, args.lr_decay)
            model.train()
        elif phase == 'val':
            if not (args.backtrack and ((epoch+1) % args.validation_every == 0)):
                continue
            model.eval()   # Set model to evaluate mode
            print(f'\nstarting validation on train set with stride {args.stride_val}...')
            # Iterate over data.
        num_iters = 0
        psnr_set = 0
        loss_set = 0

        loader = loaders[phase]

        for batch in tqdm(loader,disable=not args.tqdm):
            img_clean, img_noise = batch
            img_clean = img_clean.to(device=device)
            img_noise = img_noise.to(device=device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                output = model(img_noise)
                loss = ((output - img_clean)).pow(2).sum() / img_clean.shape[0]
                loss_psnr = -10 * torch.log10((output - img_clean).pow(2).mean([1, 2, 3])).mean()
                loss.backward()
                optimizer.step()
            psnr_set += loss_psnr.item()
            loss_set += loss.item()
            num_iters += 1
        tac = time.time()
        psnr_set /= num_iters
        loss_set /= num_iters

        psnr[phase][epoch] = psnr_set

        if torch.cuda.is_available():
            mem_used, max_mem = show_mem()
            tqdm.write(f'epoch {epoch} - {phase} psnr: {psnr[phase][epoch]:0.4f} ({tac-tic:0.1f} s,  {(tac - tic) / num_iters:0.3f} s/iter, max gpu mem allocated {max_mem:0.1f} Mb, lr {get_lr(optimizer):0.1e})')
        else:
            tqdm.write(f'epoch {epoch} - {phase} psnr: {psnr[phase][epoch]:0.4f} loss: {loss_set:0.4f} ({(tac-tic)/num_iters:0.3f} s/iter,  lr {get_lr(optimizer):0.2e})')
        with open(f'{log_file_name}', 'a') as log_file:
            log_file = open(log_file_name, 'a')
            log_file.write(
                f'epoch {epoch} - {phase} psnr: {psnr[phase][epoch]:0.4f} loss: {loss_set:0.4f} ({(tac - tic) / num_iters:0.3f} s/iter,  lr {get_lr(optimizer):0.2e})\n')
            # output_file.close()
        with open(f'{out_dir}/{phase}.psnr','a') as psnr_file:
            psnr_file.write(f'{psnr[phase][epoch]:0.4f}\n')
    epoch += 1
    ##################### saving #################
    if epoch % 1 == 0:
        save_checkpoint({'epoch': epoch,
                         'config': config_dict,
                         'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'psnr_validation': psnr_validation},   os.path.join(out_dir+'/ckpt_'+str(epoch)))
    save_checkpoint({'epoch': epoch,
                     'config': config_dict,
                     'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'psnr_validation':psnr_validation}, ckpt_path)
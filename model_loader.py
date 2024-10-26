import torch
import os
from collections import OrderedDict
from model.KMDSNet import KMDSNetParams
from model.KMDSNet import KMDSNet4D

def init_model(kernel_size,num_filters,unfoldings,lmbda_prox,stride,multi_theta,verbose,inner_num,iter_num,use_kernel,kernel_depth,kernel_relu,frame_num):
    params = KMDSNetParams(inner_num=inner_num, iter_num=iter_num, use_kernel=use_kernel, kernel_depth=kernel_depth, frame_num=frame_num)
    model = KMDSNet4D(params)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('Nb tensors: ',len(list(model.named_parameters())), "; Trainable Params: ", pytorch_total_params)
    return model
def load_model(model_name,model):
    out_dir = os.path.join(model_name)
    ckpt_path = os.path.join(out_dir)
    if os.path.isfile(ckpt_path):
        try:
            print('\n existing ckpt detected')
            checkpoint = torch.load(ckpt_path)
            start_epoch = checkpoint['epoch']
            state_dict = checkpoint['state_dict']
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                # name = k
                name = k[7:]  # remove 'module.' of dataparallel
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict, strict=True)
            print(f"=> loaded checkpoint '{ckpt_path}' (epoch {start_epoch})")
            # print(f"=> loaded checkpoint '{ckpt_path}' (epoch {start_epoch})")
        except Exception as e:
            print(e)
            print(f'ckpt loading failed @{ckpt_path}, exit ...')
            exit()
    else:
        print(f'\nno ckpt found @{ckpt_path}')
        exit()

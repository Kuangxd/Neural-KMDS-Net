# Neural KMDS-Net: A Kernel Space-based Multidimensional Sparse Model for Dynamic PET Image Denoising

## train
```bash
python 00-train.py --lr 5e-3 --gpus 1 --train_batch 2 --lr_step 80 --inner_num 24 --iter_num 20 --use_kernel 1 --kernel_depth 2 --train_path xxx --model_name xxx
```

## test
```bash
python 01-test.py --gpus 0 --frame_num 24 --inner_num 24 --iter_num 20 --use_kernel 1 --kernel_depth 2 --model_name checkpoints/ckpt
```

## Acknowledgments
Code borrows from [SMDSNet](https://github.com/bearshng/smds-net). 

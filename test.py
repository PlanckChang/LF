import imageio
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.nn.functional as functional

import argparse
import numpy as np
import os
from os.path import join
import math
import time
import sys
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import io


from model.model import DispNet
from utils import dataset, util
from model.net_utils import lf2sublfs
from model.model2 import Net1
# Device configuration

# device = 'cpu'

# Test settings
parser = argparse.ArgumentParser(description="LF depth estimation: test")
# parser.add_argument("--model_dir", type=str, default="pretrained_models", help="folder containing the pretrained models")
# parser.add_argument("--save_dir", type=str, default="results", help="folder to save the test results")

parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--step", type=int, default=250, help="Learning rate decay every n epochs")
parser.add_argument("--reduce", type=float, default=0.5, help="Learning rate decay")
parser.add_argument("--patch_size", type=int, default=48, help="Training patch size")
parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
# parser.add_argument("--mixed_precision", type=bool, default=True, help="use mixed precision or not")

parser.add_argument("--angular_num", type=int, default=7, help="angular number of the light field ")

parser.add_argument("--weight_conf", type=float, default=0.0, help="weight for smooth loss ")
parser.add_argument("--train_dataset", type=str, default="HCI", help="dataset for training")
parser.add_argument("--loss", type=str, default='MaskQuarterMinLoss')
parser.add_argument("--loss_crop", type=float, default=8)
parser.add_argument("--std_thres", type=float, default=0.3)

parser.add_argument("--test_data", type=str, default="", help="dataset for test")
parser.add_argument("--test_path", type=str, default="./LFData/test_data_HCI_RGB", help="dataset for test")

parser.add_argument("--cuda", type=int, default=0, help="specify GPU")
parser.add_argument("--test_batch_size", type=int, default=1)
parser.add_argument("--net_name", type=str, default="jingjin_LF_10-5", help="net name with epoch")
parser.add_argument("--epoch", type=int, default=520, help="test epoch")
parser.add_argument("--weight_smooth", type=float, default=0.00001, help="weight for smooth loss ")
opt = parser.parse_args()
print(opt)

def write_pfm(data, fpath, scale=1, file_identifier=b'Pf', dtype="float32"):
    # PFM format definition: http://netpbm.sourceforge.net/doc/pfm.html

    data = np.flipud(data)
    height, width = np.shape(data)[:2]
    values = np.ndarray.flatten(np.asarray(data, dtype=dtype))
    endianess = data.dtype.byteorder
    print(endianess)

    if endianess == '<' or (endianess == '=' and sys.byteorder == 'little'):
        scale *= -1

    with open(fpath, 'wb') as file:
        file.write((file_identifier))
        file.write(('\n%d %d\n' % (width, height)).encode())
        file.write(('%d\n' % scale).encode())

        file.write(values)


def main():
    # Device configuration

    opt.device = torch.device('cuda:'+str(opt.cuda) if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cuda:'+str(opt.cuda) if torch.cuda.is_available() else 'cpu')
#     opt.device = torch.device('cpu')
    device = opt.device
    # generate save folder
    if not os.path.exists('save_results'):
        os.makedirs('save_results')
    opt.save_dir = 'save_results/res_Coarse_{}_an{}_sm{}_cf{}_bs{}_ps{}_lr{}_e{}_{}'.format(opt.train_dataset, opt.angular_num, opt.weight_smooth, opt.weight_conf,  opt.batch_size, opt.patch_size, opt.lr, opt.epoch, opt.test_data)
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)

    # Data loader
    print('===> Loading test datasets')
    test_set = dataset.TestDataFromHdf5(opt)
    test_loader = DataLoader(dataset=test_set, batch_size=opt.test_batch_size, shuffle=False)
    print('loaded {} LFIs from {}'.format(len(test_loader)*opt.test_batch_size, opt.test_path))

    # Build model
    print("building net")
    opt.sub_lf_num = math.ceil(opt.angular_num / 2) ** 2
    opt.ind_ref = (opt.angular_num - 1) // 2 * (opt.angular_num + 1)
    # model = DispNet(opt).to(device)
    model = Net1(opt.device).to(opt.device)
    print('# params: ', sum(param.numel() for param in model.parameters()))

    # load model
    opt.model_dir = '{}_Coarse_ckp_{}_an{}_sm{}_cf{}_ps{}_bs{}_lr{}'.format(
        opt.net_name, opt.train_dataset, opt.angular_num, opt.weight_smooth,  opt.weight_conf, opt.patch_size,opt.batch_size, opt.lr)


    if not os.path.exists(opt.model_dir):
        print('model folder is not found ')

    resume_path = join(opt.model_dir, "model_epoch_{}.pth".format(opt.epoch))
    print(f"resume path {resume_path}")
    checkpoint = torch.load(resume_path, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.to(opt.device)
    print('loaded model'.format(resume_path))

    # testing
    print("===> testing epoch {}".format(opt.epoch))
    model.eval()
    with torch.no_grad():
        for k, batch in enumerate(test_loader):

            lf = batch[0].to(device)
            # N, an2, c, h, w = lf.shape
            # lf = lf.squeeze(2)
            # print('lf', lf.shape)
            lf_name = batch[1][0]
            print('testing LFI {}: {}'.format(k, lf_name))


            # disp_coarse_sub, conf_coarse_sub, disp_coarse, mask, disp_refine, disp_final = model(lf, opt)
            subLFs = lf2sublfs(lf)
            # print(subLFs.shape)  # 1, 4, 16, 3, 512, 512

            # patch_num = 4
            # addition_patch = 32
            # patch_size = subLFs.shape[-1] // patch_num
            # division = []
            # division.append(subLFs[:,:,:, :, :patch_size+addition_patch, :patch_size+addition_patch])
            # division.append(subLFs[:, :, :, :, :patch_size+addition_patch, patch_size-addition_patch:])
            # division.append(subLFs[:,:,:,:, patch_size-addition_patch:, :patch_size+addition_patch])
            # division.append(subLFs[:,:,:,: patch_size-addition_patch:, patch_size-addition_patch:])

            start = time.time()
            out_disp_coarse_sub, out_conf_coarse_sub, out_disp_coarse_avg = model(subLFs)


            end = time.time()
            time_consuming = end-start
            print('runtime: ', end-start)

            print('len ', len(out_disp_coarse_avg))

            for i in range(len(out_disp_coarse_avg)):
                disp_coarse_sub = out_disp_coarse_sub[i]
                conf_coarse_sub = out_conf_coarse_sub[i]
                disp_coarse_avg = out_disp_coarse_avg[i]


                mdic = {"disp_coarse_sub": disp_coarse_sub[0].cpu().numpy()}
                mat_name = '{}/{}_s{}_disp_coarse_sub.mat'.format(opt.save_dir, lf_name, i+1)
                io.savemat(mat_name, mdic)

                mdic = {"conf_coarse_sub": conf_coarse_sub[0].cpu().numpy()}
                mat_name = '{}/{}_s{}_conf_coarse_sub.mat'.format(opt.save_dir, lf_name, i+1)
                io.savemat(mat_name, mdic)

                mdic = {"disp_coarse_avg": disp_coarse_avg[0,0].cpu().numpy()}
                mat_name = '{}/{}_s{}_disp_coarse_avg.mat'.format(opt.save_dir, lf_name, i+1)
                io.savemat(mat_name, mdic)


                conf_ind = torch.argmax(conf_coarse_sub, dim=1, keepdim=True)
                disp_coarse_max = torch.gather(disp_coarse_sub, 1, conf_ind)
                # print(disp_coarse_max.shape)
                mdic = {"disp_coarse_max": disp_coarse_max[0,0].cpu().numpy()}
                mat_name = '{}/{}_s{}_disp_coarse_max.mat'.format(opt.save_dir, lf_name, i+1)
                io.savemat(mat_name, mdic)


                disp_std = torch.std(disp_coarse_sub, dim=1, keepdim=False) #[N,h,w]
                mask =  torch.where(disp_std > opt.std_thres, 1, 0).unsqueeze(1)
                disp_coarse_final = disp_coarse_max * mask + disp_coarse_avg * (1-mask)
                mdic = {"disp_coarse_final": disp_coarse_final[0,0].cpu().numpy()}
                mat_name = '{}/{}_s{}_disp_coarse_final_thres{}.mat'.format(opt.save_dir, lf_name, i+1, opt.std_thres)
                io.savemat(mat_name, mdic)

                res_tem_zzw = mdic['disp_coarse_final']

                path_judge = f'/data/GuoLongZhao/dachuang/evaluation-toolkit/algo_results/{opt.net_name}_{opt.epoch}/disp_maps/'
                path_judge2 = f'/data/GuoLongZhao/dachuang/evaluation-toolkit/algo_results/{opt.net_name}_{opt.epoch}/runtimes/'
                path_judge3 = f'./eval_result/{opt.net_name}_{opt.epoch}/'
                if not os.path.exists(path_judge):
                    # 如果不存在，则创建目录（多层）
                    os.makedirs(path_judge)
                if not os.path.exists(path_judge2):
                    os.makedirs(path_judge2)
                if not os.path.exists(path_judge3):
                    os.makedirs(path_judge3)
                write_pfm(res_tem_zzw,
                          f'{path_judge}/{lf_name}.pfm')
                imageio.imsave(f'{path_judge3}/{lf_name}.png', np.uint8(res_tem_zzw * 30 + 120))
                imageio.imsave(
                    f'{path_judge}/{lf_name}.png',
                    np.uint8(res_tem_zzw * 30 + 120))
                with open(f'{path_judge2}/{lf_name}.txt', 'w') as file:
                    file.write(str(end-start))


                mdic = {"mask": mask[0, 0].cpu().numpy()}
                mat_name = '{}/{}_s{}_mask_thres{}.mat'.format(opt.save_dir, lf_name, i+1, opt.std_thres)
                io.savemat(mat_name, mdic)
                # print(mdic)





if __name__ == '__main__':

    main()

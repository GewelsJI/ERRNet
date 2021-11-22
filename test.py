import os
import torch
import argparse
import numpy as np
from scipy import misc
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from model.ERRNet import ERRNet
from dataset import test_dataset


def Evaluator(
    test_root='./dataset/TestDataset/', 
    snap_path='./snapshot/ERRNet_Snapshot.pth', 
    save_path='./result/',
    trainsize=352):

    os.makedirs(save_path, exist_ok=True)

    model = ERRNet().cuda()        
    model.load_state_dict(torch.load(snap_path))
    model.eval()

    for _data in ['CHAMELEON', 'CAMO', 'COD10K', 'NC4K']:
        test_data_root = os.path.join(test_root, _data)
        test_dataloader = test_dataset(image_root=test_data_root + 'Imgs/', gt_root=test_data_root + 'GT/', testsize=trainsize)

        with torch.no_grad():
            for i in range(test_dataloader.size):
                image, gt, name = test_dataloader.load_data()
                gt = np.asarray(gt, np.float32)

                image = image.cuda()
                
                output = model(image)

                output = F.upsample(output[0], size=gt.shape, mode='bilinear', align_corners=False)
                output = output.sigmoid().data.cpu().numpy().squeeze()
                output = (output - output.min()) / (output.max() - output.min() + 1e-8)

                misc.imsave(save_path + name, output)
                print('Prediction: {}'.format(save_path + name))


if __name__ == '__main__':
    Evaluator()
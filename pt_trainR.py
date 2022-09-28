from pt_network import RewriteNet
from pt_dataset import Fonter
from pt_util import render_fonts_image


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision

import numpy as np


class TotalVariationLoss(nn.Module):
    def __init__(self):
        super(TotalVariationLoss, self).__init__()
        self.loss = torch.nn.MSELoss()

    def forward(self, images):
        bs_img, c_img, h_img, w_img = images.size()
        #tv_h = torch.pow(images[:,:,1:,:]-images[:,:,:-1,:], 2).sum()
        #tv_w = torch.pow(images[:,:,:,1:]-images[:,:,:,:-1], 2).sum()
        tv_h = self.loss(images[:,:,1:,:],images[:,:,:-1,:])
        tv_w = self.loss(images[:,:,:,1:],images[:,:,:,:-1])
        return tv_h / (h_img) + tv_w / (w_img)


def train(batch_size=64, lr=1e-2, epoch=30, tv= 0.0001):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    net = RewriteNet('small')
    net = net.to(device=device)
    net.train()

    #TODO: create dateset
    #train_dataset = Fonter('./path_to_save_bitmap/simsun.npy', './path_to_save_bitmap/simkai.npy', 0, 2000)
    transform = torchvision.transforms.ToTensor()
    train_dataset = Fonter('./path_to_save_bitmap/simsun.npy', './path_to_save_bitmap/simkai.npy', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_img_oring, valid_img_target = next(iter(valid_loader))
    valid_img_oring = valid_img_oring.to(device)
    valid_img_target = valid_img_target.to(device)


    total_variation_loss = TotalVariationLoss()
    pixel_abs_loss = nn.L1Loss()
    optim = torch.optim.RMSprop(params=net.parameters(), lr=lr)

    _iter = 0
    for epoch_id in range(epoch):
        for batch_id, data in enumerate(train_loader):
            net.train()
            ori_images, target_images = data
            ori_images = ori_images.to(device)
            target_images = target_images.to(device)

            predicts = net(ori_images)

            loss = tv * total_variation_loss(predicts) + pixel_abs_loss(predicts, target_images)

            optim.zero_grad()
            loss.backward()
            optim.step()

            _iter += 1

            with torch.no_grad():
                if _iter == 1:
                    render_fonts_image(valid_img_target.clone().detach().cpu().squeeze(dim=1).numpy(), f'process/target.png', int(np.sqrt(batch_size)))
                if _iter % 10 == 0:
                    print(f'epoch_id:{epoch_id}, iter:{_iter}, loss:{loss.item()}')
                    net.eval()
                    predicts = net(valid_img_oring)
                    saveimag=predicts.clone().detach().cpu().squeeze(dim=1).numpy()
                    render_fonts_image(saveimag, f'process/epoch{epoch_id}_iter{_iter}.png', int(np.sqrt(batch_size)))



    torch.save(net.state_dict(), "writing.pth")

if __name__ == "__main__":
    train()
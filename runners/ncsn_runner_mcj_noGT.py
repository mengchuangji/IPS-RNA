import numpy as np
import glob
import tqdm

import torch.nn.functional as F
import torch
import os
from torchvision.utils import make_grid, save_image
from utils.make_grid_h import make_grid_h
from torch.utils.data import DataLoader
from models.ncsnv2 import NCSNv2Deeper, NCSNv2, NCSNv2Deepest
from models.UNet import UNet
from datasets import get_dataset, data_transform, inverse_data_transform
from models import general_anneal_Langevin_dynamics,general_anneal_Langevin_dynamics_den,general_anneal_Langevin_dynamics_inp
from models import get_sigmas
from models.ema import EMAHelper
from filter_builder import get_custom_kernel
import scipy.io as sio
from seis_utils.localsimi import localsimi

__all__ = ['NCSNRunner']


def get_model(config):
    if config.data.dataset == 'CELEBA' or config.data.dataset == 'marmousi':
        return NCSNv2(config).to(config.device)
    elif config.data.dataset == 'LSUN':
        return NCSNv2Deeper(config).to(config.device)
    elif config.data.dataset == 'marmousi_unet':
        return UNet(config).to(config.device)

class NCSNRunner():
    def __init__(self, args, config):
        self.args = args
        self.config = config
        args.log_sample_path = os.path.join(args.log_path, 'samples')
        os.makedirs(args.log_sample_path, exist_ok=True)

    def sample_general(self, score, samples, init_samples, sigma_0, sigmas, num_variations = 8, deg = 'sr4'):
        ## show stochastic variation ##
        stochastic_variations = torch.zeros((3 + num_variations) * self.config.sampling.batch_size,
                                            self.config.data.channels, self.config.data.image_shape[0],
                                            self.config.data.image_shape[1])
        stochastic_variations_R = torch.zeros((3 + num_variations) * self.config.sampling.batch_size,
                                              self.config.data.channels, self.config.data.image_shape[0],
                                              self.config.data.image_shape[1])
        stochastic_variations_LS = torch.zeros((3 + num_variations) * self.config.sampling.batch_size,
                                               self.config.data.channels, self.config.data.image_shape[0],
                                               self.config.data.image_shape[1])


        img_dim = self.config.data.image_shape[0] * self.config.data.image_shape[1]

        self.config.data.image_size = self.config.data.image_shape[1]


        if deg == 'den':
            y_0 = samples.view(samples.shape[0], self.config.data.channels,
                               img_dim)
            sio.savemat(os.path.join(self.args.image_folder, "y_0.mat"),
                        {'data': (y_0).view(samples.shape[0], self.config.data.channels,
                                            self.config.data.image_shape[0], self.config.data.image_shape[1]).cpu().squeeze().numpy()})

            pinv_y_0 = y_0.view(samples.shape[0] * self.config.data.channels,
                                img_dim, 1)


        sample_y_0 = inverse_data_transform(self.config, pinv_y_0.view(samples.shape[0], self.config.data.channels,
                                      self.config.data.image_shape[0], self.config.data.image_shape[1]))



        stochastic_variations[0 * self.config.sampling.batch_size: 1 * self.config.sampling.batch_size, :, :,:] = sample_y_0
        stochastic_variations_R[0 * self.config.sampling.batch_size: 1 * self.config.sampling.batch_size, :, :,
            :] = 0
        stochastic_variations_LS[0 * self.config.sampling.batch_size: 1 * self.config.sampling.batch_size, :, :,
                :] = 0

        # If you do not want to get the sampling trajectory plot, please comment
        if deg=='den':
            index = np.abs(sigmas - sigma_0).argmin()
            index_arr=range(index,len(sigmas),(len(sigmas)-index)//4)
            index_arr_len=len(index_arr)
            if index_arr[-1]==499:
                x_t_list_len= index_arr_len+1
            else:
                x_t_list_len = index_arr_len+1
        else:
            x_t_list_len = 11  # 11
        stochastic_variations_x_t = torch.zeros(
            ((1 + x_t_list_len) * num_variations) * self.config.sampling.batch_size,
            self.config.data.channels, self.config.data.image_shape[0], self.config.data.image_shape[1])

        ## apply SNIPS ##
        for i in range(num_variations):

            # Posterior sampling for denoising tasks
            all_samples, x_t_list = general_anneal_Langevin_dynamics_den(y_0, init_samples, score, sigmas,
                                                           self.config.sampling.n_steps_each,
                                                           self.config.sampling.step_lr, verbose=True,
                                                           final_only=self.config.sampling.final_only,
                                                           denoise=self.config.sampling.denoise, c_begin=0,
                                                           sigma_0=sigma_0)


            sample = all_samples[-1].view(all_samples[-1].shape[0], self.config.data.channels,
                                      self.config.data.image_shape[0], self.config.data.image_shape[1]).to(self.config.device)
            stochastic_variations[(self.config.sampling.batch_size) * (i+1) : (self.config.sampling.batch_size) * (i+2),:,:,:] = inverse_data_transform(self.config, sample)


            stochastic_variations_x_t[i * (1+x_t_list_len): i * (1+x_t_list_len)+1 :, :,:] = sample_y_0
            stochastic_variations_R[
            (self.config.sampling.batch_size) * (i + 1): (self.config.sampling.batch_size) * (i + 2), :, :,
            :] = sample_y_0 - inverse_data_transform(self.config, sample)

            ## If you do not want to calculate LS, please comment
            # if deg == 'den':
            #     LS = localsimi(inverse_data_transform(self.config, sample).cpu().squeeze().numpy(),
            #                    (sample_y_0 - inverse_data_transform(self.config, sample)).cpu().squeeze().numpy(),
            #                    rect=[5, 5, 1], niter=20, eps=0.0, verb=1).squeeze()[np.newaxis, np.newaxis, :]
            #     energy_simi = np.sum(LS ** 2) / LS.size
            #     print("energy_simi=", energy_simi)
            #     LS = torch.from_numpy(LS).contiguous().type(torch.FloatTensor).to(sample_y_0.device)
            #     stochastic_variations_LS[
            #     (self.config.sampling.batch_size) * (i + 1): (self.config.sampling.batch_size) * (i + 2), :, :,
            #     :] = LS


            for j, x_t in enumerate(x_t_list):
                x_t = x_t.view(sample.shape[0], self.config.data.channels,
                                          self.config.data.image_shape[0], self.config.data.image_shape[1]).to(self.config.device)
                stochastic_variations_x_t[
                (self.config.sampling.batch_size)* (j+1+(1+len(x_t_list))*i): (self.config.sampling.batch_size)* (j+2+(1+len(x_t_list))*i), :, :,
                :] = inverse_data_transform(self.config, x_t)
                # print(j+1+(1+len(x_t_list))*i)

        ## x_t evolution ##
        image_grid = make_grid(stochastic_variations_x_t, 1+x_t_list_len,padding=4)
        # save_image(image_grid, os.path.join(self.args.image_folder, 'stochastic_variation_x_t.png'))
        import matplotlib.pyplot as plt
        plt.gcf().set_size_inches(15,15)
        # plt.gcf().set_size_inches(3*(2 + len(x_t_list)*num_variations) , 3*self.config.sampling.batch_size)  # 设置图像尺寸为 10x6
        plt.imshow(image_grid.numpy().squeeze().transpose((1, 2, 0))[:, :, 0], cmap=plt.cm.seismic, vmin=-1, vmax=1)
        # plt.colorbar()  # 添加色标
        plt.axis('off')  # 关闭坐标轴
        plt.savefig(os.path.join(self.args.image_folder, 'stochastic_variation_x_t.png'), dpi=300,bbox_inches='tight')
        sio.savemat(os.path.join(self.args.image_folder, "results_x_t.mat"), {'data': stochastic_variations_x_t.cpu().squeeze().numpy()})

        # calculate mean and std ##
        runs = stochastic_variations[(self.config.sampling.batch_size) * (1) : (self.config.sampling.batch_size) * (1+num_variations),:,:,:]
        runs = runs.view(-1, self.config.sampling.batch_size, self.config.data.channels,
                          self.config.data.image_shape[0], self.config.data.image_shape[1])
        stochastic_variations[(self.config.sampling.batch_size) * (-2) : (self.config.sampling.batch_size) * (-1),:,:,:] = torch.mean(runs, dim=0)
        stochastic_variations[(self.config.sampling.batch_size) * (-1) : ,:,:,:] = torch.std(runs, dim=0)
        stochastic_variations_R[(self.config.sampling.batch_size) * (-2): (self.config.sampling.batch_size) * (-1), :, :,
        :] = sample_y_0.cpu()-torch.mean(runs, dim=0)
        stochastic_variations_R[(self.config.sampling.batch_size) * (-1):, :, :, :] = 0
        if deg == 'den':
            LS = localsimi(torch.mean(runs, dim=0).cpu().squeeze().numpy(),
                           (sample_y_0.cpu() - torch.mean(runs, dim=0)).cpu().squeeze().numpy(), rect=[5, 5, 1],
                           niter=20, eps=0.0, verb=1).squeeze()[np.newaxis, np.newaxis, :]
            energy_simi = np.sum(LS ** 2) / LS.size
            print("energy_simi=", energy_simi)
            LS = torch.from_numpy(LS).contiguous().type(torch.FloatTensor).to(sample_y_0.device)
            stochastic_variations_LS[(self.config.sampling.batch_size) * (-2): (self.config.sampling.batch_size) * (-1), :,
            :,
            :] = LS
            stochastic_variations_LS[(self.config.sampling.batch_size) * (-1):, :, :, :] = 0

        ######### plot stochastic_variations ###############
        image_grid = make_grid_h(stochastic_variations,  self.config.sampling.batch_size,padding=8)
        # save_image(image_grid, os.path.join(self.args.image_folder, 'stochastic_variation.png'))
        # import matplotlib.pyplot as plt
        plt.gcf().set_size_inches(10, 10)
        # plt.gcf().set_size_inches(3*(4 + num_variations), 3*self.config.sampling.batch_size)  # 设置图像尺寸为 10x6
        plt.imshow(image_grid.numpy().squeeze().transpose((1, 2, 0))[:, :, 0], cmap=plt.cm.seismic, vmin=-1, vmax=1)
        # plt.colorbar()  # 添加色标
        plt.axis('off')  # 关闭坐标轴
        plt.savefig(os.path.join(self.args.image_folder, 'stochastic_variation.png'), dpi=300,bbox_inches='tight')
        sio.savemat(os.path.join(self.args.image_folder, "results.mat"),
                    {'data': stochastic_variations.cpu().squeeze().numpy()})

        # If you do not want to calculate residual, please comment
        ######### plot stochastic_variations_R (residual) ###############
        image_grid = make_grid_h(stochastic_variations_R, self.config.sampling.batch_size,padding=8)
        # save_image(image_grid, os.path.join(self.args.image_folder, 'stochastic_variation.png'))
        # import matplotlib.pyplot as plt
        plt.gcf().set_size_inches(10, 10)
        # plt.gcf().set_size_inches(3*(4 + num_variations), 3*self.config.sampling.batch_size)  # 设置图像尺寸为 10x6
        plt.imshow(image_grid.numpy().squeeze().transpose((1, 2, 0))[:, :, 0], cmap=plt.cm.seismic, vmin=-1, vmax=1)
        # plt.colorbar()  # 添加色标
        plt.axis('off')  # 关闭坐标轴
        plt.savefig(os.path.join(self.args.image_folder, 'stochastic_variation_R.png'), dpi=300, bbox_inches='tight')
        sio.savemat(os.path.join(self.args.image_folder, "results_residual.mat"),
                    {'data': stochastic_variations_R.cpu().squeeze().numpy()})

        # If you do not want to calculate LS, please comment
        ######### plot stochastic_variations_LS (localsimi of denoised and noise) ###############
        # if deg == 'den':
        #     image_grid = make_grid_h(stochastic_variations_LS, self.config.sampling.batch_size, padding=8,pad_value=1)
        #     # save_image(image_grid, os.path.join(self.args.image_folder, 'stochastic_variation.png'))
        #     # import matplotlib.pyplot as plt
        #     plt.gcf().set_size_inches(10, 10)
        #     # plt.gcf().set_size_inches(3*(4 + num_variations), 3*self.config.sampling.batch_size)  # 设置图像尺寸为 10x6
        #     plt.imshow(image_grid.numpy().squeeze().transpose((1, 2, 0))[:, :, 0], cmap=plt.cm.jet, vmin=0, vmax=1)
        #     # plt.colorbar()  # 添加色标
        #     plt.axis('off')  # 关闭坐标轴
        #     plt.savefig(os.path.join(self.args.image_folder, 'stochastic_variation_LS.png'), dpi=300, bbox_inches='tight')
        #     sio.savemat(os.path.join(self.args.image_folder, "results_localsimi.mat"),
        #                 {'data': stochastic_variations_LS.cpu().squeeze().numpy()})


    

    def sample(self,obs):
        score, states = 0, 0
        if self.config.sampling.ckpt_id is None:
            states = torch.load(os.path.join(self.args.log_path_model, 'checkpoint.pth'), map_location=self.config.device)
        else:
            states = torch.load(os.path.join(self.args.log_path_model, f'checkpoint_{self.config.sampling.ckpt_id}.pth'),
                                map_location=self.config.device)

        score = get_model(self.config)
        score = torch.nn.DataParallel(score)

        score.load_state_dict(states[0], strict=True)

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(score)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(score)

        sigmas_th = get_sigmas(self.config)
        sigmas = sigmas_th.cpu().numpy()

        sigma_0 = self.args.sigma_0


        score.eval()


        samples = obs.to(self.config.device)
        samples = data_transform(self.config, samples)
        init_samples = torch.rand_like(samples)

        self.sample_general(score, samples, init_samples, sigma_0, sigmas, num_variations=self.args.num_variations, deg=self.args.degradation)

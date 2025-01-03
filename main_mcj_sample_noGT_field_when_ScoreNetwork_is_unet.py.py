import cv2
import argparse
import traceback
import time
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
import torch.utils.tensorboard as tb
import copy
# from runners import *
from runners.ncsn_runner_mcj_noGT import *
import os
import matplotlib.pyplot as plt
def plot(img,dpi,figsize):
    import matplotlib.pyplot as plt
    plt.figure(dpi=dpi, figsize=figsize)
    plt.imshow(img, vmin=-1, vmax=1, cmap=plt.cm.seismic)
    # plt.title('fake_noisy')
    # plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    # plt.axis('off')
    plt.tight_layout()
    plt.show()
def plot_cmap(img,dpi,figsize,data_range,cmap,cbar=False):

    plt.figure(dpi=dpi, figsize=figsize)
    plt.imshow(img, vmin=data_range[0], vmax=data_range[1], cmap=cmap)
    # plt.title('fake_noisy')
    if cbar:
        plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    # plt.axis('off')
    plt.tight_layout()
    plt.show()

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--config', type=str, default='marmousi_unet.yml',  help='Path to the config file') #celeba.yml
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--exp', type=str, default='exp', help='Path for saving running related data.')
    parser.add_argument('--doc', type=str, default='marmousi', help='A string for documentation purpose. '
                                                               'Will be the name of the log folder.') #celeba
    parser.add_argument('--comment', type=str, default='', help='A string for experiment comment')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('-i', '--image_folder', type=str, default='images', help="The folder name of samples") #celeba
    #D:\datasets\CelebA\CelebA\Img\img_align_celeba
    parser.add_argument('-n', '--num_variations', type=int, default=1, help='Number of variations to produce')
    parser.add_argument('-s', '--sigma_0', type=float, default=0.1, help='Noise std to add to observation')
    parser.add_argument('--degradation', type=str, default='den', help='Degradation: rec | den ')

    args = parser.parse_args()
    args.log_path = os.path.join(args.exp, 'logs', args.doc)


    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.load(f,Loader = yaml.FullLoader)
    new_config = dict2namespace(config)

    tb_path = os.path.join(args.exp, 'tensorboard', args.doc)

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

    # os.makedirs(os.path.join(args.exp, 'image_samples'), exist_ok=True)
    # args.image_folder = os.path.join(args.exp, 'image_samples', args.image_folder)

    # args.image_folder ='exp/logs/celeba/results/test'
    args.image_folder = 'exp/logs/marmousi/results'

    # if not os.path.exists(args.image_folder):
    #     os.makedirs(args.image_folder)
    # else:
    #     response = input("Image folder already exists. Overwrite? (Y/N)")
    #     if response.upper() == 'Y':
    #         overwrite = True
    #
    #     if overwrite:
    #         shutil.rmtree(args.image_folder)
    #         os.makedirs(args.image_folder)
    #     else:
    #         print("Output image folder exists. Program halted.")
    #         sys.exit(0)

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    import torch
    # torch.cuda.empty_cache()
    args, config = parse_args_and_config()
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))
    logging.info("Config =")

    args.doc='MmsSegyopenf_unet'
    args.log_path = os.path.join(args.exp, 'logs', args.doc)


    args.log_path_model ='./exp/logs/MmsSegyopenf_unet'

    args.image_folder = 'exp/logs/MmsSegyopenf_unet/results'

    # The ckpt_id of the trained SGMs, the SGMs model is provided. Please retrain it if necessary.
    config.sampling.ckpt_id = 220000 #or 800000

    # 'den' represents the denoising task, and 'rec' represents the reconstruction task (or simultaneous denoising and interpolation)
    args.degradation = 'den'  #

    # The number of data processed at the same time, custom, default is 1
    config.sampling.batch_size = 1
    # The number of samples (random solutions) you want to generate, custom
    args.num_variations = 3

    # The parameters of SGMs are fixed and do not need to be set during testing (sampling).
    config.data.seis_rescaled = True # False True
    config.model.num_classes=500
    config.model.sigma_begin=32
    config.model.sigma_end = 0.01 #0.01
    config.model.sigma_dist = 'geometric'

    print(">" * 80)
    config_dict = copy.copy(vars(config))
    print(yaml.dump(config_dict, default_flow_style=False))
    print("<" * 80)



    # data_dir = '/home/shendi_mcj/datasets/seismic/fielddata/'
    data_dir = '/home/shendi_mcj/datasets/seismic/test/'


    from seis_utils.readsegy import readsegy

    original = readsegy(data_dir + 'stk_IL41_XL45_CDPx4351.sgy')[40:40+512,800:800+512] #（4096，2091）

    y_max = abs(original).max()

    obs = original#[400:400+128,50:50+128]

    obs = obs/y_max
    obs=torch.from_numpy(obs).contiguous().view(1, -1, obs.shape[0], obs.shape[1]).type(torch.FloatTensor)

    # Automatic noise level estimation by VI-non-IID or the user can set it by himself (i.e., according to the interval [sigma_dict['min'],sigma_dict['max']]).
    from utils.estimate_sigma_using_VInonIID import estimate_sigma_using_VInonIID
    sigma_dict, sigma_map_prd =estimate_sigma_using_VInonIID(obs[0].view(1,-1,obs.shape[2],obs.shape[3]))
    plot_cmap(sigma_map_prd, 300, (3.7, 3), data_range=[0.0, 0.3], cmap=plt.cm.jet, cbar=True)

    ####################################
    # Moderate suppression mode
    args.sigma_0 = 1.0 * sigma_dict['median']

    # Mild suppression mode
    # args.sigma_0 = sigma_dict['min']
    # args.sigma_0 = sigma_dict['min']+0.1*(sigma_dict['median']-sigma_dict['min'])

    # Strong Suppression Mode
    # args.sigma_0 = sigma_dict['max'] #2*
    # args.sigma_0 = sigma_dict['median']+0.5*(sigma_dict['max']-sigma_dict['median'])
    #######################################

    try:
        runner = NCSNRunner(args, config)
        runner.sample(obs)
    except:
        logging.error(traceback.format_exc())

    return 0


if __name__ == '__main__':
    sys.exit(main())

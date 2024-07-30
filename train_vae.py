import copy
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ["all_proxy"] = "http://10.24.116.74:7890"
from tqdm import tqdm
from torch import nn, optim
import torch, argparse, math

# from diffusers.models.vae \
#     import AutoencoderKL, Encoder, Decoder, AutoencoderKLOutput, DiagonalGaussianDistribution, DecoderOutput
from diffusers.models.autoencoder_kl \
    import AutoencoderKL, Encoder, Decoder, AutoencoderKLOutput, DiagonalGaussianDistribution, DecoderOutput



torch.backends.cudnn.enabled = True

torch.backends.cudnn.benchmark = True
# from model.vq_model import VQModel
from lossers.lpips import LPIPS
# from datasets import load_dataset
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision import transforms, utils
from loggers import Logger
from torch.utils.data.sampler import WeightedRandomSampler
from ControlVAE import NewEncoder, NewDecoder
import os
from diffusers import StableDiffusionPipeline, DDIMScheduler


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def binary_dataset(root, transform):
    dset = datasets.ImageFolder(root, transform)
    return dset

def get_dataset(paths, transform):
    dset_lst = []
    for path in paths:
        dset = binary_dataset(path, transform)
        dset_lst.append(dset)
    return torch.utils.data.ConcatDataset(dset_lst)

def get_bal_sampler(dataset):
    targets = []
    for d in dataset.datasets:
        targets.extend(d.targets)

    # w = torch.tensor([0.7, 0.3])
    w = torch.tensor([0.0, 1.0])
    sample_weights = w[targets]
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights))
    return sampler

def create_dataloader(args, paths, transform):
    shuffle = True
    dataset = get_dataset(paths, transform)
    # sampler = SubsetRandomSampler(list(range(len(dataset))))
    sampler = get_bal_sampler(dataset)

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=args.batch_size,
                                              # shuffle=shuffle,
                                              sampler=sampler,
                                              num_workers=int(args.num_threads))
    return data_loader

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

from torch.utils.data import Dataset, DataLoader
import glob
import numpy as np
from PIL import Image

import torch
import torch.nn as nn

# version adaptation for PyTorch > 1.7.1
IS_HIGH_VERSION = tuple(map(int, torch.__version__.split('+')[0].split('.'))) > (1, 7, 1)
if IS_HIGH_VERSION:
    import torch.fft



class FFL(nn.Module):

    def __init__(self, loss_weight=1.0, alpha=1.0, patch_factor=1, ave_spectrum=False, log_matrix=False, batch_matrix=False, use_filter=False, use_single_filter=False):
        super(FFL, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix
        self.use_filter = use_filter
        self.use_single_filter = use_single_filter

    def tensor2freq(self, x):
        patch_factor = self.patch_factor
        _, _, h, w = x.shape
        assert h % patch_factor == 0 and w % patch_factor == 0, (
            'Patch factor should be divisible by image height and width')
        patch_list = []
        patch_h = h // patch_factor
        patch_w = w // patch_factor
        for i in range(patch_factor):
            for j in range(patch_factor):
                patch_list.append(x[:, :, i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w])

        y = torch.stack(patch_list, 1)

        if IS_HIGH_VERSION:
            freq = torch.fft.fft2(y, norm='ortho')
            freq = torch.stack([freq.real, freq.imag], -1)
        else:
            freq = torch.rfft(y, 2, onesided=False, normalized=True)
        return freq

    def loss_formulation(self, recon_freq, real_freq, matrix=None):
        if matrix is not None:
            weight_matrix = matrix.detach()
        else:
            matrix_tmp = (recon_freq - real_freq) ** 2
            matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha

            if self.log_matrix:
                matrix_tmp = torch.log(matrix_tmp + 1.0)

            if self.batch_matrix:
                matrix_tmp = matrix_tmp / matrix_tmp.max()
            else:
                matrix_tmp = matrix_tmp / matrix_tmp.max(-1).values.max(-1).values[:, :, :, None, None]

            matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
            matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
            weight_matrix = matrix_tmp.clone().detach()

        assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
            'The values of spectrum weight matrix should be in the range [0, 1], '
            'but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item()))

        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]

        loss = weight_matrix * freq_distance
        return torch.mean(loss)

    def forward(self, pred, target, matrix=None, **kwargs):
        if self.use_filter:
            n_channels = 3
            nb = 20
            filter_path = "./checkpoints/dncnn_color_blind.pth"
            from models.network_dncnn import DnCNN as net
            filter = net(in_nc=n_channels, out_nc=n_channels, nc=64, nb=nb, act_mode='R').cuda()
            filter.load_state_dict(torch.load(filter_path), strict=True)
            filter = filter.eval()
            if self.use_single_filter:
                with torch.no_grad():
                    pred = filter.noise(pred)
            else:
                pred = filter.noise(pred)
                target = filter.noise(target)
        
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)

        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)

        return self.loss_formulation(pred_freq, target_freq, matrix) * self.loss_weight



class RealDataset(Dataset):
    def __init__(self, paths, res=224):
        self.paths = paths
        self.data = self.get_data()
        self.res = res

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path = self.data[index]

        image = Image.open(path).convert("RGB")

        image = image.resize((self.res, self.res), resample=Image.LANCZOS)

        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)[:, :3, :, :].squeeze(0).cuda()
        return 2.0 * image - 1.0

        return sample

    def get_data(self):
        data_list = []
        for path in self.paths:
            data_list = data_list + glob.glob(path + "/*")
        return data_list


class FakeDataset(Dataset):
    def __init__(self, paths, res=224):
        self.paths = paths
        self.data = self.get_data()
        self.res = res

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path = self.data[index]

        dataname = path.split("/")[4]
        imgname = path.split("/")[-1]


        image = Image.open(path).convert("RGB")

        image = image.resize((self.res, self.res), resample=Image.LANCZOS)

        image = np.array(image).astype(np.float32) / 255.0
        # image1 = np.array(image1).astype(np.float32) / 255.0

        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)[:, :3, :, :].squeeze(0).cuda()


        image = 2.0 * image - 1.0
        return image

    def get_data(self):
        data_list = []
        for path in self.paths:
            data_list = data_list + glob.glob(path + "/*")
        return data_list

if __name__ == '__main__':

        parser = argparse.ArgumentParser(description="Train VQModel")
        parser.add_argument("--device", type=str, default="cuda:0")
        parser.add_argument("--name", type=str, default="1")
        parser.add_argument('--num_threads', default=0, type=int, help='# threads for loading data')
        parser.add_argument("--iter", type=int, default=2, help="total training iterations")
        parser.add_argument("--batch_size", type=int, default=32, help="batch sizes for each gpus")
        parser.add_argument("--size", type=int, default=224, help="image sizes for the model")
        parser.add_argument("--freq_log", type=int, default=20, help="")
        parser.add_argument("--freq_save", type=int, default=10000, help="")
        parser.add_argument("--cache_dir", type=str, default="./.cache")
        parser.add_argument("--arch", type=str,
                            default="resnet50")
        parser.add_argument("--flag", type=bool, default=True)
        parser.add_argument("--save_dir", type=str,
                            default = "checkpoints")
        parser.add_argument("--resume", type=str,
                            default = "")
        parser.add_argument("--noise_prototype", type=str,
                            default="/path/to/pretrain")
        parser.add_argument("--dir", type=str,
                            default="/path/to/genimage")
        # parser.add_argument('-d', '--dir', nargs='+', type=str, default=[
        #     "/path/to/nature",
        # ])

        args = parser.parse_args()
        ldm_stable = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base").to('cuda')
        if not os.path.exists("log"):
            os.mkdir("log")

        logger = Logger(name='demofiles', log_path='log/train{}.log'.format(args.name))
        lpips = LPIPS(net='vgg', cache_dir=args.cache_dir).cuda()


        l = ['ADMnew/imagenet_ai_0508_adm/train/nature', 'BigGAN/imagenet_ai_0419_biggan/train/nature', 'glide/imagenet_glide/train/nature',
         'Midjourney/imagenet_midjourney/train/nature', 'stable_diffusion_v_1_4/imagenet_ai_0419_sdv4/train/nature',
         'stable_diffusion_v_1_5/imagenet_ai_0424_sdv5/train/nature', 'VQDM/imagenet_ai_0419_vqdm/train/nature',
         'wukong/imagenet_ai_0424_wukong/train/nature']
        #
        paths_real = [args.dir + "/" + item for item in l]
        # paths_real = args.dir
        real_dataset = RealDataset(paths_real)
        if not os.path.exists(args.noise_prototype):
            real_dataset = RealDataset(paths_real)
            noise_prototype = torch.zeros_like(real_dataset[0].unsqueeze(0))
            n_channels = 3
            nb = 20
            filter_path = "./checkpoints/dncnn_color_blind.pth"
            from models.network_dncnn import DnCNN as net

            filter = net(in_nc=n_channels, out_nc=n_channels, nc=64, nb=nb, act_mode='R').cuda()
            filter.load_state_dict(torch.load(filter_path), strict=True)
            filter = filter.eval()
            for i in tqdm(range(len(real_dataset))):
                real = real_dataset[i].unsqueeze(0)
                with torch.no_grad():
                    real_noise = filter.noise(real)
                noise_prototype += real_noise
            noise_prototype /= len(real_dataset)
            torch.save(noise_prototype, args.noise_prototype)
        else:
            noise_prototype = torch.load(args.noise_prototype)



        # fake_dataset = FakeDataset(paths)

        real_dataloader = DataLoader(real_dataset, batch_size=args.batch_size, shuffle=True)

        params = {"act_fn": "silu",
                  "block_out_channels": [
                      128,
                      256,
                      512,
                      512
                  ],
                  "down_block_types": [
                      "DownEncoderBlock2D",
                      "DownEncoderBlock2D",
                      "DownEncoderBlock2D",
                      "DownEncoderBlock2D"
                  ],
                  "in_channels": 3,
                  "latent_channels": 4,
                  "layers_per_block": 2,
                  "norm_num_groups": 32,
                  "out_channels": 3,
                  "sample_size": 512,
                  "up_block_types": [
                      "UpDecoderBlock2D",
                      "UpDecoderBlock2D",
                      "UpDecoderBlock2D",
                      "UpDecoderBlock2D"
                  ]
                  }

        params_encoder = {
            "in_channels": params["in_channels"],
            "out_channels": params["latent_channels"],
            "down_block_types": params["down_block_types"],
            "block_out_channels": params["block_out_channels"],
            "layers_per_block": params["layers_per_block"],
            "act_fn": params["act_fn"],
            "norm_num_groups": params["norm_num_groups"],
            "double_z": True,
        }
        params_decoder = {
            "in_channels": params["latent_channels"],
            "out_channels": params["out_channels"],
            "up_block_types": params["up_block_types"],
            "block_out_channels": params["block_out_channels"],
            "layers_per_block": params["layers_per_block"],
            "act_fn": params["act_fn"],
            "norm_num_groups": params["norm_num_groups"],
        }

        encoder = NewEncoder(**params_encoder)

        if args.resume == "":
            encoder.encoder.load_state_dict(ldm_stable.vae.encoder.state_dict(), strict=False)
        else:
            p = args.resume
            state_dict = torch.load(p)
            encoder.load_state_dict(state_dict["encoder"], strict=False)

        decoder = NewDecoder(**params_decoder)

        decoder.load_state_dict(ldm_stable.vae.decoder.state_dict(), strict=False)

        requires_grad(encoder, True)
        ldm_stable.vae.decoder = decoder
        requires_grad(decoder, False)
        ldm_stable.vae.requires_grad_(False)
        ldm_stable.text_encoder.requires_grad_(False)
        ldm_stable.unet.requires_grad_(False)

        encoder = encoder.cuda()
        decoder = decoder.cuda()



        loss_l1 = torch.nn.L1Loss()
        loss_adv = torch.nn.CrossEntropyLoss()
        optimizer = optim.AdamW(list(encoder.parameters())+list(decoder.parameters()), lr=0.001)
        ffl = FFL(loss_weight=1.0, alpha=1.0, log_matrix=False)
        ffl_filter = FFL(loss_weight=1.0, alpha=1.0, log_matrix=False, ave_spectrum=True, use_filter=True, use_single_filter=True)

        pbar = tqdm(range(args.iter))

        generator = torch.Generator().manual_seed(8888)

        for idx in pbar:
            for idy, batch in enumerate(tqdm(real_dataloader)):
                x_batch = batch
                x_batch = x_batch.cuda()
                gpu_generator = torch.Generator(device=x_batch.device)
                gpu_generator.manual_seed(generator.initial_seed())
                if args.flag:
                    down_features = encoder(x_batch)[1]
                    latents = 0.18215 * ldm_stable.vae.encode(x_batch).latent_dist.sample(generator=gpu_generator)

                else:
                    a = encoder(x_batch)[0]
                    down_features = None
                    down_features_y = None
                    moments = ldm_stable.vae.quant_conv(a)
                    posterior = DiagonalGaussianDistribution(moments)

                    latents = 0.18215 * AutoencoderKLOutput(latent_dist=posterior).latent_dist.sample(generator=gpu_generator)

                decode_latents = ldm_stable.vae.post_quant_conv(1 / 0.1825 * latents)
                rec = decoder(decode_latents, down_features)
                lpips_loss = lpips(x_batch, rec).mean()
                l1_loss = loss_l1(x_batch, rec)



                loss_ffl_filter = ffl(x_batch, rec)
                loss_a = 0.
                loss = lpips_loss + l1_loss + 0.02 * loss_ffl_filter
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if idy % args.freq_log == 0:
                    logger.info("lpips_loss:{}  l1_loss:{} loss_ffl_filter:{}".format(lpips_loss, l1_loss, loss_ffl_filter))

                    with torch.no_grad():
                        x_batch = x_batch.cuda()

                        gpu_generator = torch.Generator(device=x_batch.device)
                        gpu_generator.manual_seed(generator.initial_seed())
                        if args.flag:
                            down_features = encoder(x_batch)[1]

                            latents = 0.18215 * ldm_stable.vae.encode(x_batch).latent_dist.sample(generator=gpu_generator)
                            decode_latents = ldm_stable.vae.post_quant_conv(1 / 0.1825 * latents)
                        else:
                            a = encoder(x_batch)[0]
                            down_features = None
                            moments = ldm_stable.vae.quant_conv(a)
                            posterior = DiagonalGaussianDistribution(moments)
                            latents = 0.18215 * AutoencoderKLOutput(latent_dist=posterior).latent_dist.sample(
                                generator=gpu_generator)

                        rec = decoder(decode_latents, down_features)

                        rec = (rec / 2 + 0.5).clamp(0, 1)
                        x_batch = (x_batch / 2 + 0.5).clamp(0, 1)

                        save_dir = args.save_dir
                        os.makedirs(save_dir, exist_ok=True)

                        save_dir_raw = os.path.join(save_dir, "sampleraw")
                        save_dir_recon = os.path.join(save_dir, "samplerecon")
                        os.makedirs(save_dir_raw, exist_ok=True)
                        os.makedirs(save_dir_recon, exist_ok=True)

                        utils.save_image(
                            x_batch, save_dir+"/"+f"sampleraw/{str(idy).zfill(6)}.png",
                            nrow=int(math.sqrt(args.batch_size)), normalize=False, value_range=(0, 1),
                        )
                        utils.save_image(
                            rec, save_dir+"/"+f"samplerecon/{str(idy).zfill(6)}.png",
                            nrow=int(math.sqrt(args.batch_size)), normalize=False, value_range=(0, 1),
                        )
                if idy % args.freq_save == 0:
                    save_dir = args.save_dir
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save({
                        'encoder': encoder.state_dict(),
                    }, "{}/{}_{}.pt".format(save_dir, idx, str(idy).zfill(6)),
                    )
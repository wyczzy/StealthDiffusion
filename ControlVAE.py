# from diffusers.models.vae \
#     import AutoencoderKL, Encoder, Decoder, AutoencoderKLOutput, DiagonalGaussianDistribution, DecoderOutput
from diffusers.models.autoencoder_kl \
    import AutoencoderKL, Encoder, Decoder, AutoencoderKLOutput, DiagonalGaussianDistribution, DecoderOutput
from typing import Optional, Tuple, Union
import torch.nn as nn
import torch
import numpy as np

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class NewEncoder(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 down_block_types=("DownEncoderBlock2D",),
                 block_out_channels=(64,),
                 layers_per_block=2,
                 norm_num_groups=32,
                 act_fn="silu",
                 double_z=False,
        ):
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels,
            out_channels=out_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=double_z)
        self.zero_conv1 = nn.Conv2d(128, 128, 3, 1, 1)
        self.zero_conv2 = nn.Conv2d(128, 512, 3, 1, 1)
        self.zero_conv3 = nn.Conv2d(256, 512, 3, 1, 1)
        self.zero_conv4 = nn.Conv2d(512, 512, 3, 1, 1)

        self.zero_conv1 = zero_module(self.zero_conv1)
        self.zero_conv2 = zero_module(self.zero_conv2)
        self.zero_conv3 = zero_module(self.zero_conv3)
        self.zero_conv4 = zero_module(self.zero_conv4)

        self.zero_convs = [
                           self.zero_conv1,
                           self.zero_conv2,
                           self.zero_conv3,
                           self.zero_conv4,
        ]

    def forward(self, x):
        sample = x    #1,3,224,224
        down_block_outputs = []
        sample = self.encoder.conv_in(sample) # 1,128,224,224

        down_block_outputs.append(sample)

        # for down_block in self.down_blocks:
        sample = self.encoder.down_blocks[0](sample) #1,128,112,112
        down_block_outputs.append(sample)
        sample = self.encoder.down_blocks[1](sample) #1,256,56,56
        down_block_outputs.append(sample)
        sample = self.encoder.down_blocks[2](sample)  #1,512,28,28
        sample = self.encoder.down_blocks[3](sample)  #1,512,28,28

        sample = self.encoder.mid_block(sample) #1,512,28,28
        down_block_outputs.append(sample)
        sample = self.encoder.conv_norm_out(sample) #1,512,28,28
        sample = self.encoder.conv_act(sample) #1,512,28,28
        sample = self.encoder.conv_out(sample) #1,8,28,28

        conv_down_outputs = []
        for down_block_output, zero_conv in zip(down_block_outputs, self.zero_convs):
            down_block_output = zero_conv(down_block_output)
            conv_down_outputs.append(down_block_output)


        return sample, conv_down_outputs

from diffusers.models.unet_2d_blocks import UNetMidBlock2D, get_down_block, get_up_block
class NewDecoder(Decoder):
    def __init__(
            self,
            in_channels=3,
            out_channels=3,
            up_block_types=("UpDecoderBlock2D",),
            block_out_channels=(64,),
            layers_per_block=2,
            norm_num_groups=32,
            act_fn="silu",
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2d(in_channels, block_out_channels[-1], kernel_size=3, stride=1, padding=1)

        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attn_num_head_channels=None,
            resnet_groups=norm_num_groups,
            temb_channels=None,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                prev_output_channel=None,
                add_upsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attn_num_head_channels=None,
                temb_channels=None,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

    def forward(self, z, l=None):

        sample = z
        sample = self.conv_in(sample) # 1 512 28 28
        if l is not None:
            l = l[::-1]
            # middle
            sample = self.mid_block(sample) + l[0]

            sample = self.up_blocks[0](sample) + l[1]
            sample = self.up_blocks[1](sample) + l[2]
            sample = self.up_blocks[2](sample)
            sample = self.up_blocks[3](sample) + l[3]
            sample = self.conv_norm_out(sample)

        else:
            sample = self.mid_block(sample) # 1 512 28 28

            sample = self.up_blocks[0](sample) # 1 512 56 56
            sample = self.up_blocks[1](sample) # 1 512 112 112
            sample = self.up_blocks[2](sample) # 1 256 224 224
            sample = self.up_blocks[3](sample) # 1 128 224 224
            sample = self.conv_norm_out(sample) # 1 128 224 224

        # post-process
        sample = self.conv_act(sample) # 1 128 224 224
        sample = self.conv_out(sample) # 1 3 224 224

        return sample


class NewAutoencoderKL(nn.Module):
        def __init__(
                self,
                in_channels: int = 3,
                out_channels: int = 3,
                down_block_types: Tuple[str] = ("DownEncoderBlock2D",),
                up_block_types: Tuple[str] = ("UpDecoderBlock2D",),
                block_out_channels: Tuple[int] = (64,),
                layers_per_block: int = 1,
                act_fn: str = "silu",
                latent_channels: int = 4,
                norm_num_groups: int = 32,
                sample_size: int = 32,
        ):
            super().__init__()

            # pass init params to Encoder
            self.encoder = Encoder(
                in_channels=in_channels,
                out_channels=latent_channels,
                down_block_types=down_block_types,
                block_out_channels=block_out_channels,
                layers_per_block=layers_per_block,
                act_fn=act_fn,
                norm_num_groups=norm_num_groups,
                double_z=True,
            )

            # pass init params to Decoder
            self.decoder = NewDecoder(
                in_channels=latent_channels,
                out_channels=out_channels,
                up_block_types=up_block_types,
                block_out_channels=block_out_channels,
                layers_per_block=layers_per_block,
                norm_num_groups=norm_num_groups,
                act_fn=act_fn,
            )

            self.quant_conv = torch.nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1)
            self.post_quant_conv = torch.nn.Conv2d(latent_channels, latent_channels, 1)

        def encode(self, x: torch.FloatTensor, return_dict: bool = True) -> AutoencoderKLOutput:
            h = self.encoder(x)
            moments = self.quant_conv(h)
            posterior = DiagonalGaussianDistribution(moments)

            if not return_dict:
                return (posterior,)

            return AutoencoderKLOutput(latent_dist=posterior)

        def decode(self, z: torch.FloatTensor, down_block_outputs: list = None, return_dict: bool = True) -> Union[DecoderOutput, torch.FloatTensor]:
            z = self.post_quant_conv(z)
            dec = self.decoder(z, down_block_outputs)

            if not return_dict:
                return (dec,)

            return DecoderOutput(sample=dec)

        def forward(
                self,
                sample: torch.FloatTensor,
                down_block_outputs: list = None,
                sample_posterior: bool = False,
                return_dict: bool = True,
                generator: Optional[torch.Generator] = None,
        ) -> Union[DecoderOutput, torch.FloatTensor]:
            x = sample
            posterior = self.encode(x).latent_dist
            if sample_posterior:
                z = posterior.sample(generator=generator)
            else:
                z = posterior.mode()
            dec = self.decode(z, down_block_outputs).sample

            if not return_dict:
                return (dec,)

            return DecoderOutput(sample=dec)




if __name__ == '__main__':
    # model = NewAutoencoderKL()
    import torch
    import os
    from diffusers import StableDiffusionPipeline, DDIMScheduler

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
    model = NewDecoder(**params_decoder)

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

    encoder = NewEncoder(**params_encoder)
    model(torch.randn(1,4,28,28))

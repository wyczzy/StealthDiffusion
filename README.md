## Requirements

1. Hardware Requirements and Software Requirements
    - GPU: 1x high-end NVIDIA GPU with at least 32GB memory
    - Python: 3.8
    - CUDA: 11.7
    - cuDNN: 8.9.7

   To install other requirements:

   ```
   pip install -r requirements.txt
   ```

2. Datasets and Pretrain-weights
Please download the dataset [GenImage](https://github.com/GenImage-Dataset/GenImage) and the data root directory should be organized as follows.
```
├── ADM
│   ├── imagenet_ai
│   │   ├── train
│   │   │   ├── ai
│   │   │   ├── nature
│   │   ├── val
│   │   │   ├── ai
│   │   │   ├── nature
├── BigGAN
│   ├── imagenet_ai
│   │   ├── train
│   │   │   ├── ai
│   │   │   ├── nature
│   │   ├── val
│   │   │   ├── ai
│   │   │   ├── nature
├── Glide
│   ├── imagenet_ai
│   │   ├── train
│   │   │   ├── ai
│   │   │   ├── nature
│   │   ├── val
│   │   │   ├── ai
│   │   │   ├── nature
├── Midjourney
│   ├── ...
├── Stable Diffusion V1.4
│   ├── ...
├── Stable Diffusion V1.5
│   ├── ...
├── VQDM
│   ├── ...
├── Wukong
│   ├── ...
```
The pre-trained weights are located at [checkpoints](https://pan.baidu.com/s/1gWCPS--IUbu3QWKlWD5bRA), with the extraction code z8se. Place them in the checkpoints folder.


## Train Forensic Detector
To train a detector with full GenImage training set, run this command:
```
python train.py --exam_dir <log path> --name <model name> --blur_prob 0.5 --blur_sig 0.0,3.0 --jpg_prob 0.5 --jpg_method cv2,pil --jpg_qual 30,100 --dataroot <dataset path>  --classes "ADM/imagenet_ai,BigGAN/imagenet_ai,glide/imagenet_ai,Midjourney/imagenet_ai,stable_diffusion_v_1_4/imagenet_ai,stable_diffusion_v_1_5/imagenet_ai,VQDM/imagenet_ai,wukong/imagenet_ai" --gpu_ids <gpu id> --loadSize <load size> --cropSize <crop size> --batch_size <batch size> --class_bal --arch <model type>
```
You can use the above commands to train Efficientnet-B0 (E), ResNet50 (R), DeiT (D), and Swin-T (S) models. After training, you can test the performance of the model in demo_dir.py. Replace "efficientnet-b0," "resnet50," "deit," and "swin-t" in the dic with the corresponding training weight paths. run this command:
```
python demo_dir.py -d <test dataset> -m <model path>
```


## Train Control-VAE
To train Control-VAE with full GenImage training set, run this command:
```
python train_vae.py --iter <training iterations> --save_dir <save path> --dir "/path/to/genimage"
```


## Generate Adversarial Examples
First, you should replace E, R, D, and S in other_attacks.py's dic with the corresponding training weight paths.

To generate adversarial examples, run this command:
```
python main.py --model_name <surrogate model> --save_dir <save path> --images_root </path/to/GenImage> --encoder_weights </path/to/Controlvae.pt>
```

## Ackownledgments
Some of the codes are built upon [DiffAttack](https://github.com/WindVChen/DiffAttack) and [CNNDetection](https://github.com/PeterWang512/CNNDetection). The experimental dataset comes from [GenImage](https://github.com/GenImage-Dataset/GenImage). Thanks them for their great works!
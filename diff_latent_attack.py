import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "3"
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from torch import optim
from utils import view_images, aggregate_attention
from distances import LpDistance
import cv2
import math
from lossers.lpips import LPIPS
from diffusers.models.autoencoder_kl \
    import AutoencoderKL, Encoder, Decoder, AutoencoderKLOutput, DiagonalGaussianDistribution, DecoderOutput
from ControlVAE import NewEncoder, NewDecoder, NewAutoencoderKL
import torchattacks


def calculate_psnr(img1, img2, border=0):
    # img1 and img2 have range [0, 255]
    # img1 = img1.squeeze()
    # img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h - border, border:w - border]
    img2 = img2[border:h - border, border:w - border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


# --------------------------------------------
# SSIM
# --------------------------------------------
def calculate_ssim(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    # img1 = img1.squeeze()
    # img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h - border, border:w - border]
    img2 = img2[border:h - border, border:w - border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def preprocess(image, res=512):
    image = image.resize((res, res), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)[:, :3, :, :].cuda()
    return 2.0 * image - 1.0


def encoder(image, model, res=512):
    generator = torch.Generator().manual_seed(8888)
    image = preprocess(image, res)
    gpu_generator = torch.Generator(device=image.device)
    gpu_generator.manual_seed(generator.initial_seed())
    return 0.18215 * model.vae.encode(image).latent_dist.sample(generator=gpu_generator)


@torch.no_grad()
def ddim_reverse_sample(image, prompt, model, num_inference_steps: int = 20, guidance_scale: float = 2.5,
                        res=512):
    """
            ==========================================
            ============ DDIM Inversion ==============
            ==========================================
    """
    batch_size = 1

    max_length = 77
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    text_input = model.tokenizer(
        prompt[0],
        # "",
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]

    context = [uncond_embeddings, text_embeddings]
    context = torch.cat(context)

    model.scheduler.set_timesteps(num_inference_steps)


    latents = encoder(image, model, res=res)
    timesteps = model.scheduler.timesteps.flip(0)

    all_latents = [latents]

    #  Not inverse the last step, as the alpha_bar_next will be set to 0 which is not aligned to its real value (~0.003)
    #  and this will lead to a bad result.
    for t in tqdm(timesteps[:-1], desc="DDIM_inverse"):
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]

        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)

        next_timestep = t + model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
        alpha_bar_next = model.scheduler.alphas_cumprod[next_timestep] \
            if next_timestep <= model.scheduler.config.num_train_timesteps else torch.tensor(0.0)

        "leverage reversed_x0"
        reverse_x0 = (1 / torch.sqrt(model.scheduler.alphas_cumprod[t]) * (
                latents - noise_pred * torch.sqrt(1 - model.scheduler.alphas_cumprod[t])))

        latents = reverse_x0 * torch.sqrt(alpha_bar_next) + torch.sqrt(1 - alpha_bar_next) * noise_pred

        all_latents.append(latents)

    #  all_latents[N] -> N: DDIM steps  (X_{T-1} ~ X_0)
    return latents, all_latents



def init_latent(latent, model, height, width, batch_size):
    latents = latent.expand(batch_size, model.unet.in_channels, height // 8, width // 8).to(model.device)
    return latent, latents


def diffusion_step(model, latents, context, t, guidance_scale):
    latents_input = torch.cat([latents] * 2)
    noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
    noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    return latents


def latent2image(vae, latents, args, decoder, down_features):
    if args.is_encoder == 1:
        latents = 1 / 0.18215 * latents

        latents = vae.post_quant_conv(latents)

        image = decoder(latents, down_features)
    else:
        latents = 1 / 0.18215 * latents
        image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).detach().numpy()
    image = (image * 255).astype(np.uint8)
    return image


@torch.enable_grad()
def diffattack(
        model,
        label,
        num_inference_steps: int = 20,
        guidance_scale: float = 2.5,
        image=None,
        compare=None,
        model_name=["E", "R", "D", "S"],
        save_path="",
        res=224,
        start_step=15,
        classes=None,
        iterations=30,
        verbose=True,
        topN=1,
        logger=None,
        args=None,
        mode=None,
        idx=0,
        adm=None,
):

    if args.dataset_name == "ours_try":
        from dataset_caption import ours_label as imagenet_label
    else:
        raise NotImplementedError

    label = torch.from_numpy(label).long().cuda()

    classifier, classifier_supp, classifier_supp1, classifier_supp2 = classes

    classifier = classifier.eval()
    classifier_supp = classifier_supp.eval()
    classifier_supp1 = classifier_supp1.eval()
    classifier_supp2 = classifier_supp2.eval()


    height = width = res

    test_image = image.resize((height, height), resample=Image.LANCZOS)
    test_image = np.float32(test_image) / 255.0
    test_image = test_image[:, :, :3]
    test_image[:, :, ] -= (np.float32(0.485), np.float32(0.456), np.float32(0.406))
    test_image[:, :, ] /= (np.float32(0.229), np.float32(0.224), np.float32(0.225))
    test_image = test_image.transpose((2, 0, 1))
    test_image = torch.from_numpy(test_image).unsqueeze(0)

    test_image = test_image.cuda()

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)
    denormalize = transforms.Normalize(
        mean=[-m / s for m, s in zip(mean, std)],
        std=[1 / s for s in std]
    )

    attack = torchattacks.PGD(classifier, eps=(4  / 255)/(idx+1), alpha=4 / 255, steps=10, random_start=False)
    # attack = torchattacks.FGSM(classifier, eps=(4  / 255)/(idx+1))
    attack.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    adv_images = attack(test_image, torch.tensor([0]).to(test_image.device))
    imgg = np.uint8(denormalize(adv_images).detach().cpu().numpy()[0].transpose((1, 2, 0)) * 255.)
    image = Image.fromarray(imgg)

    pred = classifier_supp(test_image)
    pred1 = classifier_supp(test_image)
    pred2 = classifier_supp1(test_image)
    pred3 = classifier_supp2(test_image)
    # pred = classifier_supp(adv_images)
    # pred1 = classifier_supp(adv_images)
    # pred2 = classifier_supp1(adv_images)
    # pred3 = classifier_supp2(adv_images)
    pred_accuracy_clean = (torch.argmax(pred, 1).detach() == label).sum().item() / len(label)
    pred_accuracy_clean1 = (torch.argmax(pred1, 1).detach() == label).sum().item() / len(label)
    pred_accuracy_clean2 = (torch.argmax(pred2, 1).detach() == label).sum().item() / len(label)
    pred_accuracy_clean3 = (torch.argmax(pred3, 1).detach() == label).sum().item() / len(label)
    logger.info("\nAccuracy on benign examples: {}%".format(pred_accuracy_clean * 100))
    logger.info("\nAccuracy on benign examples: {}%".format(pred_accuracy_clean1 * 100))
    logger.info("\nAccuracy on benign examples: {}%".format(pred_accuracy_clean2 * 100))
    logger.info("\nAccuracy on benign examples: {}%".format(pred_accuracy_clean3 * 100))

    logit = torch.nn.Softmax()(pred)
    logit1 = torch.nn.Softmax()(pred1)
    logit2 = torch.nn.Softmax()(pred2)
    logit3 = torch.nn.Softmax()(pred3)
    logger.info(f"gt_label: {label[0].item()}, pred_label: {torch.argmax(pred, 1).detach().item()}, pred_clean_logit: {logit[0, label[0]].item()}")
    logger.info(f"gt_label: {label[0].item()}, pred_label: {torch.argmax(pred1, 1).detach().item()}, pred_clean_logit: {logit1[0, label[0]].item()}")
    logger.info(f"gt_label: {label[0].item()}, pred_label: {torch.argmax(pred2, 1).detach().item()}, pred_clean_logit: {logit2[0, label[0]].item()}")
    logger.info(f"gt_label: {label[0].item()}, pred_label: {torch.argmax(pred3, 1).detach().item()}, pred_clean_logit: {logit3[0, label[0]].item()}")

    _, pred_labels = pred.topk(topN, largest=True, sorted=True)

    target_prompt = " ".join([imagenet_label.refined_Label[label.item()] for i in range(1, topN)])

    prompt = [""] * 2
    logger.info(f"prompt generate: {prompt[0]} \tlabels: {pred_labels.cpu().numpy().tolist()}")

    true_label = model.tokenizer.encode(imagenet_label.refined_Label[label.item()])
    target_label = model.tokenizer.encode(target_prompt)
    logger.info(f"decoder: {true_label}, {target_label}")

    """
            ==========================================
            ============ DDIM Inversion ==============
            ==========================================
    """
    latent, inversion_latents = ddim_reverse_sample(image, prompt, model,
                                                    num_inference_steps,
                                                    0, res=height)
    inversion_latents = inversion_latents[::-1]

    # init_prompt = [prompt[0]]
    init_prompt = [""]
    batch_size = len(init_prompt)
    latent = inversion_latents[start_step - 1]

    max_length = 77
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )

    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    text_input = model.tokenizer(
        init_prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    all_uncond_emb = []
    latent, latents = init_latent(latent, model, height, width, batch_size)

    uncond_embeddings.requires_grad_(True)

    #  The DDIM should begin from 1, as the inversion cannot access X_T but only X_{T-1}
    for ind, t in enumerate(tqdm(model.scheduler.timesteps[1 + start_step - 1:], desc="Optimize_uncond_embed")):
            all_uncond_emb.append(uncond_embeddings.detach().clone())

    """
            ==========================================
            ============ Latents Attack ==============
            ==== Details please refer to Section 3 ===
            ==========================================
    """

    batch_size = len(prompt)

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]

    context = [[torch.cat([all_uncond_emb[i]] * batch_size), text_embeddings] for i in range(len(all_uncond_emb))]
    context = [torch.cat(i) for i in context]

    original_latent = latent.clone()

    latent.requires_grad_(True)

    optimizer = optim.AdamW([latent], lr=1e-2)
    cross_entro = torch.nn.CrossEntropyLoss()
    loss_l1 = torch.nn.L1Loss()
    lpips = LPIPS(net='vgg', cache_dir="./.cache").cuda()
    init_image = preprocess(image, res)
    init_image_compare = preprocess(compare, res)



    #  load newencoder
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


    pbar = tqdm(range(iterations), desc="Iterations")
    for _, _ in enumerate(pbar):
        if mode == "Generate":
            break
        latents = torch.cat([original_latent, latent])
        for ind, t in enumerate(model.scheduler.timesteps[1 + start_step - 1:]):
            latents = diffusion_step(model, latents, context[ind], t, guidance_scale)

        B,C,H,W = init_image.shape
        init_mask = torch.ones((B,1,H,W)).to(init_image.device)


        init_out_image = model.vae.decode(1 / 0.18215 * latents)['sample'][1:] * init_mask + (
                1 - init_mask) * init_image

        lpips_loss = lpips(init_image_compare, init_out_image).mean()
        l1_loss = loss_l1(init_image_compare, init_out_image)
        l1_loss_latent = loss_l1(original_latent, latent)

        out_image = (init_out_image / 2 + 0.5).clamp(0, 1)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean, std=std)
        denormalize = transforms.Normalize(
            mean=[-m / s for m, s in zip(mean, std)],
            std=[1 / s for s in std]
        )
        out_image = normalize(out_image)

        pred = classifier(out_image) / 10

        attack_loss = - cross_entro(pred, label) * args.attack_loss_weight

        loss = attack_loss

        if mode is None:
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        if verbose:
            pbar.set_postfix_str(
                f"attack_loss: {attack_loss.item():.5f} "
                f"l1_loss: {l1_loss.item():.5f} "
                f"l1_loss_latent: {l1_loss_latent.item():.5f} "
                f"lpips_loss: {lpips_loss.item():.5f} "
                f"loss: {loss.item():.5f}")


    with torch.no_grad():
        latents = torch.cat([original_latent, latent])

        for ind, t in enumerate(model.scheduler.timesteps[1 + start_step - 1:]):
            latents = diffusion_step(model, latents, context[ind], t, guidance_scale)

    if args.is_encoder == 1:

        attack = torchattacks.PGD(classifier, eps=(4 / 255)/(idx+1), alpha=4 / 255, steps=10, random_start=False)
        # attack = torchattacks.FGSM(classifier, eps=(4 / 255)/(idx+1))
        attack.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean, std=std)
        denormalize = transforms.Normalize(
            mean=[-m / s for m, s in zip(mean, std)],
            std=[1 / s for s in std]
        )
        adv_images = attack(normalize(init_image * 0.5 + 0.5), torch.tensor([0]).to(init_image.device))
        if adm is not None:
            x_diff = adm.sdedit(denormalize(adv_images), 1).detach()
            x_diff = torch.clamp(x_diff, 0, 1)
            adv_images = normalize(x_diff)
        init_image1 = 2. * denormalize(adv_images) - 1


        newencoder = NewEncoder(**params_encoder).cuda()

        p = args.encoder_weights
        state_dict = torch.load(p)

        newencoder.load_state_dict(state_dict["encoder"], strict=False)
        newencoder.requires_grad_(False)
        newencoder = newencoder.cuda()

        decoder = NewDecoder(**params_decoder)
        decoder.load_state_dict(model.vae.decoder.state_dict(), strict=False)
        decoder.requires_grad_(False)
        decoder = decoder.cuda()

        generator = torch.Generator().manual_seed(8888)
        gpu_generator = torch.Generator(device=init_image1.device)
        gpu_generator.manual_seed(generator.initial_seed())

        down_features = newencoder(init_image1)[1]

        decode_latents = model.vae.post_quant_conv(1 / 0.1825 * latents)

        out_image = decoder(decode_latents, down_features)[1:]

    else:
        decoder = None
        down_features = None

        out_image = model.vae.decode(1 / 0.18215 * latents.detach())['sample'][1:] * init_mask + (
                1 - init_mask) * init_image
    out_image = (out_image / 2 + 0.5).clamp(0, 1)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)
    denormalize = transforms.Normalize(
        mean=[-m / s for m, s in zip(mean, std)],
        std=[1 / s for s in std]
    )

    out_image = normalize(out_image)

    imgg = np.uint8(denormalize(out_image).detach().cpu().numpy()[0].transpose((1, 2, 0)) * 255.)


    pred = classifier(out_image)
    pred_label = torch.argmax(pred, 1).detach()
    pred_accuracy = (torch.argmax(pred, 1).detach() == label).sum().item() / len(label)
    logger.info("Accuracy on adversarial examples: {}%".format(pred_accuracy * 100))

    pred1 = classifier_supp(out_image)
    pred_label1 = torch.argmax(pred1, 1).detach()
    pred_accuracy1 = (torch.argmax(pred1, 1).detach() == label).sum().item() / len(label)
    logger.info("Accuracy on adversarial examples: {}%".format(pred_accuracy1 * 100))

    pred2 = classifier_supp1(out_image)
    pred_label2 = torch.argmax(pred2, 1).detach()
    pred_accuracy2 = (torch.argmax(pred2, 1).detach() == label).sum().item() / len(label)
    logger.info("Accuracy on adversarial examples: {}%".format(pred_accuracy2 * 100))


    pred3 = classifier_supp2(out_image)
    pred_label3 = torch.argmax(pred3, 1).detach()
    pred_accuracy3 = (torch.argmax(pred3, 1).detach() == label).sum().item() / len(label)
    logger.info("Accuracy on adversarial examples: {}%".format(pred_accuracy3 * 100))

    logit = torch.nn.Softmax()(pred)
    logit1 = torch.nn.Softmax()(pred1)
    logit2 = torch.nn.Softmax()(pred2)
    logit3 = torch.nn.Softmax()(pred3)
    logger.info(f"after_pred: {pred_label}, {logit[0, pred_label[0]]}")
    logger.info(f"after_pred_supp: {pred_label1}, {logit1[0, pred_label1[0]]}")
    logger.info(f"after_pred_supp1: {pred_label2}, {logit2[0, pred_label2[0]]}")
    logger.info(f"after_pred_supp2: {pred_label3}, {logit3[0, pred_label3[0]]}")
    logger.info(f"after_true: {label}, {logit[0, label[0]]}")

    """
            ==========================================
            ============= Visualization ==============
            ==========================================
    """


    real = (init_image_compare / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    image = imgg[None]
    perturbed = image.astype(np.float32) / 255
    view_images(perturbed * 255, show=False, save_path=save_path + "_adv_image.png")

    L1 = LpDistance(1)
    L2 = LpDistance(2)
    Linf = LpDistance(float("inf"))

    logger.info("L1: {}\tL2: {}\tLinf: {}".format(L1(real, perturbed), L2(real, perturbed), Linf(real, perturbed)))


    psnrv = calculate_psnr(image[0], (real[0]*255.).astype(np.uint8), border=0)
    ssimv = calculate_ssim(image[0], (real[0]*255.).astype(np.uint8), border=0)

    logger.info(f"psnrv: {psnrv}")
    logger.info(f"ssimv: {ssimv}")

    return Image.fromarray(image[0]), pred_label, pred_label1, pred_label2, pred_label3, psnrv, ssimv

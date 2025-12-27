import pandas as pd
import torch
import torch.nn as nn
import os
from network import noise
from network.decoder import DifferenceLearner
from network.encoder import Encoder
from network.decoder import Decoder
from tabulate import tabulate
from config import training_config as cfg
from network.noise import stargan_for_test
from utils.Quality import psnr
import random
from pytorch_msssim import ssim
from random import randint
from torch import optim
from datetime import datetime
from utils.DataLoad_highpass import *
from utils.torch_utils import decoded_message_error_rate_batch
from dataloader import train_dataloader, val_dataloader
import json
from tqdm import tqdm
import warnings
import numpy as np
from utils import DTCWT_highpass

warnings.filterwarnings("ignore")

history = []
indices_encoder = torch.tensor([0, 1, 2]).to(cfg.device)
indices_decoder_d = torch.tensor([0, 1, 2]).to(cfg.device)
indices_decoder_t = torch.tensor([0, 1, 2, 3, 4, 5]).to(cfg.device)


def seed_torch(seed=42):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True



def preprocess(images):
    images_Y = images[:, [0], :, :]
    images_U = images[:, [1], :, :]
    images_V = images[:, [2], :, :]
    low_pass, high_pass = DTCWT_highpass.images_U_dtcwt_with_low(images_V)
    return images_Y, images_U, images_V, low_pass, high_pass



def lr_decay(lr, epoch, opt, module_type='all'):
    """为不同模块设置不同的学习率调度"""
    if module_type == 'encoder' or module_type == 'all':
        if epoch == 3:
            for param_group in opt.param_groups:
                param_group["lr"] = 1e-4
        elif epoch == 5:
            for param_group in opt.param_groups:
                param_group["lr"] = 5e-5
        elif epoch == 7:
            for param_group in opt.param_groups:
                param_group["lr"] = 1e-5

    elif module_type == 'decoder':
        # 解码器使用更慢的学习率下降
        if epoch == 5:
            for param_group in opt.param_groups:
                param_group["lr"] = 1e-4  # 比编码器更高的学习率
        elif epoch == 10:
            for param_group in opt.param_groups:
                param_group["lr"] = 5e-5
        elif epoch == 15:
            for param_group in opt.param_groups:
                param_group["lr"] = 1e-5
        elif epoch == 20:
            for param_group in opt.param_groups:
                param_group["lr"] = 1e-6




class IWNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder().to(cfg.device)
        self.decoder_t = Decoder(type="tracer").to(cfg.device)
        self.decoder_d = Decoder(type="detector").to(cfg.device)
        self.diff = DifferenceLearner().to(cfg.device)

    def fit(self, log_dir=False, batch_size=cfg.batch_size, lr=float(cfg.lr), epochs=cfg.epochs):
        if not log_dir:
            log_dir = f'exp_highpass/{(datetime.now().strftime("%Y.%m.%d-%H.%M.%S"))}'
        os.makedirs(log_dir, exist_ok=True)

        train = train_dataloader
        val = val_dataloader

        optimizer_encoder = optim.Adam(self.encoder.parameters(), lr=lr, weight_decay=0.00001)
        optimizer_decoder_t = optim.Adam(self.decoder_t.parameters(), lr=lr, weight_decay=0.00001)
        optimizer_decoder_d = optim.Adam(self.decoder_d.parameters(), lr=lr, weight_decay=0.00001)

        with open(os.path.join(log_dir, "config.json"), "wt") as out:
            out.write(json.dumps(cfg, indent=2, default=lambda o: str(o)))

        identity = noise.Identity()
        jpeg = noise.jpeg_compression_train
        resize = noise.Resize()
        medianblur = noise.MedianBlur()
        gau_noise = noise.GaussianNoise()
        gau_blur = noise.GaussianBlur()
        dropout_noise = noise.Dropout()
        salt_pepper_noise = noise.SaltPepper()
        brightness_noise = noise.Brightness()
        contrast_noise = noise.Contrast()
        saturation_noise = noise.Saturation()
        hue_noise = noise.Hue()
        stargan = noise.stargan_noise
        ganimation = noise.ganimation_noise
        simswap = noise.simswap_noise



        def add_noise(input, u_embedded, type):
            #print('ininininininin',input[0].size(),u_embedded.size())

            label_map = {
                0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0,
                8: 0, 9: 0, 10: 0, 11: 0, 12: 3, 13: 2, 14: 1
            }

            if type == "all":
                choice = randint(0, 14)
            elif type == "common":
                choice = randint(0, 11)
            elif type == "deepfake":
                choice = random.choice([12, 13, 14])
            else:
                choice = int(type)

            if choice == 0:
                attacked_image, noised = identity(input)
            elif choice == 1:
                attacked_image, noised = jpeg(input)
            elif choice == 2:
                attacked_image, noised = resize(input)
            elif choice == 3:
                attacked_image, noised = medianblur(input)
            elif choice == 4:
                attacked_image, noised = gau_noise(input)
            elif choice == 5:
                attacked_image, noised = gau_blur(input)
            elif choice == 6:
                attacked_image, noised = dropout_noise(input)
            elif choice == 7:
                attacked_image, noised = salt_pepper_noise(input)
            elif choice == 8:
                attacked_image, noised = brightness_noise(input)
            elif choice == 9:
                attacked_image, noised = contrast_noise(input)
            elif choice == 10:
                attacked_image, noised = saturation_noise(input)
            elif choice == 11:
                attacked_image, noised = hue_noise(input)
            elif choice == 12:
                #print('888888888888888888888888')
                attacked_image, noised = stargan(input, type)
            elif choice == 13:
                attacked_image, noised = ganimation(input, type)
            elif choice == 14:
                attacked_image, noised = simswap(input, type)

            noised_u = u_embedded + noised
            #print('aaa',choice,noised_u.size())
            label = torch.tensor([label_map[choice]] * input[0].shape[0], dtype=torch.long).to(cfg.device)
            #print(f"Choice: {choice}, Label: {label_map[choice]}")
            return noised_u, label, attacked_image

        def specific_noise(noise_fn, label, require_type=False):
            def wrapper(input, type):
                if require_type:
                    attacked_image, noised = noise_fn(input, type)
                else:
                    attacked_image, noised = noise_fn(input)
                perturb_label = torch.tensor([label] * input[0].shape[0], dtype=torch.long).to(cfg.device)
                return noised, perturb_label, attacked_image

            return wrapper
        def decode(u_embedded, decoder, indices_decoder, diff, aa):
            high_pass_extract = DTCWT_highpass.images_U_dtcwt_without_low(u_embedded)
            selected_areas_extract = torch.index_select(high_pass_extract[1], 2, indices_decoder)
            selected_areas_extract = selected_areas_extract[:, :, :, :, :, 0].squeeze(1)
            message, perturb_pred = decoder(selected_areas_extract, diff, aa)
            return message, perturb_pred

        def validation_attack(input, u_embedded, noise, decoder, indices_decoder, V, type="default"):
            noised, perturb_label, attacked_image = noise(input, type)
            difference_features, _ = self.diff(u_embedded + noised, None)
            message, perturb_pred = decode(u_embedded + noised, decoder, indices_decoder, difference_features, attacked_image)
            # if type == "deepfake":
            #     print("%nPredicted labelsvvvvvvvvvvvvvv:", torch.argmax(perturb_pred, dim=1))
            #     print("True labelsvvvvvvvvvvv:", perturb_label)
            return message, perturb_pred, perturb_label


        for epoch in range(1, epochs + 1):
            #控制PerturbationClassifier的训练
            # if epoch <= 2:
            #     for param in self.decoder_d.classifier.parameters():
            #         param.requires_grad = False
            # else:
            #     for param in self.decoder_d.classifier.parameters():
            #         param.requires_grad = True
            metrics = {
                "train_loss": [],
                "train_vis": [],
                "train_ssim": [],  # Added to track SSIM
                "train_er_all": [],
                "train_er_common": [],
                "train_er_df": [],
                "train_perturb_acc_common": [],
                "train_perturb_acc_df": [],
                "val_psnr": [],
                "val_ssim": [],
                "val_jpeg_er_all": [],
                "val_resize_er_all": [],
                "val_medianblur_er_all": [],
                "val_gaublur_er_all": [],
                "val_gauNoise_er_all": [],
                "val_dropout_er_all": [],
                "val_saltPepper_er_all": [],
                "val_identity_er_all": [],
                "val_brightness_er_all": [],
                "val_contrast_er_all": [],
                "val_saturation_er_all": [],
                "val_hue_er_all": [],
                "val_simswap_er_all": [],
                "val_stargan_er_all": [],
                "val_ganimation_er_all": [],
                "val_jpeg_er_common": [],
                "val_resize_er_common": [],
                "val_medianblur_er_common": [],
                "val_gaublur_er_common": [],
                "val_gauNoise_er_common": [],
                "val_dropout_er_common": [],
                "val_saltPepper_er_common": [],
                "val_identity_er_common": [],
                "val_brightness_er_common": [],
                "val_contrast_er_common": [],
                "val_saturation_er_common": [],
                "val_hue_er_common": [],
                "val_simswap_er_df": [],
                "val_stargan_er_df": [],
                "val_ganimation_er_df": [],
                "val_perturb_acc_cls_0": [],
                "val_perturb_acc_cls_1": [],
                "val_perturb_acc_cls_2": [],
                "val_perturb_acc_cls_3": [],
                "train_disc_loss": [],
                "train_gen_loss": []
            }
            self.encoder.train()
            self.decoder_t.train()
            self.decoder_d.train()
            #self.diff.train()
            iterator = tqdm(train)
            cur_lr = 0.0

            for step, (cover_images, mask) in enumerate(iterator):
                cover_images = cover_images.to(cfg.device)
                mask = mask.to(cfg.device)

                Y, U, V, low_pass, high_pass = preprocess(cover_images)
                # print("low_pass shape:", low_pass.shape)
                # print("low_pass shape11:", high_pass[1].shape)

                for param_group in optimizer_encoder.param_groups:
                    cur_lr = param_group["lr"]

                lr_decay(cur_lr, epoch, optimizer_encoder, 'encoder')
                lr_decay(cur_lr, epoch, optimizer_decoder_t, 'decoder')
                lr_decay(cur_lr, epoch, optimizer_decoder_d, 'decoder')
                #lr_decay(cur_lr, epoch, optimizer_diff)

                watermark = torch.Tensor(np.random.choice([-cfg.message_range, cfg.message_range],
                                                          (cover_images.shape[0], cfg.message_length))).to(cfg.device)
                selected_areas_embed = torch.index_select(high_pass[1], 2, indices_encoder)[:, :, :, :, :, 0].squeeze(1)
                high_pass[1][:, :, indices_encoder, :, :, 0] = self.encoder(selected_areas_embed, watermark).unsqueeze(
                    1)
                v_embedded = DTCWT_highpass.dtcwt_images_U(low_pass, high_pass)
                watermarked_images = torch.cat([Y, U, v_embedded], dim=1)

                forward_u_embedded = v_embedded.clone().detach()
                forward_watermarked_images = watermarked_images.clone().detach()
                forward_cover_images = cover_images.clone().detach()
                forward_mask = mask.clone().detach()

                input = [forward_u_embedded, forward_watermarked_images, forward_cover_images, forward_mask]
                #print('1231213',input[0].size(),input[3].size())

                # 正常训练阶段：使用所有噪声类型
                u_embedded_attack_type_all, perturb_label_all, _ = add_noise(input, v_embedded, type="all")
                u_embedded_attack_type_common, perturb_label_common, attacked_image_common = add_noise(input, v_embedded, type="common")
                u_embedded_attack_type_df, perturb_label_df, attacked_image_df = add_noise(input, v_embedded, type="deepfake")

                use_pred = epoch > 5
                difference_features_all, diff_loss_all = self.diff(u_embedded_attack_type_all, V, use_pred=use_pred)
                difference_features_common, diff_loss_common = self.diff(u_embedded_attack_type_common, V,
                                                                         use_pred=use_pred)
                difference_features_df, diff_loss_df = self.diff(u_embedded_attack_type_df, V, use_pred=use_pred)

                extract_wm_all, _ = decode(u_embedded_attack_type_all, self.decoder_t, indices_decoder_t,
                                           difference_features_all, _)
                extract_wm_common, perturb_pred_common = decode(u_embedded_attack_type_common, self.decoder_d,
                                                                indices_decoder_d, difference_features_common, attacked_image_common)
                extract_wm_df, perturb_pred_df = decode(u_embedded_attack_type_df, self.decoder_d, indices_decoder_d,
                                                        difference_features_df, attacked_image_df)

                # print("%nPredicted labelsttttttttt:", torch.argmax(perturb_pred_df, dim=1))
                # print("True labelsttttttttttt:", perturb_label_df)


                mse = nn.MSELoss().to(cfg.device)
                ce_loss = nn.CrossEntropyLoss().to(cfg.device)

                loss_encoder = mse(V, v_embedded)
                loss_noise_all = mse(extract_wm_all, watermark)
                loss_noise_common = mse(extract_wm_common, watermark)
                loss_noise_df = mse(extract_wm_df, torch.zeros_like(watermark))

                loss_perturb_all = 0
                loss_perturb_common = ce_loss(perturb_pred_common, perturb_label_common)
                loss_perturb_df = ce_loss(perturb_pred_df, perturb_label_df)

                ssim_value = ssim(cover_images, watermarked_images, data_range=1.0, size_average=True)
                ssim_loss = 1.0 - ssim_value
                # ssim_value = ssim(cover_images.to(torch.device('cpu')),
                #                  watermarked_images.to(torch.device('cpu')), batch_size)
                # ssim_loss = 1.0 - ssim_value

                diff_loss_total = (diff_loss_all + diff_loss_common + diff_loss_df) / 3
                loss_total = (
                        loss_encoder * cfg.encoder_w +
                        loss_noise_all * cfg.all_w +
                        loss_noise_common * cfg.common_w +
                        loss_noise_df * cfg.df_w +
                        loss_perturb_common * cfg.perturb_w +
                        loss_perturb_df * cfg.perturb_w1 +
                        diff_loss_total * cfg.diff_w +
                        ssim_loss * cfg.ssim_w  # Added SSIM loss term
                )

                optimizer_encoder.zero_grad()
                optimizer_decoder_t.zero_grad()
                optimizer_decoder_d.zero_grad()
                #optimizer_diff.zero_grad()
                loss_total.backward()
                optimizer_encoder.step()
                optimizer_decoder_t.step()
                optimizer_decoder_d.step()
                #optimizer_diff.step()

                batch_acc_common = (
                            torch.argmax(perturb_pred_common, dim=1) == perturb_label_common).float().mean().item()
                batch_acc_df = (torch.argmax(perturb_pred_df, dim=1) == perturb_label_df).float().mean().item()


                metrics["train_perturb_acc_common"].append(batch_acc_common)
                metrics["train_perturb_acc_df"].append(batch_acc_df)
                metrics["train_loss"].append(loss_total.item())
                metrics["train_vis"].append(loss_encoder.item())
                metrics["train_ssim"].append(ssim_value.item())  # Track SSIM value
                metrics["train_er_all"].append(
                    decoded_message_error_rate_batch(extract_wm_all, watermark).detach().cpu())
                metrics["train_er_common"].append(
                    decoded_message_error_rate_batch(extract_wm_common, watermark).detach().cpu())
                metrics["train_er_df"].append(decoded_message_error_rate_batch(extract_wm_df, watermark).detach().cpu())

                iterator.set_description(
                    "Epoch %s | Loss %.6f | Vis %.6f | ssim %.6f | all %.6f | common %.6f | df %.6f | P_common %.6f | P_df %.6f" % (
                        epoch,
                        np.mean(metrics["train_loss"]),
                        np.mean(metrics["train_vis"]),
                        np.mean(metrics["train_ssim"]),
                        np.mean(metrics["train_er_all"]),
                        np.mean(metrics["train_er_common"]),
                        np.mean(metrics["train_er_df"]),
                        np.mean(metrics["train_perturb_acc_common"]),
                        np.mean(metrics["train_perturb_acc_df"]),
                    )
                )

            self.encoder.eval()
            self.decoder_t.eval()
            self.decoder_d.eval()
            iterator = tqdm(val)
            with torch.no_grad():
                val_acc_cls_0 = []
                val_acc_cls_1 = []
                val_acc_cls_2 = []
                val_acc_cls_3 = []

                for step, (images, mask) in enumerate(iterator):
                    cover_images = images.to(cfg.device)
                    mask = mask.to(cfg.device)
                    Y, U, V, low_pass, high_pass = preprocess(cover_images)

                    watermark = torch.Tensor(np.random.choice([-cfg.message_range, cfg.message_range],
                                                              (cover_images.shape[0], cfg.message_length))).to(
                        cfg.device)
                    selected_areas_embed = torch.index_select(high_pass[1], 2, indices_encoder)
                    selected_areas_embed = selected_areas_embed[:, :, :, :, :, 0].squeeze(1)
                    ans = self.encoder(selected_areas_embed, watermark)
                    ans = ans.unsqueeze(1)
                    high_pass[1][:, :, indices_encoder, :, :, 0] = ans
                    v_embedded = DTCWT_highpass.dtcwt_images_U(low_pass, high_pass)
                    watermarked_images = torch.cat([Y, U, v_embedded], dim=1)

                    forward_u_embedded = v_embedded.clone().detach()

                    forward_watermarked_images = watermarked_images.clone().detach()
                    forward_cover_images = cover_images.clone().detach()
                    forward_mask = mask.clone().detach()
                    input = [forward_u_embedded, forward_watermarked_images, forward_cover_images, forward_mask]

                    cover_images = cover_images.detach().cpu()
                    embedded_images = watermarked_images.clamp(-1, 1).detach().cpu()
                    metrics["val_psnr"].append(psnr(cover_images, embedded_images, batch_size))
                    metrics["val_ssim"].append(ssim(cover_images, embedded_images, data_range=1.0, size_average=True)
)

                    for noise_type, noise_fn in [
                        ("jpeg", specific_noise(jpeg, 0)),
                        ("resize", specific_noise(resize, 0)),
                        ("medianblur", specific_noise(medianblur, 0)),
                        ("gaublur", specific_noise(gau_blur, 0)),
                        ("gauNoise", specific_noise(gau_noise, 0)),
                        ("dropout", specific_noise(dropout_noise, 0)),
                        ("saltPepper", specific_noise(salt_pepper_noise, 0)),
                        ("identity", specific_noise(identity, 0)),
                        ("brightness", specific_noise(brightness_noise, 0)),
                        ("contrast", specific_noise(contrast_noise, 0)),
                        ("saturation", specific_noise(saturation_noise, 0)),
                        ("hue", specific_noise(hue_noise, 0)),
                    ]:
                        val_wm_all, _, val_label_all = validation_attack(
                            input, v_embedded, noise_fn, self.decoder_t, indices_decoder_t, None
                        )
                        val_wm_common, val_perturb_common, val_label_common = validation_attack(
                            input, v_embedded, noise_fn, self.decoder_d, indices_decoder_d, None
                        )
                        metrics[f"val_{noise_type}_er_all"].append(
                            decoded_message_error_rate_batch(val_wm_all, watermark).detach().cpu())
                        metrics[f"val_{noise_type}_er_common"].append(
                            decoded_message_error_rate_batch(val_wm_common, watermark).detach().cpu())
                        acc_cls_0 = (torch.argmax(val_perturb_common, dim=1) == val_label_common).float().mean().item()
                        val_acc_cls_0.append(acc_cls_0)

                    val_simswap_wm_all, _, val_simswap_label_all = validation_attack(
                        input, v_embedded, specific_noise(simswap, 1, require_type=True), self.decoder_t,
                        indices_decoder_t, None, type="all"
                    )
                    val_simswap_wm_df, val_simswap_perturb_df, val_simswap_label_df = validation_attack(
                        input, v_embedded, specific_noise(simswap, 1, require_type=True), self.decoder_d,
                        indices_decoder_d, None, type="deepfake"
                    )
                    metrics["val_simswap_er_all"].append(
                        decoded_message_error_rate_batch(val_simswap_wm_all, watermark).detach().cpu())
                    metrics["val_simswap_er_df"].append(
                        decoded_message_error_rate_batch(val_simswap_wm_df, watermark).detach().cpu())
                    acc_cls_1 = (torch.argmax(val_simswap_perturb_df,
                                              dim=1) == val_simswap_label_df).float().mean().item()
                    val_acc_cls_1.append(acc_cls_1)

                    val_ganimation_wm_all, _, val_ganimation_label_all = validation_attack(
                        input, v_embedded, specific_noise(ganimation, 2, require_type=True), self.decoder_t,
                        indices_decoder_t, None, type="all"
                    )
                    val_ganimation_wm_df, val_ganimation_perturb_df, val_ganimation_label_df = validation_attack(
                        input, v_embedded, specific_noise(ganimation, 2, require_type=True), self.decoder_d,
                        indices_decoder_d, None, type="deepfake"
                    )
                    metrics["val_ganimation_er_all"].append(
                        decoded_message_error_rate_batch(val_ganimation_wm_all, watermark).detach().cpu())
                    metrics["val_ganimation_er_df"].append(
                        decoded_message_error_rate_batch(val_ganimation_wm_df, watermark).detach().cpu())
                    acc_cls_2 = (torch.argmax(val_ganimation_perturb_df,
                                              dim=1) == val_ganimation_label_df).float().mean().item()
                    val_acc_cls_2.append(acc_cls_2)

                    val_stargan_wm_all, _, val_stargan_label_all = validation_attack(
                        input, v_embedded, specific_noise(stargan, 3, require_type=True), self.decoder_t,
                        indices_decoder_t, None, type="all"
                    )
                    #print('valval1111111111111111')
                    val_stargan_wm_df, val_stargan_perturb_df, val_stargan_label_df = validation_attack(
                        input, v_embedded, specific_noise(stargan, 3, require_type=True), self.decoder_d,
                        indices_decoder_d, None, type="deepfake"
                    )
                    metrics["val_stargan_er_all"].append(
                        decoded_message_error_rate_batch(val_stargan_wm_all, watermark).detach().cpu())
                    metrics["val_stargan_er_df"].append(
                        decoded_message_error_rate_batch(val_stargan_wm_df, watermark).detach().cpu())
                    acc_cls_3 = (torch.argmax(val_stargan_perturb_df,
                                              dim=1) == val_stargan_label_df).float().mean().item()
                    val_acc_cls_3.append(acc_cls_3)

                metrics["val_perturb_acc_cls_0"].append(np.mean(val_acc_cls_0) if val_acc_cls_0 else float('nan'))
                metrics["val_perturb_acc_cls_1"].append(np.mean(val_acc_cls_1) if val_acc_cls_1 else float('nan'))
                metrics["val_perturb_acc_cls_2"].append(np.mean(val_acc_cls_2) if val_acc_cls_2 else float('nan'))
                metrics["val_perturb_acc_cls_3"].append(np.mean(val_acc_cls_3) if val_acc_cls_3 else float('nan'))

                print(f"val-epoch-{epoch}: %n")
                data_vis = [
                    ["PSNR", "SSIM"],
                    [np.mean(metrics["val_psnr"]) if metrics["val_psnr"] else "NaN",
                     np.mean(metrics["val_ssim"]) if metrics["val_ssim"] else "NaN"],
                ]
                data_err = [
                    ["Attack", "All(er)", "Common(er)", "DF(er)", "Perturb_acc"],
                    ["Jpeg", np.mean(metrics["val_jpeg_er_all"]) if metrics["val_jpeg_er_all"] else "NaN",
                     np.mean(metrics["val_jpeg_er_common"]) if metrics["val_jpeg_er_common"] else "NaN",
                     "-", "-"],
                    ["Resize", np.mean(metrics["val_resize_er_all"]) if metrics["val_resize_er_all"] else "NaN",
                     np.mean(metrics["val_resize_er_common"]) if metrics["val_resize_er_common"] else "NaN",
                     "-", "-"],
                    ["MedianBlur",
                     np.mean(metrics["val_medianblur_er_all"]) if metrics["val_medianblur_er_all"] else "NaN",
                     np.mean(metrics["val_medianblur_er_common"]) if metrics["val_medianblur_er_common"] else "NaN",
                     "-", "-"],
                    ["Gau_blur", np.mean(metrics["val_gaublur_er_all"]) if metrics["val_gaublur_er_all"] else "NaN",
                     np.mean(metrics["val_gaublur_er_common"]) if metrics["val_gaublur_er_common"] else "NaN",
                     "-", "-"],
                    ["Gau_noise", np.mean(metrics["val_gauNoise_er_all"]) if metrics["val_gauNoise_er_all"] else "NaN",
                     np.mean(metrics["val_gauNoise_er_common"]) if metrics["val_gauNoise_er_common"] else "NaN",
                     "-", "-"],
                    ["Dropout", np.mean(metrics["val_dropout_er_all"]) if metrics["val_dropout_er_all"] else "NaN",
                     np.mean(metrics["val_dropout_er_common"]) if metrics["val_dropout_er_common"] else "NaN",
                     "-", "-"],
                    ["SaltPepper",
                     np.mean(metrics["val_saltPepper_er_all"]) if metrics["val_saltPepper_er_all"] else "NaN",
                     np.mean(metrics["val_saltPepper_er_common"]) if metrics["val_saltPepper_er_common"] else "NaN",
                     "-", "-"],
                    ["Identity", np.mean(metrics["val_identity_er_all"]) if metrics["val_identity_er_all"] else "NaN",
                     np.mean(metrics["val_identity_er_common"]) if metrics["val_identity_er_common"] else "NaN",
                     "-", "-"],
                    ["Brightness",
                     np.mean(metrics["val_brightness_er_all"]) if metrics["val_brightness_er_all"] else "NaN",
                     np.mean(metrics["val_brightness_er_common"]) if metrics["val_brightness_er_common"] else "NaN",
                     "-", "-"],

                    ["Contrast",
                     np.mean(metrics["val_contrast_er_all"]) if metrics["val_contrast_er_all"] else "NaN",
                     np.mean(metrics["val_contrast_er_common"]) if metrics["val_contrast_er_common"] else "NaN",
                     "-", "-"],

                    ["Saturation",
                     np.mean(metrics["val_saturation_er_all"]) if metrics["val_saturation_er_all"] else "NaN",
                     np.mean(metrics["val_saturation_er_common"]) if metrics["val_saturation_er_common"] else "NaN",
                     "-", "-"],

                    ["Hue",
                     np.mean(metrics["val_hue_er_all"]) if metrics["val_hue_er_all"] else "NaN",
                     np.mean(metrics["val_hue_er_common"]) if metrics["val_hue_er_common"] else "NaN",
                     "-", "-"],

                    ["Simswap", np.mean(metrics["val_simswap_er_all"]) if metrics["val_simswap_er_all"] else "NaN",
                     "-", np.mean(metrics["val_simswap_er_df"]) if metrics["val_simswap_er_df"] else "NaN",
                     np.mean(metrics["val_perturb_acc_cls_1"]) if metrics["val_perturb_acc_cls_1"] else "NaN"],
                    ["StarGan", np.mean(metrics["val_stargan_er_all"]) if metrics["val_stargan_er_all"] else "NaN",
                     "-", np.mean(metrics["val_stargan_er_df"]) if metrics["val_stargan_er_df"] else "NaN",
                     np.mean(metrics["val_perturb_acc_cls_3"]) if metrics["val_perturb_acc_cls_3"] else "NaN"],
                    ["Ganimation",
                     np.mean(metrics["val_ganimation_er_all"]) if metrics["val_ganimation_er_all"] else "NaN",
                     "-", np.mean(metrics["val_ganimation_er_df"]) if metrics["val_ganimation_er_df"] else "NaN",
                     np.mean(metrics["val_perturb_acc_cls_2"]) if metrics["val_perturb_acc_cls_2"] else "NaN"],
                    ["Class-wise Acc", "-", "-", "-",
                     f"Cls0: {np.mean(metrics['val_perturb_acc_cls_0']) if metrics['val_perturb_acc_cls_0'] else 'NaN'}, "
                     f"Cls1: {np.mean(metrics['val_perturb_acc_cls_1']) if metrics['val_perturb_acc_cls_1'] else 'NaN'}, "
                     f"Cls2: {np.mean(metrics['val_perturb_acc_cls_2']) if metrics['val_perturb_acc_cls_2'] else 'NaN'}, "
                     f"Cls3: {np.mean(metrics['val_perturb_acc_cls_3']) if metrics['val_perturb_acc_cls_3'] else 'NaN'}"]
                ]
                table_str = tabulate(data_vis, headers="firstrow", tablefmt="grid")
                print(table_str)
                with open(os.path.join(log_dir, "metrics_table_visual.json"), "at") as file0:
                    print(table_str, file=file0)

                table_str2 = tabulate(data_err, headers="firstrow", tablefmt="grid")
                print(table_str2)
                with open(os.path.join(log_dir, "metrics_table_err.json"), "at") as file1:
                    print(table_str2, file=file1)

            metrics = {
                k: round(np.mean(v), 7) if len(v) > 0 else "NaN"
                for k, v in metrics.items()
            }
            metrics["epoch"] = epoch
            metrics["LR"] = cur_lr
            history.append(metrics)
            pd.DataFrame(history).to_csv(os.path.join(log_dir, "metrics.tsv"), index=False, sep="\t")
            with open(os.path.join(log_dir, "metrics.json"), "at") as out:
                out.write(json.dumps(metrics, indent=2, default=lambda o: str(o)))
            torch.save(self, os.path.join(log_dir, f"model_{epoch}.pth"))
            torch.save(self.state_dict(), os.path.join(log_dir, f"model_state_{epoch}.pth"))
        return history


if __name__ == "__main__":
    seed_torch(42)
    model = IWNet()
    model.fit()
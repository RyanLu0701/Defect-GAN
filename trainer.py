from model.model import Generator,Discriminator
from model.model import SPADE
import torch 
import torch.nn.functional as F
import numpy as np
import os
import time
from tqdm import tqdm
from torchvision.utils import save_image
from torch.cuda.amp import GradScaler, autocast
from model.model import AdaptiveNoiseLayer


def gradient_penalty(y, x, device):
    weight= torch.ones(y.size()).to(device)
    dydx = torch.autograd.grad(outputs=y,
                               inputs=x ,
                               grad_outputs=weight,
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True)[0]
    
    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.norm(dydx, p=2, dim=1)

    return torch.mean((dydx_l2norm - 1)**2)

def denormalized(data: torch.Tensor, to_uint8: bool = False) -> torch.Tensor:
    """
    將數據從 [-1, 1] 反正規化回 [0, 1] 或 [0, 255]。

    Args:
        data (torch.Tensor): 正規化到 [-1, 1] 的張量
        to_uint8 (bool): 是否轉換為 [0, 255] 並轉為 uint8 格式，預設為 False

    Returns:
        torch.Tensor: 反正規化後的張量
    """
    # 限制範圍，避免值超出 [-1, 1]
    data = torch.clamp(data, -1, 1).detach().cpu()
    
    # 反正規化回 [0, 1]
    data = (data + 1) / 2

    # 如果需要轉換為 [0, 255] 並轉為 uint8
    if to_uint8:
        data = (data * 255).byte()

    return data


def generate_sample_images(generator, dataloader, epoch, device, sample_path="samples"):
    os.makedirs(sample_path, exist_ok=True)

    generator.eval()

    # 遍歷生成器中的所有 AdaptiveNoiseLayer 並禁用噪聲
    for module in generator.modules():
        print(module)
        if isinstance(module, AdaptiveNoiseLayer):
            module.training = False

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            real_images = data["img"].to(device)
            segmap = data["label"].to(device)
            generated_restore_images, generated_defect_images, defect_spatial_map, restore_spatail_map = generator(real_images, segmap)

            # 反正規化圖片
            generated_restore_images = denormalized(generated_restore_images)
            generated_defect_images = denormalized(generated_defect_images)

            # 儲存圖片
            for j in range(min(2, len(real_images))):
                save_image(segmap[j], os.path.join(sample_path, f"epoch_{epoch}_segmap_{j}.png"))

                save_image(generated_defect_images[j],
                           os.path.join(sample_path, f"epoch_{epoch}_defect_sample_{j}.png"))
                save_image(generated_restore_images[j],
                           os.path.join(sample_path, f"epoch_{epoch}_restore_sample_{j}.png"))
                save_image(defect_spatial_map[j],
                           os.path.join(sample_path, f"epoch_{epoch}_defect_spatial_map_{j}.png"))
                save_image(restore_spatail_map[j],
                           os.path.join(sample_path, f"epoch_{epoch}_restore_spatail_map_{j}.png"))
            break

    # 恢復 AdaptiveNoiseLayer 的 training 屬性
    for module in generator.modules():
        if isinstance(module, AdaptiveNoiseLayer):
            module.training = True

    generator.train()
    print(f"Sample images generated for epoch {epoch}")


def train(args, discriminator, generator, dataloader, D_optimizer, G_optimizer, g_scheduler, d_scheduler):
    scaler = GradScaler()  # 初始化 GradScaler

    for epoch in range(args.epoch + 1):
        # 初始化損失累加器
        total_d_loss = 0.0
        total_g_loss = 0.0
        num_batches = len(dataloader)
        
        generator.train()
        discriminator.train()
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
        
        for i, data in progress_bar:
            x_real = data["img"].float().to(args.device)
            segmap = data["label"].to(args.device)

            # 判別器損失計算
            with autocast():  # 開始混合精度區域
                out_src, out_cls = discriminator(x_real)
                d_loss_real = -torch.mean(out_src)
                D_x        =  out_cls.mean().item()
                d_loss_cls = F.binary_cross_entropy_with_logits(out_cls, segmap)

                x_fake, _, de_spatial_map, re_spatial_map = generator(x_real, segmap)

                out_src, out_cls = discriminator(x_fake.detach())
                d_loss_fake = torch.mean(out_src)
                D_G_z1 = d_loss_fake.mean().item()

                alpha = torch.rand(x_real.size(0), 1, 1, 1).to(args.device).float()
                x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
                out_src, _ = discriminator(x_hat)


                d_loss_gp = gradient_penalty(out_src, x_hat, args.device)

                d_loss = d_loss_real + d_loss_fake + args.lambda_gp * d_loss_gp + args.lambda_real_cls * d_loss_cls

            total_d_loss += d_loss.item()  # 累加判別器損失

            discriminator.zero_grad()
            scaler.scale(d_loss).backward()  # 使用 GradScaler 縮放損失
            scaler.step(D_optimizer)  # 更新權重
            scaler.update()  # 更新 GradScaler

            if i % args.gen_update_time == 0:
                with autocast():  # 混合精度處理生成器部分
                    x_fake, _, de_spatial_map, re_spatial_map = generator(x_real, segmap)
                    out_src, out_cls = discriminator(x_fake)
                    g_adv_loss = -torch.mean(out_src)
                    g_cls_loss = F.binary_cross_entropy_with_logits(out_cls, segmap)
                    D_G_z2 = out_src.mean().item()

                    reconstruction_loss = F.l1_loss(x_fake, x_real)
                    spatial_constraint_loss = F.l1_loss(de_spatial_map, re_spatial_map)
                    region_constraint_loss = F.l1_loss(de_spatial_map, torch.zeros_like(de_spatial_map)) + \
                                             F.l1_loss(re_spatial_map, torch.zeros_like(re_spatial_map))

                    g_loss = g_adv_loss + args.lambda_fake_cls * g_cls_loss + args.lambda_rec * reconstruction_loss + \
                             args.lambda_cyc * spatial_constraint_loss + args.lambda_con * region_constraint_loss
                
                total_g_loss += g_loss.item()  # 累加生成器損失

                generator.zero_grad()
                scaler.scale(g_loss).backward()  # 使用 GradScaler 縮放損失
                scaler.step(G_optimizer)  # 更新權重
                scaler.update()  # 更新 GradScaler


            progress_bar.set_description(f"[{epoch + 1}/{args.epoch}][{i + 1}/{len(dataloader)}] "
                            f"Loss_D: {d_loss.item():.4f} Loss_G: {g_loss.item():.4f} "
                            f"D(x): {D_x:.6f} D(G(z)): {D_G_z1:.6f}/{D_G_z2:.6f}")

    # 計算每個 epoch 的平均損失
        avg_d_loss = total_d_loss / num_batches
        avg_g_loss = total_g_loss / num_batches

        # 更新學習率調度器
        d_scheduler.step(avg_d_loss)
        g_scheduler.step(avg_g_loss)
      
        if epoch % 5 == 0:
            torch.save(generator.state_dict(), f"save_model/generator_epoch_{epoch}.pt")
            generate_sample_images(generator=generator, dataloader=dataloader, epoch=epoch, device=args.device)
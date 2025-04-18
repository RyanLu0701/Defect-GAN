import argparse
import os
import torch
from PIL import Image
from torchvision import transforms
from model.model import Generator

def load_image(path, size):
    image = Image.open(path).convert('RGB').resize(size)
    return transforms.ToTensor()(image)

def load_mask(path, size):
    mask = Image.open(path).convert('L').resize(size)
    return transforms.ToTensor()(mask)

@torch.no_grad()
def inference(generator, image_tensor, mask_tensor, device):
    image_tensor = image_tensor.unsqueeze(0).to(device)  # (1, 3, H, W)
    mask_tensor = mask_tensor.unsqueeze(0).to(device)    # (1, 1, H, W)
    _, synthesized_defect_sample, _, _ = generator(image_tensor, mask_tensor)
    return synthesized_defect_sample.squeeze(0).cpu()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Defect-GAN Inference")

    parser.add_argument("--checkpoint",      type=str,   required=True,                     help="Path to Generator model checkpoint (.pth)")
    parser.add_argument("--input_dir",       type=str,   required=True,                     help="Directory containing input images and masks")
    parser.add_argument("--output_dir",      type=str,   required=True,                     help="Directory to save synthesized defect images")
    parser.add_argument("--im_channels",     type=int,   default=3,                         help="Input image channel count")
    parser.add_argument("--num_classes",     type=int,   default=1,                         help="Number of mask channels (1 for grayscale)")
    parser.add_argument("--num_layers",      type=int,   default=3,                         help="Number of encoder/decoder layers")
    parser.add_argument("--base_channels",   type=int,   default=128,                       help="Base channel size")
    parser.add_argument("--repeat_num",      type=int,   default=6,                         help="Residual block repeat count")
    parser.add_argument("--device",          type=str,   default='cuda',                    help="Device: 'cuda' or 'cpu'")
    parser.add_argument("--size",            type=int,   nargs=2, default=(128, 128),       help="Resize dimensions (H W)")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    G = Generator(curr_dim=args.im_channels,
                  label_nc=args.num_classes,
                  num_layers=args.num_layers,
                  base_channels=args.base_channels,
                  repeat_num=args.repeat_num,
                  training=False).to(args.device)
    G.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    G.eval()

    input_files = [f for f in os.listdir(args.input_dir) if f.endswith('.png') and not f.endswith('_mask.png')]

    for img_file in input_files:
        img_path = os.path.join(args.input_dir, img_file)
        mask_path = os.path.join(args.input_dir, img_file.replace('.png', '_mask.png'))

        if not os.path.exists(mask_path):
            print(f"[警告] 找不到對應的 mask: {mask_path}，跳過")
            continue

        image = load_image(img_path, args.size)
        mask = load_mask(mask_path, args.size)

        synthesized_defect = inference(G, image, mask, args.device)

        out_path = os.path.join(args.output_dir, img_file)
        transforms.ToPILImage()(synthesized_defect).save(out_path)

        print(f"[輸出完成] {out_path}")

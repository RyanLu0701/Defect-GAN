import os
import random
import torch
import torchvision
from PIL import Image
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F


def check_nc(img):
    """
    確保圖片為3通道。如果為灰階或RGBA格式，則進行相應轉換。
    """
    img = transforms.ToTensor()(img)
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    elif img.shape[0] == 4:  # RGBA
        img = img[:3, :, :]
    img = transforms.ToPILImage()(img)
    return img


class MyOwnData(Dataset):
    def __init__(self, im_path, num_classes, im_size=256, im_channels=3, im_ext="jpg"):
        self.num_classes = num_classes
        self.im_size = im_size  # 圖片尺寸
        self.im_channels = im_channels  # 圖片通道數
        self.im_ext = im_ext  # 圖片格式
        self.im_path = im_path  # 圖片路徑

        self.normal_images = []  # 正常圖片清單
        self.defected_images = []  # 瑕疵圖片清單
        self.masks = []  # 瑕疵 Mask 清單

        self.load_images_and_masks(im_path)

    def load_images_and_masks(self, im_path):
        """
        根據 Mask 分類正常與瑕疵圖片，並隨機打亂 Mask。
        """
        img_paths = [
            f"{im_path}/{i}" for i in os.listdir(im_path) if "label" not in i and "bmp" in i
        ]
        mask_paths = [f"{im_path}/{i}" for i in os.listdir(im_path) if "label" in i]

        transform = transforms.Compose([
            transforms.Resize((self.im_size, self.im_size), interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])

        for img_path, mask_path in zip(img_paths, mask_paths):
            mask = Image.open(mask_path)
            mask_tensor = transform(mask)

            unique_values = torch.unique(mask_tensor)
            if len(unique_values) == 1 and unique_values[0].item() == 1.0:  # 全部為255，代表背景
                self.normal_images.append(img_path)
            else:
                self.defected_images.append(img_path)
                self.masks.append(mask_path)


        print(f"找到 {len(self.normal_images)} 張正常圖片.")
        print(f"找到 {len(self.defected_images)} 張瑕疵圖片.")
        print(f"找到 {len(self.masks)} 張瑕疵 Mask.")

    def get_label(self, mask_path):
        """
        將 Mask 轉為 One-Hot 編碼。
        """
        mask_im = Image.open(mask_path)
        transform = transforms.Compose([
            transforms.Resize((self.im_size, self.im_size), interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])
        control_map_tensor = transform(mask_im)

        control_map_tensor = (control_map_tensor * 255).long()  # 縮放到 0-255
        invalid_idx = control_map_tensor == 255
        control_map_tensor[invalid_idx] = self.num_classes

        # One-hot 編碼
        one_hot_map = F.one_hot(
            control_map_tensor.squeeze(0), num_classes=self.num_classes + 1
        )
        one_hot_map = one_hot_map.permute(2, 0, 1).float()  # 重排為 [C, H, W]
        one_hot_map = one_hot_map[:self.num_classes, :, :]
        return one_hot_map

    def __len__(self):
        return max(len(self.normal_images), len(self.defected_images))

    def __getitem__(self, index):
        """
        根據 index 返回圖片與對應的隨機 Mask。
        """
        if index < len(self.normal_images):
            img_path = self.normal_images[index]
            mask_path = random.choice(self.masks)  # 隨機選擇瑕疵 Mask
        else:
            img_path = self.defected_images[index - len(self.normal_images)]
            mask_path = self.masks[index - len(self.normal_images)]

        # 處理圖片
        img = Image.open(img_path)
        img = check_nc(img)

        transform = transforms.Compose([
            transforms.Resize((self.im_size, self.im_size), interpolation=Image.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 正規化到 [-1, 1]
        ])
        img_tensor = transform(img)

        # 處理 Mask
        one_hot_map = self.get_label(mask_path)

        return {
            "img": img_tensor,
            "label": one_hot_map
        }



class ViSAData(Dataset):
    def __init__(self, im_path, num_classes, im_size=256, im_channels=3):
        self.num_classes = num_classes
        self.im_size = im_size  # 圖片尺寸
        self.im_channels = im_channels  # 圖片通道數
        self.im_path = im_path  # 圖片路徑

        self.normal_images = []  # 正常圖片清單
        self.defected_images = []  # 瑕疵圖片清單
        self.masks = []  # 瑕疵 Mask 清單

        self.load_images_and_masks(im_path)

    def load_images_and_masks(self, im_path):
        """
        根據 Mask 分類正常與瑕疵圖片，並隨機打亂 Mask。
        """
        img_paths = [
            os.path.join(im_path, i) for i in os.listdir(im_path) if i.endswith(".JPG")
        ]
        mask_paths = [os.path.join(im_path, i) for i in os.listdir(im_path) if i.endswith(".png")]
        
        for img_path in img_paths:
            if img_path.split("\\")[-1].split(".")[0] in [os.path.basename(p).split(".")[0] for p in mask_paths]:
                self.defected_images.append(img_path)
            else:
                self.normal_images.append(img_path)

        self.masks = mask_paths



        print(f"找到 {len(self.normal_images)} 張正常圖片.")
        print(f"找到 {len(self.defected_images)} 張瑕疵圖片.")
        print(f"找到 {len(self.masks)} 張瑕疵 Mask.")

    def get_label(self, mask_path):
        """
        將 Mask 轉為 One-Hot 編碼。
        """
        mask_im = Image.open(mask_path)
        transform = transforms.Compose([
            transforms.Resize((self.im_size, self.im_size), interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])
        control_map_tensor = transform(mask_im)

        # 縮放到 0-255
        control_map_tensor = (control_map_tensor * 255).long()

        # 設定背景為 num_classes
        invalid_idx = control_map_tensor == 255
        control_map_tensor[invalid_idx] = self.num_classes

        # One-hot 編碼
        one_hot_map = F.one_hot(
            control_map_tensor.squeeze(0), num_classes=self.num_classes + 1
        )
        one_hot_map = one_hot_map.permute(2, 0, 1).float()  # 重排為 [C, H, W]
        one_hot_map = one_hot_map[:self.num_classes, :, :]
        return one_hot_map

    def __len__(self):
        # 返回正常圖片與瑕疵圖片總數
        return len(self.normal_images) + len(self.defected_images)

    def __getitem__(self, index):

        if index < len(self.normal_images):
            # 正常圖片部分
            img_path = self.normal_images[index]
            mask_path = random.choice(self.masks)  # 隨機選擇瑕疵 Mask
        else:
            # 瑕疵圖片部分
            defected_index = index - len(self.normal_images)
            img_path = self.defected_images[defected_index]
            mask_path = self.masks[defected_index]  # 對應瑕疵的 Mask

        # 處理圖片
        img = Image.open(img_path)
        img = check_nc(img)

        transform = transforms.Compose([
            transforms.Resize((self.im_size, self.im_size), interpolation=Image.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 正規化到 [-1, 1]

        ])
        img_tensor = transform(img)

        # 處理 Mask
        one_hot_map = self.get_label(mask_path)

        return {
            "img": img_tensor,
            "label": one_hot_map
        }

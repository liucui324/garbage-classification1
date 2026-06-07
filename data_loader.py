import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class TrashDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for class_idx in range(40):
            class_dir = os.path.join(data_dir, str(class_idx))
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith('.jpg') or img_name.endswith('.jpeg') or img_name.endswith('.png'):
                        img_path = os.path.join(class_dir, img_name)
                        self.image_paths.append(img_path)
                        self.labels.append(class_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

def get_data_loaders(train_dir, test_dir, batch_size=32, image_size=224):
    label_names = {
        0: "其他垃圾/一次性快餐盒", 1: "其他垃圾/污损塑料", 2: "其他垃圾/烟蒂",
        3: "其他垃圾/牙签", 4: "其他垃圾/破碎花盆及碟碗", 5: "其他垃圾/竹筷",
        6: "厨余垃圾/剩饭剩菜", 7: "厨余垃圾/大骨头", 8: "厨余垃圾/水果果皮",
        9: "厨余垃圾/水果果肉", 10: "厨余垃圾/茶叶渣", 11: "厨余垃圾/菜叶菜根",
        12: "厨余垃圾/蛋壳", 13: "厨余垃圾/鱼骨",
        14: "可回收物/充电宝", 15: "可回收物/包", 16: "可回收物/化妆品瓶",
        17: "可回收物/塑料玩具", 18: "可回收物/塑料碗盆", 19: "可回收物/塑料衣架",
        20: "可回收物/快递纸袋", 21: "可回收物/插头电线", 22: "可回收物/旧衣服",
        23: "可回收物/易拉罐", 24: "可回收物/枕头", 25: "可回收物/毛绒玩具",
        26: "可回收物/洗发水瓶", 27: "可回收物/玻璃杯", 28: "可回收物/皮鞋",
        29: "可回收物/砧板", 30: "可回收物/纸板箱", 31: "可回收物/调料瓶",
        32: "可回收物/酒瓶", 33: "可回收物/金属食品罐", 34: "可回收物/锅",
        35: "可回收物/食用油桶", 36: "可回收物/饮料瓶",
        37: "有害垃圾/干电池", 38: "有害垃圾/软膏", 39: "有害垃圾/过期药物"
    }

    category_names = {
        "其他垃圾": ["一次性快餐盒", "污损塑料", "烟蒂", "牙签", "破碎花盆及碟碗", "竹筷"],
        "厨余垃圾": ["剩饭剩菜", "大骨头", "水果果皮", "水果果肉", "茶叶渣", "菜叶菜根", "蛋壳", "鱼骨"],
        "可回收物": ["充电宝", "包", "化妆品瓶", "塑料玩具", "塑料碗盆", "塑料衣架", "快递纸袋",
                   "插头电线", "旧衣服", "易拉罐", "枕头", "毛绒玩具", "洗发水瓶", "玻璃杯",
                   "皮鞋", "砧板", "纸板箱", "调料瓶", "酒瓶", "金属食品罐", "锅", "食用油桶", "饮料瓶"],
        "有害垃圾": ["干电池", "软膏", "过期药物"]
    }

    category_colors = {
        "其他垃圾": "#808080",
        "厨余垃圾": "#4CAF50",
        "可回收物": "#2196F3",
        "有害垃圾": "#F44336"
    }

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = TrashDataset(train_dir, transform=train_transform)
    test_dataset = TrashDataset(test_dir, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return {
        'train_loader': train_loader,
        'test_loader': test_loader,
        'train_size': len(train_dataset),
        'test_size': len(test_dataset),
        'label_names': label_names,
        'category_names': category_names,
        'category_colors': category_colors,
        'num_classes': 40
    }

def get_category_from_class(class_idx):
    if class_idx <= 5:
        return "其他垃圾"
    elif class_idx <= 13:
        return "厨余垃圾"
    elif class_idx <= 36:
        return "可回收物"
    else:
        return "有害垃圾"

if __name__ == "__main__":
    train_dir = r"C:\Users\Admin\Desktop\python\垃圾分类\trash40\images\train"
    test_dir = r"C:\Users\Admin\Desktop\python\垃圾分类\trash40\images\test"

    data = get_data_loaders(train_dir, test_dir, batch_size=32)

    print(f"训练集样本数: {data['train_size']}")
    print(f"测试集样本数: {data['test_size']}")
    print(f"类别数量: {data['num_classes']}")
    print(f"训练批次数: {len(data['train_loader'])}")
    print(f"测试批次数: {len(data['test_loader'])}")
    print("\n类别名称示例:")
    for i in range(5):
        print(f"  {i}: {data['label_names'][i]}")
    print("  ...")
    for i in range(37, 40):
        print(f"  {i}: {data['label_names'][i]}")

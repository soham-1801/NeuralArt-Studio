from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms


class ImageFolderDataset(Dataset):
    def __init__(self, root, transform=None):
        super(ImageFolderDataset, self).__init__()
        self.root = root
        self.transform = transform
        if not os.path.isdir(root):
            self.files = []
        else:
            # FIX #16: Use os.walk() for recursive scan across all subdirectories.
            # os.listdir() only found images in the top-level folder —
            # large datasets (MS-COCO, WikiArt) are always organized in subfolders.
            self.files = [
                os.path.join(dirpath, fname)
                for dirpath, _, fnames in os.walk(root)
                for fname in fnames
                if fname.lower().endswith(('.jpg', '.png', '.jpeg'))
            ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # FIX #16: self.files now stores full absolute paths (not just basenames)
        image_path = self.files[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image


def get_transform(size, crop, final_size):
    transform_list = []
    if size > 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.RandomCrop(final_size))
    else:
        transform_list.append(transforms.Resize(final_size))

    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)
        

def adaptive_instance_normalization(content_feat, style_feat):
    # [batch size, channels, h, w]
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_content_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_content_feat * style_std.expand(size) + style_mean.expand(size)

def calc_mean_std(feat, eps=1e-5):
    # [batch size, channels, h, w]
    size = feat.size()
    if len(size) != 4:
        raise ValueError(
            f"calc_mean_std expects a 4D tensor [batch, channels, H, W], got shape {tuple(size)}"
        )
    batch_size, channels = size[:2]
    feat_mean = feat.view(batch_size, channels, -1).mean(dim=2).view(batch_size, channels, 1, 1)
    feat_var = feat.view(batch_size, channels, -1).var(dim=2, unbiased=False) + eps
    feat_std = feat_var.sqrt().view(batch_size, channels, 1, 1)
    return feat_mean, feat_std
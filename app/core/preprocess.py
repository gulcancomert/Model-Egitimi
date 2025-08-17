from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def build_infer_transform(image_size=224, normalize=True, use_center_crop=True):
    tfms = []
    if use_center_crop:
        tfms += [
            T.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
        ]
    else:
        tfms += [T.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC)]
    tfms += [T.ToTensor()]
    if normalize:
        tfms += [T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    return T.Compose(tfms)

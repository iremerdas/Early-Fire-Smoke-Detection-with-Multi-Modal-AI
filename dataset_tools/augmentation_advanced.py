import albumentations as A
from albumentations.pytorch import ToTensorV2

bbox_params = A.BboxParams(
    format='yolo',
    label_fields=['class_labels'],
    min_visibility=0.0,
    min_area=0.0,
    check_each_transform=False
)

additional_targets = {
    'red_ratio': 'mask',
    'flow':      'mask',
    'mhi':       'mask'
}

# --- Fire ---
fire_transform = A.Compose([
    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=50, val_shift_limit=50, p=0.8),
    A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.7),
    A.RandomShadow(p=0.2),
    A.RandomSunFlare(p=0.1),
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=10, p=0.1),
    A.CoarseDropout(max_holes=8, max_height=60, max_width=60, min_holes=1, min_height=20, min_width=20, p=0.3),
    A.MotionBlur(blur_limit=7, p=0.2),
    #A.GridMask(num_grid=(2, 4), p=0.1),  # GridMask için albumentations >=1.3.0 gerekir
    A.Normalize(),
    ToTensorV2(),
], bbox_params=bbox_params, additional_targets=additional_targets)

# --- Smoke ---
smoke_transform = A.Compose([
    A.GaussianBlur(blur_limit=(5,15), p=0.7),
    A.GaussNoise(mean=0, var_limit=(10.0, 50.0), p=0.5),
    A.RandomFog(fog_coef_limit=(0.1, 0.3), p=0.5),
    A.RandomRain(blur_value=3, p=0.2),
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=10, p=0.1),
    A.MotionBlur(blur_limit=5, p=0.2),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    #A.GridMask(num_grid=(2, 4), p=0.1),
    A.Normalize(),
    ToTensorV2(),
], bbox_params=bbox_params, additional_targets=additional_targets)

# --- Both (Fire + Smoke) ---
both_transform = A.Compose([
    A.OneOf([
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=50, val_shift_limit=50, p=0.7),
        A.GaussianBlur(blur_limit=(7,15), p=0.7)
    ], p=0.9),
    A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.7),
    A.MotionBlur(blur_limit=7, p=0.4),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
    A.RandomFog(fog_coef_limit=(0.1, 0.3), p=0.4),
    A.RandomRain(blur_value=3, p=0.2),
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=10, p=0.2),
    A.RandomShadow(p=0.2),
    A.RandomSunFlare(p=0.1),
    A.CoarseDropout(max_holes=10, max_height=60, max_width=60, p=0.5),
    A.Cutout(num_holes=8, max_h_size=32, max_w_size=32, fill_value=0, p=0.5),
    A.RandomAffine(scale=(0.8, 1.2), translate_percent=0.1, rotate=15, shear=10, p=0.3),
    #A.GridMask(num_grid=(2, 4), p=0.2),
    A.Normalize(),
    ToTensorV2(),
], bbox_params=bbox_params, additional_targets=additional_targets)

# --- None (Negative) ---
none_transform = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussianBlur(blur_limit=7, p=0.4),
    A.GaussNoise(mean=0, var_limit=(5.0, 20.0), p=0.3),
    A.HorizontalFlip(p=0.5),
    A.RandomFog(fog_coef_limit=(0.05, 0.15), p=0.2),
    A.RandomShadow(p=0.1),
    #A.GridMask(num_grid=(2, 4), p=0.1),
    A.Normalize(),
    ToTensorV2(),
], bbox_params=bbox_params, additional_targets=additional_targets)

# --- Sınıfa Göre Transform Seçimi ---
def get_transform_for_labels(labels):
    """
    labels: list veya tensor şeklinde sınıf ID'leri 
      0 -> smoke, 1 -> fire, 2 -> none
    Returns: ilgili Compose objesi
    """
    unique = set(labels)
    if unique == {1}:        # only fire
        return fire_transform
    elif unique == {0}:      # only smoke
        return smoke_transform
    elif unique == {0, 1}:   # both
        return both_transform
    else:                    # none veya diğer
        return none_transform

# --- MixUp ve Mosaic için altyapı şablonu (batch bazlı uygulanır) ---
def mixup(image1, label1, image2, label2, alpha=0.4):
    """İki görüntü ve etiketi MixUp ile karıştırır."""
    import numpy as np
    lam = np.random.beta(alpha, alpha)
    mixed_image = lam * image1 + (1 - lam) * image2
    mixed_label = lam * label1 + (1 - lam) * label2
    return mixed_image.astype(image1.dtype), mixed_label

# Mosaic için de benzer şekilde 4 görüntü birleştirilebilir.
# Bunlar batch oluşturulurken veri loader veya collate_fn içinde uygulanmalı.

# --- Frame atlama ve temporal jitter (video tabanlı augmentasyon) ---
def temporal_augment(frames, shuffle_prob=0.3, skip_prob=0.3):
    """Frame sırası karıştırma ve frame atlama uygular."""
    import random
    frames = frames.copy()
    # Frame sırası karıştırma
    if random.random() < shuffle_prob:
        random.shuffle(frames)
    # Frame atlama
    if random.random() < skip_prob and len(frames) > 3:
        idx_to_remove = random.randint(1, len(frames)-2)
        del frames[idx_to_remove]
        # Aynı uzunlukta tutmak için bir frame'i tekrar ekle
        frames.insert(idx_to_remove, frames[idx_to_remove-1])
    return frames 
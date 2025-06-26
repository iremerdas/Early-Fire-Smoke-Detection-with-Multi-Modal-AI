import cv2
import os
import numpy as np
from collections import defaultdict
from pathlib import Path
import argparse
from tqdm import tqdm
import re

def extract_base_name(img_name):
    """Augment edilmiş görsel adından orijinal adı çıkar"""
    if img_name.startswith('aug_'):
        # aug_0_original_name.jpg -> original_name
        parts = img_name.split('_', 2)  # En fazla 2 kez böl
        if len(parts) >= 3:
            return parts[2].rsplit('.', 1)[0]  # Uzantıyı kaldır
    return img_name.rsplit('.', 1)[0]  # Uzantıyı kaldır

def group_images_by_prefix(image_folder, include_augmented=True):
    groups = defaultdict(list)
    augmented_groups = defaultdict(list)
    
    for img in os.listdir(image_folder):
        if not img.lower().endswith('.jpg'):
            continue
            
        if img.startswith('aug_'):
            if not include_augmented:
                continue
            # Augment edilmiş görsel - orijinal adına göre grupla
            base_name = extract_base_name(img)
            if base_name.startswith('WEB'):
                prefix = 'WEB'
            elif base_name.startswith('AoF'):
                prefix = 'AoF'
            elif base_name.startswith('PublicDataset'):
                prefix = 'PublicDataset'
            elif base_name.startswith('ck0q'):
                prefix = 'ck0q'
            else:
                continue
            augmented_groups[prefix].append(img)
        else:
            # Normal görsel
            if img.startswith('WEB'):
                prefix = 'WEB'
            elif img.startswith('AoF'):
                prefix = 'AoF'
            elif img.startswith('PublicDataset'):
                prefix = 'PublicDataset'
            elif img.startswith('ck0q'):
                prefix = 'ck0q'
            else:
                continue
            groups[prefix].append(img)
    
    # Her grup için sırala
    for prefix in groups:
        groups[prefix].sort()
    for prefix in augmented_groups:
        augmented_groups[prefix].sort()
    
    return groups, augmented_groups

def group_augmented_images_by_original(image_folder):
    """Augment edilmiş görselleri orijinal görsel adına göre grupla"""
    original_to_augmented = defaultdict(list)
    
    for img in os.listdir(image_folder):
        if not img.lower().endswith('.jpg') or not img.startswith('aug_'):
            continue
            
        base_name = extract_base_name(img)
        original_to_augmented[base_name].append(img)
    
    # Her grup için sırala
    for original_name in original_to_augmented:
        original_to_augmented[original_name].sort()
    
    return original_to_augmented

def compute_mhi_for_group(image_folder, images, N=5, threshold=30, output_folder=None):
    if len(images) < N:
        print(f"MHI için yeterli ardışık kare yok: {images}")
        return
    h, w = None, None
    for start in range(len(images) - N + 1):
        mhi = None
        for i in range(N - 1):
            img1 = cv2.imread(os.path.join(image_folder, images[start + i]))
            img2 = cv2.imread(os.path.join(image_folder, images[start + i + 1]))
            if img1 is None or img2 is None:
                continue
            if h is None or w is None:
                h, w = img1.shape[:2]
            if img1.shape[:2] != (h, w):
                img1 = cv2.resize(img1, (w, h))
            if img2.shape[:2] != (h, w):
                img2 = cv2.resize(img2, (w, h))
            if img1.ndim == 2:
                img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
            if img2.ndim == 2:
                img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray2, gray1)
            _, motion_mask = cv2.threshold(diff, threshold, 1, cv2.THRESH_BINARY)
            if mhi is None:
                mhi = np.zeros((h, w), dtype=np.float32)
            mhi[motion_mask == 1] = N
            mhi[motion_mask == 0] -= 1
            mhi[mhi < 0] = 0
        # Her pencere için ayrı MHI kaydet
        mhi_img = np.uint8((mhi / N) * 255)
        if output_folder is not None:
            os.makedirs(output_folder, exist_ok=True)
            out_base = f"{images[start].split('.')[0]}_to_{images[start+N-1].split('.')[0]}_mhi"
            out_path = os.path.join(output_folder, out_base + ".jpg")
            cv2.imwrite(out_path, mhi_img)
            print(f"MHI kaydedildi: {out_path}")
    return

def compute_mhi_for_all_groups(image_folder, output_folder, N=5, threshold=30, include_augmented=True):
    groups, augmented_groups = group_images_by_prefix(image_folder, include_augmented)
    print(f"Toplam {len(groups)} normal grup bulundu.")
    if include_augmented:
        print(f"Toplam {len(augmented_groups)} augment edilmiş grup bulundu.")
    
    # Normal gruplar için MHI üret
    for prefix, images in groups.items():
        print(f"{prefix} (normal) için {len(images)} görsel ile MHI üretiliyor...")
        compute_mhi_for_group(image_folder, images, N=N, threshold=threshold, output_folder=output_folder)
    
    # Augment edilmiş gruplar için MHI üret
    if include_augmented:
        for prefix, images in augmented_groups.items():
            print(f"{prefix} (augmented) için {len(images)} görsel ile MHI üretiliyor...")
            compute_mhi_for_group(image_folder, images, N=N, threshold=threshold, output_folder=output_folder)

def compute_mhi_for_augmented_sequences(image_folder, output_folder, N=5, threshold=30):
    """Augment edilmiş görselleri orijinal görsel adına göre gruplayıp MHI üret"""
    original_to_augmented = group_augmented_images_by_original(image_folder)
    
    print(f"Toplam {len(original_to_augmented)} orijinal görsel için augment edilmiş versiyonlar bulundu.")
    
    for original_name, augmented_images in original_to_augmented.items():
        if len(augmented_images) >= N:
            print(f"{original_name} için {len(augmented_images)} augment edilmiş görsel ile MHI üretiliyor...")
            compute_mhi_for_group(image_folder, augmented_images, N=N, threshold=threshold, output_folder=output_folder)
        else:
            print(f"{original_name} için yeterli augment edilmiş görsel yok ({len(augmented_images)} < {N})")

def generate_mhi(image_paths, tau=30):
    """Motion History Image (MHI) üret"""
    # İlk görüntüyü yükle
    first_img = cv2.imread(str(image_paths[0]))
    if first_img is None:
        raise ValueError(f"Görüntü yüklenemedi: {image_paths[0]}")
    
    height, width = first_img.shape[:2]
    mhi = np.zeros((height, width), dtype=np.float32)
    
    # Her görüntü çifti için
    for i in range(len(image_paths) - 1):
        # Görüntüleri yükle
        img1 = cv2.imread(str(image_paths[i]))
        img2 = cv2.imread(str(image_paths[i + 1]))
        
        if img1 is None or img2 is None:
            print(f"Uyarı: Görüntü yüklenemedi, atlanıyor: {image_paths[i]} veya {image_paths[i + 1]}")
            continue
        
        # Gri tonlamaya çevir
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Fark görüntüsü
        diff = cv2.absdiff(gray1, gray2)
        
        # Eşikleme
        _, motion_mask = cv2.threshold(diff, 25, 1, cv2.THRESH_BINARY)
        
        # MHI güncelle
        mhi = cv2.addWeighted(mhi, 1.0, motion_mask, 1.0, 0)
        mhi = np.clip(mhi, 0, tau)
    
    # Normalize et
    mhi = mhi / tau
    
    return mhi

def process_sequence(image_paths, output_path):
    """Görüntü dizisi için MHI üret ve kaydet"""
    try:
        mhi = generate_mhi(image_paths)
        np.save(output_path, mhi)
        return True
    except Exception as e:
        print(f"Hata: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Motion History Image (MHI) üret')
    parser.add_argument('--input_dir', type=str, required=True, help='Görüntülerin bulunduğu dizin')
    parser.add_argument('--output_dir', type=str, required=True, help='MHI dosyalarının kaydedileceği dizin')
    parser.add_argument('--sequence_length', type=int, default=5, help='MHI için kullanılacak görüntü sayısı')
    parser.add_argument('--include_augmented', action='store_true', help='Augment edilmiş verileri de dahil et')
    parser.add_argument('--augmented_only', action='store_true', help='Sadece augment edilmiş veriler için MHI üret')
    args = parser.parse_args()
    
    # Dizinleri oluştur
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.augmented_only:
        # Sadece augment edilmiş veriler için MHI üret
        print("Sadece augment edilmiş veriler için MHI üretiliyor...")
        compute_mhi_for_augmented_sequences(
            str(input_dir), 
            str(output_dir), 
            N=args.sequence_length
        )
    else:
        # Normal ve/veya augment edilmiş veriler için MHI üret
        print("Normal ve augment edilmiş veriler için MHI üretiliyor...")
        compute_mhi_for_all_groups(
            str(input_dir), 
            str(output_dir), 
            N=args.sequence_length,
            include_augmented=args.include_augmented
        )

if __name__ == "__main__":
    main()
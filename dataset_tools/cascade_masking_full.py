import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from pathlib import Path
import argparse
from tqdm import tqdm

# Sınıfa göre HSV aralıkları ve özel parametreler
HSV_THRESHOLDS = {
    "fire":  {"lower": (0, 100, 100), "upper": (30, 255, 255)},      # Turuncu/kırmızı
    "smoke": {"lower": (0, 0, 50),    "upper": (180, 60, 220)},      # Gri/beyaz
    "both":  {"lower": (0, 0, 0),     "upper": (180, 255, 255)},     # Geniş aralık
}

# both için özel parametreler
BOTH_PARAMS = {
    "mhi_thresh": 3,
    "lbp_thresh": 0.8
}

def get_image_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return np.mean(hsv[...,2])

def get_histogram_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0,256])
    hist = hist.ravel() / hist.sum()
    mean = np.sum(hist * np.arange(256))
    std = np.sqrt(np.sum(hist * (np.arange(256) - mean)**2))
    return mean, std

def get_texture_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    hist = hist.astype('float') / (hist.sum() + 1e-6)
    uniformity = np.std(hist)
    entropy = -np.sum(hist * np.log2(hist + 1e-6))
    return uniformity, entropy

def auto_params(image, mhi, class_name):
    brightness = get_image_brightness(image)
    mean, std = get_histogram_features(image)
    uniformity, entropy = get_texture_features(image)

    # Sınıfa göre HSV aralığı
    if class_name in HSV_THRESHOLDS:
        lower_hsv = HSV_THRESHOLDS[class_name]["lower"]
        upper_hsv = HSV_THRESHOLDS[class_name]["upper"]
    else:
        lower_hsv = (0,0,0)
        upper_hsv = (180,255,255)

    # both için özel parametreler
    if class_name == "both":
        mhi_thresh = BOTH_PARAMS["mhi_thresh"]
        lbp_thresh = BOTH_PARAMS["lbp_thresh"]
    else:
        # MHI eşiği: parlaklık ve histogram ortalamasına göre
        if brightness < 60 or mean < 60:
            mhi_thresh = 2
        elif brightness < 120 or mean < 120:
            mhi_thresh = 5
        else:
            mhi_thresh = 10

        # LBP eşiği: doku uniformity ve entropy'ye göre
        if uniformity < 0.15 or entropy < 1.5:
            lbp_thresh = 0.9
        elif uniformity < 0.25 or entropy < 2.0:
            lbp_thresh = 0.7
        else:
            lbp_thresh = 0.5

    # (İsteğe bağlı) histogram std'si çok düşükse HSV aralığını genişlet
    if std < 30:
        lower_hsv = (0, 0, 0)
        upper_hsv = (180, 255, 255)

    return mhi_thresh, lbp_thresh, lower_hsv, upper_hsv

def apply_mhi_mask(frame, mhi, mhi_thresh):
    mhi_resized = cv2.resize(mhi, (frame.shape[1], frame.shape[0]))
    mhi_mask = (mhi_resized > mhi_thresh).astype(np.uint8)
    masked = cv2.bitwise_and(frame, frame, mask=mhi_mask)
    return masked, mhi_mask

def apply_color_filter(frame, lower_hsv, upper_hsv):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    masked = cv2.bitwise_and(frame, frame, mask=mask)
    return masked, mask

def apply_lbp_texture_filter(frame, lbp_thresh):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    hist = hist.astype('float') / (hist.sum() + 1e-6)
    uniformity = np.std(hist)
    if uniformity < lbp_thresh:
        mask = np.ones_like(gray, dtype=np.uint8)
    else:
        mask = np.zeros_like(gray, dtype=np.uint8)
    masked = cv2.bitwise_and(frame, frame, mask=mask)
    return masked, mask

def cascade_mask(frame, mhi, mhi_thresh, lower_hsv, upper_hsv, lbp_thresh):
    mhi_masked, mhi_mask = apply_mhi_mask(frame, mhi, mhi_thresh)
    color_masked, color_mask = apply_color_filter(mhi_masked, lower_hsv, upper_hsv)
    lbp_masked, lbp_mask = apply_lbp_texture_filter(color_masked, lbp_thresh)
    final_mask = cv2.bitwise_and(mhi_mask, color_mask)
    final_mask = cv2.bitwise_and(final_mask, lbp_mask)
    final_result = cv2.bitwise_and(frame, frame, mask=final_mask)
    return final_result, final_mask

def apply_cascade_masking(image_path, output_path):
    """Cascade masking uygula ve sonucu kaydet"""
    try:
        # Görüntüyü yükle
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Görüntü yüklenemedi: {image_path}")
        
        # Görüntüyü HSV'ye dönüştür
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Ateş için renk aralıkları
        lower_fire1 = np.array([0, 50, 50])
        upper_fire1 = np.array([10, 255, 255])
        lower_fire2 = np.array([170, 50, 50])
        upper_fire2 = np.array([180, 255, 255])
        
        # Duman için renk aralıkları
        lower_smoke = np.array([0, 0, 100])
        upper_smoke = np.array([180, 30, 255])
        
        # Ateş maskelerini oluştur
        mask_fire1 = cv2.inRange(hsv, lower_fire1, upper_fire1)
        mask_fire2 = cv2.inRange(hsv, lower_fire2, upper_fire2)
        mask_fire = cv2.bitwise_or(mask_fire1, mask_fire2)
        
        # Duman maskesini oluştur
        mask_smoke = cv2.inRange(hsv, lower_smoke, upper_smoke)
        
        # Morfolojik işlemler
        kernel = np.ones((5,5), np.uint8)
        mask_fire = cv2.morphologyEx(mask_fire, cv2.MORPH_OPEN, kernel)
        mask_fire = cv2.morphologyEx(mask_fire, cv2.MORPH_CLOSE, kernel)
        mask_smoke = cv2.morphologyEx(mask_smoke, cv2.MORPH_OPEN, kernel)
        mask_smoke = cv2.morphologyEx(mask_smoke, cv2.MORPH_CLOSE, kernel)
        
        # Maskeleri birleştir
        mask_combined = cv2.bitwise_or(mask_fire, mask_smoke)
        
        # Maskeyi 3 kanala genişlet
        mask_3ch = np.stack([mask_combined] * 3, axis=-1)
        
        # Maskeyi normalize et
        mask_3ch = mask_3ch.astype(np.float32) / 255.0
        
        # Orijinal görüntüyü normalize et
        img_norm = img.astype(np.float32) / 255.0
        
        # Görüntü ve maskeyi birleştir
        masked_img = np.concatenate([img_norm, mask_3ch], axis=-1)
        
        # Sonucu kaydet
        np.save(output_path, masked_img)
        return True
    
    except Exception as e:
        print(f"Hata: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Cascade masking uygula')
    parser.add_argument('--input_dir', type=str, required=True, help='İşlenecek görüntülerin bulunduğu dizin')
    parser.add_argument('--output_dir', type=str, required=True, help='İşlenmiş görüntülerin kaydedileceği dizin')
    args = parser.parse_args()
    
    # Dizinleri oluştur
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Görüntüleri işle
    for img_path in tqdm(sorted(input_dir.glob('*.jpg'))):
        if img_path.stem.startswith('aug_'):
            continue
        
        # Çıktı dosyası
        output_path = output_dir / f"{img_path.stem}_cascade.npy"
        
        # Eğer çıktı dosyası yoksa işle
        if not output_path.exists():
            apply_cascade_masking(img_path, output_path)

if __name__ == "__main__":
    main()
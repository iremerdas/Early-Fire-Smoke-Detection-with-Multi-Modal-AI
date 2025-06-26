import cv2
import os
import numpy as np
from skimage.feature import local_binary_pattern
import csv
from collections import defaultdict

# mhi_generator'daki gruplandırma fonksiyonu
def group_images_by_prefix(image_folder):
    groups = defaultdict(list)
    for img in os.listdir(image_folder):
        if img.lower().endswith('.jpg') and not img.startswith('aug_'):
            if img.startswith('WEB'):
                prefix = 'WEB'
            elif img.startswith('AoF'):
                prefix = 'AoF'
            elif img.startswith('PublicDataset'):
                prefix = 'PublicDataset'
            elif img.startswith('ck0q'):
                prefix = 'ck0q'
            else:
                continue  # Sadece tanımlı gruplar, diğerlerini atla
            groups[prefix].append(img)
    for prefix in groups:
        groups[prefix].sort()
    return groups

# Sınıf id -> isim eşlemesi
id2name = {0: "smoke", 1: "fire", 2: "none"}

def compute_red_ratio(img):
    b, g, r = cv2.split(img)
    red_ratio = np.sum(r > 150) / (img.shape[0] * img.shape[1])
    return red_ratio

def compute_lbp_hist(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    (lbp_hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    lbp_hist = lbp_hist.astype('float')
    lbp_hist /= (lbp_hist.sum() + 1e-6)
    return lbp_hist

def compute_color_hist(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256])
    hist_h = cv2.normalize(hist_h, hist_h).flatten()
    hist_s = cv2.normalize(hist_s, hist_s).flatten()
    hist_v = cv2.normalize(hist_v, hist_v).flatten()
    return np.concatenate([hist_h, hist_s, hist_v])

def extract_features_for_group(image_folder, labels_folder, images, output_csv):
    rows = []
    for img_name in images:
        base = os.path.splitext(img_name)[0]
        img_path = os.path.join(image_folder, img_name)
        label_path = os.path.join(labels_folder, base + ".txt")
        img = cv2.imread(img_path)
        if img is None or not os.path.exists(label_path):
            print(f"{img_path} veya etiketi okunamadı, atlanıyor.")
            continue
        with open(label_path, "r") as f:
            for line in f:
                if line.strip() == "":
                    continue
                parts = line.strip().split()
                class_id = int(parts[0])
                class_name = id2name.get(class_id, "unknown")
                # YOLO formatı: class x_center y_center width height
                x_center, y_center, width, height = map(float, parts[1:5])
                # Kutu koordinatlarını piksel cinsine çevir
                h, w = img.shape[:2]
                x = int((x_center - width / 2) * w)
                y = int((y_center - height / 2) * h)
                bw = int(width * w)
                bh = int(height * h)
                x = max(0, x)
                y = max(0, y)
                bw = min(w - x, bw)
                bh = min(h - y, bh)
                crop = img[y:y+bh, x:x+bw]
                if crop.size == 0:
                    continue
                # Özellikler
                red_ratio = compute_red_ratio(crop) if class_id == 1 else None
                lbp_hist = compute_lbp_hist(crop)
                color_hist = compute_color_hist(crop)
                feature_vec = [
                    img_name, class_id, class_name, x_center, y_center, width, height,
                    red_ratio if red_ratio is not None else "",
                ] + lbp_hist.tolist() + color_hist.tolist()
                rows.append(feature_vec)
    # CSV'ye yaz
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = [
            'image', 'class_id', 'class_name', 'x_center', 'y_center', 'width', 'height', 'red_ratio'
        ] + [f'lbp_{i}' for i in range(10)] + \
            [f'hist_h_{i}' for i in range(16)] + [f'hist_s_{i}' for i in range(16)] + [f'hist_v_{i}' for i in range(16)]
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Özellikler kaydedildi: {output_csv}")

def extract_features_for_all_groups(image_folder, labels_folder, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    groups = group_images_by_prefix(image_folder)
    for prefix, images in groups.items():
        output_csv = os.path.join(output_dir, f"{prefix}_features.csv")
        extract_features_for_group(image_folder, labels_folder, images, output_csv)

if __name__ == "__main__":
    # Kendi yollarını gir
    image_folder = r"C:\Users\Irem\OneDrive - Bursa Teknik Universitesi\Belgeler\TEZ\KOD\D-Fire\data_split\val\cascade_masked"
    labels_folder = r"C:\Users\Irem\OneDrive - Bursa Teknik Universitesi\Belgeler\TEZ\KOD\D-Fire\data_split\val\labels"
    output_dir = r"C:\Users\Irem\OneDrive - Bursa Teknik Universitesi\Belgeler\TEZ\KOD\D-Fire\data_split\train\features_train""ozellikler_val"
    extract_features_for_all_groups(image_folder, labels_folder, output_dir)
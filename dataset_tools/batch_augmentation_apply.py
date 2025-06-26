import os
import cv2
import shutil
from augmentation_advanced import get_transform_for_labels

# Klasör ve dosya yolları
IMAGE_DIR = r"C:\Users\Irem\OneDrive - Bursa Teknik Universitesi\Belgeler\TEZ\KOD\D-Fire\data_split\train\images"     # Orijinal görsellerin olduğu klasör
LABEL_DIR = r"C:\Users\Irem\OneDrive - Bursa Teknik Universitesi\Belgeler\TEZ\KOD\D-Fire\data_split\train\labels"     # Etiket dosyalarının bulunduğu klasör
# Augment edilmiş görseller ve etiketler de bu klasörlere kaydedilecek

# Sınıfa göre augmentasyon oranları
AUG_PER_CLASS = {
    0: 5,   # smoke
    1: 5,   # fire
    2: 0,   # none
    'both': 2  # hem fire hem smoke
}

def get_label_set_for_image(img_name):
    base = os.path.splitext(img_name)[0]
    label_txt = os.path.join(LABEL_DIR, base + '.txt')
    label_set = set()
    if os.path.exists(label_txt):
        with open(label_txt, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                class_id = line.split()[0]
                label_set.add(int(class_id))
    return label_set

def get_aug_count(label_set):
    if label_set == {0, 1}:
        return AUG_PER_CLASS['both']
    elif len(label_set) == 1:
        return AUG_PER_CLASS.get(list(label_set)[0], 0)
    else:
        return 0

for img_name in os.listdir(IMAGE_DIR):
    if not img_name.lower().endswith('.jpg'):
        continue
    label_set = get_label_set_for_image(img_name)
    if not label_set:
        print(f"Etiketi bulunamadı veya boş: {img_name}")
        continue
    aug_count = get_aug_count(label_set)
    if aug_count == 0:
        continue  # None sınıfı için augmentasyon yok
    img_path = os.path.join(IMAGE_DIR, img_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Görsel okunamadı: {img_path}")
        continue
    # Sınıfa göre transform seç
    transform = get_transform_for_labels(list(label_set))
    for i in range(aug_count):
        augmented = transform(image=img, class_labels=list(label_set))
        aug_img = augmented['image'].permute(1,2,0).cpu().numpy()
        aug_img = (aug_img * 255).astype('uint8') if aug_img.max() <= 1.0 else aug_img.astype('uint8')
        out_img_name = f"aug_{i}_{img_name}"
        out_img_path = os.path.join(IMAGE_DIR, out_img_name)
        cv2.imwrite(out_img_path, aug_img)
        # Etiket dosyasını da kopyala
        orig_label_name = os.path.splitext(img_name)[0] + '.txt'
        aug_label_name = os.path.splitext(out_img_name)[0] + '.txt'
        orig_label_path = os.path.join(LABEL_DIR, orig_label_name)
        aug_label_path = os.path.join(LABEL_DIR, aug_label_name)
        shutil.copy2(orig_label_path, aug_label_path)
        print(f"Augment edildi: {out_img_path} ve {aug_label_path}")
 
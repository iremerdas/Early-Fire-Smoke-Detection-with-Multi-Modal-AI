import os
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

def extract_sequence_info_from_filename(npy_filename):
    """
    .npy dosya adından 5'li grup bilgisini çıkar
    Örn: AoF00001_to_AoF00005_rgbmhi.npy -> (AoF00001, AoF00005)
    """
    # _rgbmhi.npy kısmını kaldır
    base_name = npy_filename.replace('_rgbmhi.npy', '').replace('_rgbmhiflow.npy', '')
    
    if '_to_' in base_name:
        start_name, end_name = base_name.split('_to_')
        return start_name, end_name
    else:
        return None, None

def get_sequence_images(start_name, end_name):
    """
    Başlangıç ve bitiş isimlerinden 5'li grup görsellerini oluştur
    """
    if start_name.startswith('aug_'):
        # Augment edilmiş dosyalar
        parts = start_name.split('_', 2)
        if len(parts) >= 3:
            aug_start = int(parts[1])
            base_name = parts[2]
            sequence = []
            for i in range(5):
                sequence.append(f"aug_{aug_start + i}_{base_name}.jpg")
            return sequence
    else:
        # Normal dosyalar (AoF, WEB, vs.)
        prefix = start_name[:-5]  # AoF
        start_num = int(start_name[-5:])
        sequence = []
        for i in range(5):
            num = start_num + i
            sequence.append(f"{prefix}{num:05d}.jpg")
        return sequence

def create_labels_for_sequence(labels_dir, image_sequence):
    """
    5'li görsel grubunun etiketlerini birleştir
    """
    all_labels = set()
    all_bboxes = []
    
    for img_name in image_sequence:
        txt_name = img_name.rsplit('.', 1)[0] + ".txt"
        txt_path = os.path.join(labels_dir, txt_name)
        
        if not os.path.exists(txt_path):
            continue
            
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                label_id = int(parts[0])
                all_labels.add(label_id)
                all_bboxes.append(line.strip())
    
    # Sınıf etiketi belirleme
    class_info = None
    if all_labels == {2} or len(all_labels) == 0:
        class_info = 'none'
    elif 1 in all_labels and 0 in all_labels:
        class_info = 'smoke+fire'
    elif 1 in all_labels:
        class_info = 'fire'
    elif 0 in all_labels:
        class_info = 'smoke'
    else:
        class_info = 'none'
    
    return class_info, all_bboxes

def main():
    parser = argparse.ArgumentParser(description='RGBMHI/RGBMHIFLOW dosyaları için etiket dosyaları oluştur')
    parser.add_argument('--rgbmhiflow_dir', type=str, required=True, help='RGBMHI/RGBMHIFLOW .npy dosyalarının bulunduğu dizin')
    parser.add_argument('--labels_dir', type=str, required=True, help='Orijinal etiketlerin bulunduğu dizin')
    parser.add_argument('--output_dir', type=str, required=True, help='Etiket dosyalarının kaydedileceği dizin')
    args = parser.parse_args()

    rgbmhiflow_dir = Path(args.rgbmhiflow_dir)
    labels_dir = Path(args.labels_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # .npy dosyalarını bul
    npy_files = [f for f in os.listdir(rgbmhiflow_dir) if f.endswith('.npy')]
    print(f"Toplam {len(npy_files)} .npy dosyası bulundu.")

    processed_count = 0
    error_count = 0

    for npy_file in tqdm(npy_files, desc="Etiket dosyaları oluşturuluyor"):
        try:
            # Dosya adından 5'li grup bilgisini çıkar
            start_name, end_name = extract_sequence_info_from_filename(npy_file)
            if start_name is None or end_name is None:
                print(f"Uyarı: Dosya adı formatı tanınmadı: {npy_file}")
                error_count += 1
                continue

            # 5'li grup görsellerini oluştur
            image_sequence = get_sequence_images(start_name, end_name)
            
            # Etiketleri oluştur
            class_info, all_bboxes = create_labels_for_sequence(str(labels_dir), image_sequence)
            
            # Çıktı dosya adı
            base_name = npy_file.replace('.npy', '')
            txt_path = output_dir / (base_name + ".txt")
            
            # Etiket dosyasını kaydet
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(f"# class_info: {class_info}\n")
                for bbox in all_bboxes:
                    f.write(bbox + "\n")
            
            processed_count += 1
            
        except Exception as e:
            print(f"Hata: {npy_file} işlenirken hata oluştu: {str(e)}")
            error_count += 1
            continue

    print(f"\nİşlem tamamlandı!")
    print(f"Başarıyla işlenen: {processed_count}")
    print(f"Hata alan: {error_count}")
    print(f"Toplam: {len(npy_files)}")

if __name__ == "__main__":
    main() 
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

def get_five_image_group_from_mhi_filename(mhi_filename, rgb_dir):
    """
    MHI dosya adına göre ilgili 5'li ardışık görsel grubunun dosya adlarını bulur.
    Sıralama: 0,1,2,3,4 (veya WEB00000_to_WEB00004_mhi.npy -> WEB00000.jpg ... WEB00004.jpg)
    """
    if mhi_filename.startswith('aug_'):
        ana, son = mhi_filename.split('_to_')
        ana_idx = int(ana.split('_')[1])  # 0
        ana_name = ana.split('_', 2)[2].rsplit('.', 1)[0]  # PublicDataset00519
        son_idx = int(son.split('_')[1])  # 4
        group = []
        for i in range(ana_idx, son_idx+1):
            found = False
            for ext in ['.jpg', '.jpeg', '.png']:
                pattern = f"aug_{i}_{ana_name}"
                for file in os.listdir(rgb_dir):
                    if file.startswith(pattern) and file.endswith(ext):
                        group.append(os.path.join(rgb_dir, file))
                        found = True
                        break
                if found:
                    break
            if not found:
                group.append(None)  # Eksik dosya
        return group
    else:
        # WEB00000_to_WEB00004_mhi.jpg veya AoF00000_to_AoF00004_mhi.jpg
        ana, son = mhi_filename.split('_to_')
        prefix = ana[:-5]  # WEB veya AoF
        start = int(ana[-5:])
        end_core = son.split('_')[0]  # 'AoF00006'
        end_idx = int(end_core[-5:])
        group = []
        for i in range(start, end_idx+1):
            fname = f"{prefix}{i:05d}.jpg"
            fpath = os.path.join(rgb_dir, fname)
            if os.path.exists(fpath):
                group.append(fpath)
            else:
                group.append(None)
        return group

def find_rgb_image_variants(rgb_dir, base_name):
    """
    Eğer 'aug_2_' dosyası yoksa, diğer varyasyonları da dene.
    base_name: örn. '1000_png'
    """
    for i in [2, 0, 1, 3, 4]:
        for ext in ['.jpg', '.jpeg', '.png']:
            pattern = f"aug_{i}_{base_name}"
            for file in os.listdir(rgb_dir):
                if file.startswith(pattern) and file.endswith(ext):
                    return os.path.join(rgb_dir, file)
    return None

def find_normal_image_variants(rgb_dir, prefix, middle_num):
    """
    Normal dosyalar için ortadaki görsel bulunamazsa, yakın varyasyonları dene.
    Örn: AoF00175.jpg yoksa, AoF00174.jpg, AoF00176.jpg gibi
    """
    for offset in [0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5]:  # Daha geniş arama
        num = middle_num + offset
        fname = f"{prefix}{num:05d}.jpg"
        fpath = os.path.join(rgb_dir, fname)
        if os.path.exists(fpath):
            return fpath
    return None

def compute_optical_flow_magnitude_stack(image_paths):
    """
    5'li ardışık görsel grubundan optical flow magnitude ortalamasını döndürür (normalize edilmiş).
    image_paths: 5 adet görsel yolu (eksik varsa None)
    """
    flows = []
    target_shape = None
    for i in range(4):
        p1 = image_paths[i]
        p2 = image_paths[i+1]
        if p1 is None or p2 is None:
            flows.append(None)
            continue
        img1 = cv2.imread(p1)
        img2 = cv2.imread(p2)
        if img1 is None or img2 is None:
            flows.append(None)
            continue
        if img1.shape[:2] != img2.shape[:2]:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            img2 = img2.astype(img1.dtype)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        if gray1.shape != gray2.shape:
            gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
        flow = cv2.calcOpticalFlowFarneback(gray1.astype(np.uint8), gray2.astype(np.uint8), np.zeros_like(gray1, dtype=np.float32), 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        if target_shape is None:
            target_shape = mag.shape
        else:
            if mag.shape != target_shape:
                mag = cv2.resize(mag, (target_shape[1], target_shape[0]))
                mag = mag.astype(np.float32)
        flows.append(mag)
    valid_flows = [f for f in flows if f is not None]
    if not valid_flows:
        return None
    avg_mag = np.mean(valid_flows, axis=0)
    if np.max(avg_mag) > 0:
        avg_mag = avg_mag / np.max(avg_mag)
    return avg_mag.astype(np.float32)

def stack_rgb_mhi_flow(rgb_path, mhi_path, optical_flow_mag, output_path):
    try:
        rgb_img = cv2.imread(str(rgb_path))
        if rgb_img is None:
            print(f"Uyarı: RGB görüntüsü yüklenemedi: {rgb_path}")
            return False
        if str(mhi_path).endswith('.jpg'):
            mhi = cv2.imread(str(mhi_path), cv2.IMREAD_GRAYSCALE)
            if mhi is None:
                print(f"Uyarı: MHI görüntüsü yüklenemedi: {mhi_path}")
                return False
            mhi = mhi.astype(np.float32) / 255.0
        else:
            mhi = np.load(str(mhi_path))
        if rgb_img.shape[:2] != mhi.shape[:2]:
            mhi = cv2.resize(mhi, (rgb_img.shape[1], rgb_img.shape[0]))
            mhi = mhi.astype(np.float32)
        rgb_norm = rgb_img.astype(np.float32) / 255.0
        mhi_ch = mhi[..., np.newaxis]  # (H,W,1)
        if optical_flow_mag is not None:
            if rgb_img.shape[:2] != optical_flow_mag.shape[:2]:
                optical_flow_mag = cv2.resize(optical_flow_mag, (rgb_img.shape[1], rgb_img.shape[0]))
                optical_flow_mag = optical_flow_mag.astype(np.float32)
            optical_flow_mag = optical_flow_mag[..., np.newaxis]  # (H,W,1)
        else:
            optical_flow_mag = np.zeros((rgb_img.shape[0], rgb_img.shape[1], 1), dtype=np.float32)
        stacked = np.concatenate([rgb_norm, mhi_ch, optical_flow_mag], axis=-1)  # (H,W,5)
        np.save(output_path, stacked)
        print(f"[✓] Kaydedildi: {output_path.name}")
        return True
    except Exception as e:
        print(f"Hata: {str(e)}")
        return False

def load_frames(rgb_dir, frame_names):
    frames = []
    for fname in frame_names:
        fpath = os.path.join(rgb_dir, fname)
        if not os.path.exists(fpath):
            return None  # Eksik frame varsa None dön
        img = cv2.imread(fpath)
        if img is None:
            return None
        frames.append(img)
    return frames

def compute_mhi(frames, tau=30):
    if len(frames) < 2:
        return None
    height, width = frames[0].shape[:2]
    mhi = np.zeros((height, width), dtype=np.float32)
    for i in range(len(frames) - 1):
        gray1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frames[i+1], cv2.COLOR_BGR2GRAY)
        if gray1.shape != gray2.shape:
            gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
        diff = cv2.absdiff(gray1, gray2)
        _, motion_mask = cv2.threshold(diff, 25, 1, cv2.THRESH_BINARY)
        motion_mask = motion_mask.astype(np.float32)
        if mhi.shape != motion_mask.shape:
            motion_mask = cv2.resize(motion_mask, (mhi.shape[1], mhi.shape[0]))
            motion_mask = motion_mask.astype(np.float32)
        mhi = cv2.addWeighted(mhi, 1.0, motion_mask, 1.0, 0)
        mhi = np.clip(mhi, 0, tau)
    mhi = mhi / tau
    return mhi

def compute_optical_flow(frames):
    if len(frames) < 2:
        return None
    flows = []
    target_shape = None
    for i in range(len(frames) - 1):
        gray1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frames[i+1], cv2.COLOR_BGR2GRAY)
        if gray1.shape != gray2.shape:
            gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
        flow = cv2.calcOpticalFlowFarneback(
            gray1.astype(np.uint8), gray2.astype(np.uint8), np.zeros_like(gray1, dtype=np.float32), 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        if target_shape is None:
            target_shape = mag.shape
        else:
            if mag.shape != target_shape:
                mag = cv2.resize(mag, (target_shape[1], target_shape[0]))
        flows.append(mag)
    avg_mag = np.mean(flows, axis=0)
    if np.max(avg_mag) > 0:
        avg_mag = avg_mag / np.max(avg_mag)
    return avg_mag.astype(np.float32)

def create_augmented_group(frame_names, aug_type):
    if aug_type == 'original':
        return frame_names
    elif aug_type == 'skip':
        return frame_names[::2]
    elif aug_type == 'jitter':
        # 1 frame ileri kaydır (ör: 1-2-3-4-5)
        if len(frame_names) > 1:
            offset = 1
            jittered = frame_names[offset:] + frame_names[:offset]
            return jittered[:len(frame_names)]
        else:
            return frame_names
    else:
        return frame_names

def process_mhi_file(mhi_path, rgb_dir, output_dir, aug_type):
    mhi_name = mhi_path.name
    # Dosya ismine göre frame isimlerini çıkar (ör: WEB00000_to_WEB00004_mhi.npy)
    base = mhi_name.replace('_mhi.npy', '')
    parts = base.split('_to_')
    if len(parts) != 2:
        return
    start, end = parts
    # Frame isimleri: WEB00000.jpg, WEB00001.jpg, ...
    start_core = start.split('_')[0]  # 'AoF00000' veya 'WEB00000'
    end_core = end.split('_')[0]      # 'AoF00006' veya 'WEB00006'
    start_idx = int(start_core[-5:])
    end_idx = int(end_core[-5:])
    frame_names = [f"WEB{str(i).zfill(5)}.jpg" for i in range(start_idx, end_idx+1)]
    aug_frame_names = create_augmented_group(frame_names, aug_type)
    frames = load_frames(rgb_dir, aug_frame_names)
    if frames is None or len(frames) < 2:
        return
    mhi = compute_mhi(frames)
    flow = compute_optical_flow(frames)
    if mhi is None or flow is None:
        return
    middle_frame = frames[len(frames)//2]
    rgb_norm = middle_frame.astype(np.float32) / 255.0
    mhi_resized = cv2.resize(mhi, (rgb_norm.shape[1], rgb_norm.shape[0]))
    flow_resized = cv2.resize(flow, (rgb_norm.shape[1], rgb_norm.shape[0]))
    combined = np.concatenate([
        rgb_norm, 
        mhi_resized[..., np.newaxis], 
        flow_resized[..., np.newaxis]
    ], axis=-1)  # (H, W, 5)
    # Dosya ismi
    if aug_type == 'original':
        out_name = base + '_rgbmhiflow.npy'
    elif aug_type == 'skip':
        out_name = f'aug_skip2_{base}_rgbmhiflow.npy'
    elif aug_type == 'jitter':
        out_name = f'aug_jitter1_{base}_rgbmhiflow.npy'
    else:
        out_name = base + '_rgbmhiflow.npy'
    out_path = output_dir / out_name
    if out_path.exists():
        return  # Üzerine yazma
    np.save(str(out_path), combined)

def main():
    parser = argparse.ArgumentParser(description='RGB+MHI+Flow 5-kanal .npy üretici')
    parser.add_argument('--rgb_dir', type=str, required=True, help='RGB görüntü klasörü')
    parser.add_argument('--mhi_dir', type=str, required=True, help='MHI .npy klasörü')
    parser.add_argument('--output_dir', type=str, required=True, help='Çıktı .npy klasörü')
    args = parser.parse_args()

    rgb_dir = args.rgb_dir
    mhi_dir = Path(args.mhi_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mhi_files = sorted([f for f in mhi_dir.iterdir() if f.name.endswith('_mhi.npy') or f.name.endswith('_mhi.jpg')])
    aug_types = ['original', 'skip', 'jitter']

    for aug_type in aug_types:
        print(f'Augmentasyon türü: {aug_type}')
        for mhi_path in tqdm(mhi_files, desc=f'{aug_type}'): 
            process_mhi_file(mhi_path, rgb_dir, output_dir, aug_type)

if __name__ == '__main__':
    main() 
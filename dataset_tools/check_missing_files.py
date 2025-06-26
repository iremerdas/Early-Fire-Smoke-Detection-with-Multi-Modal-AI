import os

image_folder = r"C:\Users\Irem\OneDrive - Bursa Teknik Universitesi\Belgeler\TEZ\KOD\D-Fire\data_split\train\images"
mhi_folder = r"C:\Users\Irem\OneDrive - Bursa Teknik Universitesi\Belgeler\TEZ\KOD\D-Fire\data_split\train\mhi2"

# Tüm RGB görsellerin isimlerini al (uzantısız)
rgb_names = {os.path.splitext(f)[0] for f in os.listdir(image_folder) if f.endswith('.jpg')}

# Silinecek MHI dosyalarını bul ve sil
silinecekler = []
for mhi_file in os.listdir(mhi_folder):
    if mhi_file.endswith('_mhi.jpg'):
        base = mhi_file.split('_to_')[0]
        if base not in rgb_names:
            mhi_path = os.path.join(mhi_folder, mhi_file)
            os.remove(mhi_path)
            silinecekler.append(mhi_file)
            print(f"SİLİNDİ: {mhi_path}")

print(f"Toplam silinen MHI dosyası: {len(silinecekler)}")
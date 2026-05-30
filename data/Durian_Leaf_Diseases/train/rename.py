from pathlib import Path
import pandas as pd
from PIL import Image

# Thư mục train của bạn
TRAIN_ROOT = Path(
    r"C:\Users\Lenovo\Downloads\A Durian Leaf Image Dataset of Common Diseases in Vietnam for Agricultural Diagnosis\Durian_Leaf_Diseases\train"
)

# Nơi lưu metadata
METADATA_PATH = TRAIN_ROOT / "metadata_train.csv"

# Thứ tự class muốn duyệt
CLASS_ORDER = [
    "algal",
    "allocaridara attack",
    "blight",
    "healthy",
    "phomopsis"
]

# Map label_id cố định
LABEL_MAP = {
    "healthy": 0,
    "algal": 1,
    "allocaridara attack": 2,
    "blight": 3,
    "phomopsis": 4
}

# Tên label chuẩn để đưa vào CSV
NORMALIZED_LABEL = {
    "healthy": "healthy",
    "algal": "algal",
    "allocaridara attack": "allocaridara_attack",
    "blight": "blight",
    "phomopsis": "phomopsis"
}

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"]

# Xóa metadata cũ nếu có
if METADATA_PATH.exists():
    METADATA_PATH.unlink()

# Bước 1: đổi toàn bộ ảnh sang tên tạm để tránh trùng tên
temp_records = []

for class_name in CLASS_ORDER:
    class_dir = TRAIN_ROOT / class_name

    if not class_dir.exists():
        print(f"Không thấy folder: {class_dir}")
        continue

    image_paths = []
    for ext in IMAGE_EXTS:
        image_paths.extend(class_dir.glob(f"*{ext}"))
        image_paths.extend(class_dir.glob(f"*{ext.upper()}"))

    image_paths = sorted(set(image_paths))

    print(f"{class_name}: {len(image_paths)} ảnh")

    for i, old_path in enumerate(image_paths, start=1):
        temp_name = f"__tmp_rename_{class_name.replace(' ', '_')}_{i:06d}{old_path.suffix.lower()}"
        temp_path = class_dir / temp_name

        # Nếu còn file tạm từ lần chạy lỗi trước đó thì xóa
        if temp_path.exists():
            temp_path.unlink()

        old_path.rename(temp_path)

        temp_records.append({
            "class_name": class_name,
            "temp_path": temp_path,
            "original_file_name": old_path.name,
            "original_path": str(old_path.relative_to(TRAIN_ROOT)).replace("\\", "/"),
            "original_extension": old_path.suffix.lower()
        })

# Bước 2: convert sang JPG và đổi tên chuẩn tăng liên tục
rows = []
counter = 1

for record in temp_records:
    class_name = record["class_name"]
    temp_path = record["temp_path"]

    image_id = f"DLDD_TRAIN_{counter:06d}"
    new_name = f"{image_id}.jpg"
    new_path = temp_path.parent / new_name

    # Nếu tên mới đã tồn tại thì xóa để ghi đè
    if new_path.exists():
        new_path.unlink()

    try:
        with Image.open(temp_path) as img:
            img = img.convert("RGB")
            img.save(new_path, "JPEG", quality=95)

        # Xóa ảnh tạm sau khi convert thành công
        temp_path.unlink()

        rows.append({
            "image_id": image_id,
            "file_name": new_name,
            "file_path": str(new_path.relative_to(TRAIN_ROOT)).replace("\\", "/"),
            "original_file_name": record["original_file_name"],
            "original_path": record["original_path"],
            "original_extension": record["original_extension"],
            "split": "train",
            "label_id": LABEL_MAP[class_name],
            "disease_type": NORMALIZED_LABEL[class_name],
            "verification_status": "verified"
        })

        counter += 1

    except Exception as e:
        print(f"Lỗi convert {temp_path}: {e}")
        # Không xóa temp_path nếu lỗi để còn kiểm tra lại

df = pd.DataFrame(rows)
df.to_csv(METADATA_PATH, index=False, encoding="utf-8-sig")

print("\nHoàn tất đổi tên + convert JPG cho TRAIN.")
print(f"Tổng số ảnh xử lý thành công: {len(df)}")
print(f"Metadata lưu tại: {METADATA_PATH}")

if len(df) > 0:
    print(df.groupby("disease_type").size())
else:
    print("Không có ảnh nào được xử lý.")
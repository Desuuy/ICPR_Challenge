import os, json

ROOT = r"C:\Users\anhhu\MultiFrame-LPR\Data\train"
OUT_DIR = r"C:\Users\anhhu\lp_for_openocr"
IMG_DIR = os.path.join(OUT_DIR, "images")
LABEL_FILE = os.path.join(OUT_DIR, "labels.txt")

os.makedirs(IMG_DIR, exist_ok=True)

lines = []
for root, dirs, files in os.walk(ROOT):
    if "annotations.json" in files:
        jpath = os.path.join(root, "annotations.json")
        with open(jpath, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                data = data[0]
        label = data.get("plate_text") or data.get("license_plate") or data.get("text")
        if not label:
            continue
        # lấy 1 ảnh HR đại diện, ví dụ hr-001.png
        img_name = "hr-001.png"
        ipath = os.path.join(root, img_name)
        if not os.path.exists(ipath):
            continue
        # copy ảnh sang thư mục phẳng
        track_id = os.path.basename(root)
        new_name = f"{track_id}.png"
        new_path = os.path.join(IMG_DIR, new_name)
        if not os.path.exists(new_path):
            import shutil
            shutil.copy2(ipath, new_path)
        # dòng: relative_path \t label
        rel_path = os.path.relpath(new_path, IMG_DIR)
        lines.append(f"{rel_path}\t{label}\n")

with open(LABEL_FILE, "w", encoding="utf-8") as f:
    f.writelines(lines)

print(f"Saved {len(lines)} lines to {LABEL_FILE}")
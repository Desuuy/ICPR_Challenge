# Merge SR (feat/Restoration_Module) vào main – Step by step

Chỉ cần chạy các lệnh theo thứ tự dưới đây trong terminal (PowerShell hoặc CMD).

---

## Bước 1: Commit thay đổi (nếu có)

```bash
git status
git add .
git commit -m "SR integration ready for merge"
```

---

## Bước 2: Chuyển sang nhánh main

```bash
git checkout main
```

---

## Bước 3: Merge nhánh feat/Restoration_Module vào main

```bash
git merge feat/Restoration_Module -m "Merge SR (LP-Diff) integration into main"
```

Nếu có conflict: mở file conflict, sửa xong chạy `git add <file>` rồi `git commit`.

---

## Bước 4: Kiểm tra kết quả

```bash
git log --oneline -3
```

Đảm bảo có `src/sr/`, `sr_model/`, `train.py` có `--use-sr`, `dataset.py` có `sr_enhancer`.

---

## Bước 5: Push lên remote

```bash
git push origin main
```

**Lưu ý:** Nếu push fail vì file weights quá lớn, weights có thể đã bị commit trước đó trong history. Cần xóa khỏi history bằng `git filter-repo` hoặc tương tự.

---

## Bước 6: Test SR

```bash
python train.py --batch-size 32 --use-sr --epochs 1
```

Cần có checkpoint SR đặt tại `weights/I80000_E41_gen_best_psnr.pth` (hoặc đường dẫn trong config).

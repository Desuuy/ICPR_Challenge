# Đề xuất cải tiến độ chính xác – MultiFrame LPR

README này tóm tắt các phương án đề xuất để tăng độ chính xác (accuracy / CER) cho model. Bạn có thể xem trước, chọn mục muốn áp dụng, rồi triển khai theo tài liệu chi tiết trong `IMPROVEMENT_PROPOSALS.md`.

---

## Tổng quan

| Ưu tiên | Cải tiến | Effort | Tác động ước lượng |
|--------|----------|--------|---------------------|
| 1 | Cùng augmentation cho 5 frame | Thấp | +1–3% acc |
| 2 | Focal-style CTC loss | Thấp | +0.5–2% acc |
| 3 | CER metric + CTC beam decode | Thấp | +0.5–2% acc |
| 4 | Dropout trong STN / Fusion / Head | Thấp | +0.5–1.5% acc |
| 5 | Augmentation hướng LPR (blur, noise) | Trung bình | +1–3% acc |
| 6 | Early stopping + lưu best theo CER | Thấp | Gián tiếp |
| 7 | Backbone SVTRv2 Base (lớn hơn) | Cao | +2–5% acc |
| 8 | TTA / Ensemble | Trung bình | +1–3% acc |

---

## 1. Data & Augmentation

### 1.1 Cùng augmentation cho cả 5 frame
- **Vấn đề:** Mỗi frame đang augment riêng → 5 frame không còn đồng bộ, fusion bị nhiễu.
- **Giải pháp:** Dùng chung một bộ tham số augment cho cả 5 frame (seed theo sample).
- **Cách bật:** Trong `configs/config.py` set `SAME_AUG_PER_SAMPLE = True` (nếu đã có trong code).

### 1.2 Augmentation phù hợp LPR
- Thêm MotionBlur, GaussianBlur, GaussNoise nhẹ (giống camera thật).
- Perspective nhẹ cho biển nghiêng/xa.
- Có thể tăng cường degradation cho synthetic LR (blur, nén, downscale).

---

## 2. Loss & Training

### 2.1 Focal-style CTC
- Tăng trọng số sample “khó” (loss cao), giảm ảnh hưởng sample “dễ”.
- **Cách bật:** Trong `configs/config.py` set `USE_FOCAL_CTC = True`.
- Nên thử khi dataset có nhiều ảnh dễ và ít ảnh khó.

### 2.2 Learning rate & epoch
- Có thể giảm LR (vd. 2e-4) khi fine-tune pretrained.
- Tăng EPOCHS (40–50) nếu chưa overfit.

---

## 3. Model

### 3.1 Dropout
- Thêm Dropout 0.1–0.2 trong STN, Fusion, hoặc Head để giảm overfit.

### 3.2 Backbone lớn hơn (SVTRv2 Base)
- Trong `configs/config.py` đã có comment sẵn cấu hình Base (dims/depths/heads).
- Bật khi đã tối ưu data + loss; cần nhiều GPU và thời gian train hơn.

---

## 4. Decoding & Validation

### 4.1 CTC beam search
- Thay greedy decode bằng beam search (beam 5–10) có thể cải thiện 0.5–2% acc.
- **Cách bật:** Trong `configs/config.py` set `CTC_BEAM_WIDTH = 5` (hoặc 10).

### 4.2 Theo dõi CER
- Validation ngoài accuracy nên tính **CER** (Character Error Rate) để đánh giá sát hơn.
- Có thể chọn best checkpoint theo CER thay vì chỉ theo accuracy.

---

## 5. Khác

### 5.1 Early stopping
- Dừng train nếu val acc/CER không cải thiện sau N epoch để tránh overfit.

### 5.2 Test-time augmentation (TTA)
- Inference nhiều phiên bản (gốc, flip, v.v.) rồi vote hoặc average logits → thường +0.5–1% acc, tốn 2–4x thời gian.

### 5.3 Ensemble
- Train 2–3 model (khác seed / aug) rồi average logits hoặc majority vote → thường +1–3% acc.

---

## Cấu hình đề xuất (config)

Trong `configs/config.py` có thể thêm/bật:

```python
# Đã có sẵn trong config (nếu đã tích hợp)
USE_FOCAL_CTC = False   # Bật True để focus vào sample khó
CTC_BEAM_WIDTH = 1       # 5 hoặc 10 để dùng beam decode
SAME_AUG_PER_SAMPLE = True  # Cùng aug cho 5 frame
```

---

## Tài liệu chi tiết

- **Chi tiết kỹ thuật và thứ tự triển khai:** xem `docs/IMPROVEMENT_PROPOSALS.md`.

Sau khi xem README này, bạn có thể quyết định bật/tắt từng mục trong config hoặc làm theo từng bước trong file đề xuất đầy đủ.

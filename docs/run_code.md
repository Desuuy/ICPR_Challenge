## 4-Phase Training Strategy cho MultiFrame-LPR

File này mô tả **các giai đoạn huấn luyện** và **lệnh tương ứng** (dùng `train_optimized.py` + config hiện tại).  
Giả định bạn đang ở thư mục gốc project `MultiFrame-LPR` và dùng **PowerShell/CMD** trên Windows.

---

## Giai đoạn 1 – Warm-up & Context Alignment (15–20 epochs)

**Mục tiêu**  
- Làm quen với input **10 frames (5 LQ + 5 HQ)** và **Country Embedding**.  
- Không bật STN, không dùng SR, dùng CTC thường để hội tụ nhanh.  
- Hạn chế làm hỏng backbone pretrained.

**Cấu hình chính (GPU 4GB, có gradient accumulation)**  
- `USE_STN = False` (dùng flag `--no-stn` để ép tắt).  
- `USE_SR = False` (không dùng `--use-sr`).  
- `USE_FOCAL_CTC = False` (giữ mặc định CTC thường trong `config.py`).  
- `BATCH_SIZE PHYSICAL = 4` (do hạn chế VRAM).  
- `ACCUM_STEPS = 8` (trong `config.py` → Effective Batch = 4 × 8 = 32).  
- `LEARNING_RATE ≈ 1e-4` khi fine-tune từ pretrained; khi warm-up từ random có thể cao hơn, nhưng với GPU 4GB nên ưu tiên LR thấp để tránh NaN.  
- Optimizer: `AdamW` với 3 nhóm tham số (STN / Backbone / Head+Fusion+Country) như đã tích hợp trong `trainer.py`.

**Lệnh gợi ý (Phase 1 – Recovery từ pretrained, 15–20 epochs)**

```bash
python train_optimized.py -n phase1_recovery -m mf_svtrv2 --epochs 20 --batch-size 4 --learning-rate 0.0001 --no-stn --aug-level light
```

Ghi chú:
- **Không dùng `--no-pretrained`** để tận dụng `weights/best.pth` đã cấu hình trong `config.py` (backbone ổn định hơn rất nhiều).  
- `ACCUM_STEPS` đã được set trong `configs/config.py`, Trainer sẽ tự apply gradient accumulation.  
- 5 epoch đầu có thể giữ `--aug-level light` để model “nhìn rõ” ký tự HQ, sau đó nếu ổn định có thể nâng lên `full`.  
- Sau Phase 1, kiểm tra log trong `results/` (loss không NaN, acc bắt đầu >0%, CER giảm dần là tín hiệu tốt).

---

## Giai đoạn 2 – Geometric Rectification (STN Tuning) (10–15 epochs)

**Mục tiêu**  
- Bật STN để nắn thẳng biển số bị nghiêng, nhưng **hạn chế gây mất ổn định** (tránh loss âm).  
- Dùng **Focal CTC** để tập trung hơn vào mẫu khó.

**Cấu hình chính**  
- `USE_STN = True` (mặc định từ `config`, không dùng `--no-stn`).  
- `USE_SR = False`.  
- `USE_FOCAL_CTC = True` (set trong `config.py`: `USE_FOCAL_CTC = True`, `FOCAL_GAMMA = 2.0` nếu có).  
- `LEARNING_RATE ≈ 1e-4`.  
- Scheduler: nên chuyển sang `CosineAnnealing` (set trong `config.py`: `SCHEDULER_TYPE = 'cosine'`).  
- Trong optimizer (đã code sẵn):
  - `LR_MULT_BACKBONE ≈ 0.1` → Backbone học chậm.
  - Có thể thêm vào `config.py`: `LR_MULT_STN = 0.1`, `LR_MULT_HEAD = 1.0`.

**Lệnh gợi ý (fine-tune trên checkpoint Phase 1)**

Giả sử Phase 1 đã lưu checkpoint tốt nhất dưới tên `EXPERIMENT_NAME_best.pth` (trong `results/`).  
Bạn đặt `EXPERIMENT_NAME = phase1_warmup` trong `config.py` hoặc dùng `-n` giống hệt để load lại.

```bash
python train_optimized.py -n phase2_stn_tuning -m mf_svtrv2 --epochs 15 --batch-size 32 --learning-rate 0.0001
```

Gợi ý chỉnh trong `configs/config.py` trước khi chạy:
- `USE_STN = True`
- `USE_FOCAL_CTC = True`
- `SCHEDULER_TYPE = 'cosine'`
- (tuỳ chọn) `LR_MULT_BACKBONE = 0.1`, `LR_MULT_STN = 0.1`, `LR_MULT_HEAD = 1.0`

---

## Giai đoạn 3 – Deep Restoration Integration (LP-Diff) (15–20 epochs)

**Mục tiêu**  
- Kích hoạt mô-đun **MF_LPR_SR (LP-Diff)** để phục hồi đặc trưng từ 5 khung hình LQ, có tham chiếu từ 5 HQ.  
- Giảm CER đặc biệt trên các frame mờ/xấu.

**Cấu hình chính**  
- `USE_SR = True` (dùng `--use-sr`).  
- `SR_CHECKPOINT_PATH`: trỏ tới checkpoint GEN của LP-Diff (set trong `config.py` hoặc qua flag).  
- `SR_CONFIG_PATH`: thường là `sr_model/config/LP-Diff.json`.  
- `SR_N_TIMESTEP = 100` (override qua `--sr-n-timestep 100`).  
- `USE_FOCAL_CTC = True` (giữ từ Phase 2).  
- LR: giữ khoảng `1e-4` hoặc thấp hơn chút (tuỳ độ ổn định GPU).

**Lệnh gợi ý (fine-tune tiếp)**

```bash
python train_optimized.py -n phase3_sr_integration -m mf_svtrv2 --epochs 20 --batch-size 16 --learning-rate 0.0001 --use-sr --sr-checkpoint-path "sr_model/checkpoints/LP-Diff_GEN.pth" --sr-config-path "sr_model/config/LP-Diff.json" --sr-n-timestep 100
```

Ghi chú:
- Giảm `BATCH_SIZE` (vd 16) vì SR tốn VRAM hơn.  
- Kiểm tra log GPU và thời gian/epoch; nếu quá chậm, có thể hạ `SR_N_TIMESTEP` xuống 50.

---

## Giai đoạn 4 – Post-processing & Fine-tuning (5–10 epochs)

**Mục tiêu**  
- Fine-tune nhẹ với LR siêu thấp, tối ưu **acc > 85%** và CER.  
- Tăng beam width cho decoding, đồng thời chuẩn bị tích hợp rule-based filter theo Country ID.

**Cấu hình chính**  
- `LEARNING_RATE ≈ 5e-6`.  
- `SCHEDULER_TYPE = 'cosine'` hoặc giữ OneCycle nhưng với LR nhỏ.  
- `CTC_BEAM_WIDTH = 20` (set trong `config.py`).  
- `LABEL_SMOOTHING = 0.1`.  
- `DROPOUT ≈ 0.1` (giảm từ 0.3 để model học thêm chi tiết).

**Lệnh gợi ý (fine-tune cuối)**

```bash
python train_optimized.py -n phase4_finetune -m mf_svtrv2 --epochs 10 --batch-size 16 --learning-rate 0.000005
```

Sau Phase 4:
- Dùng script beam search (vd `test_beam.py`) với `CTC_BEAM_WIDTH=20`.  
- Áp dụng **rule-based filter theo Country ID** trong hậu xử lý (ví dụ trong `src/utils/postprocess.py` hoặc script đánh giá):
  - Nếu `country_id = 0` (Scenario-A) và kết quả beam top-1 thiếu prefix chữ cái, thử lấy ứng viên top-2/top-3 có prefix hợp lệ.

---

## Tóm tắt nhanh các tham số “vàng”

- **Optimizer**: AdamW (đã dùng trong `Trainer`, với param groups).
- **Main LR sau warm-up**: `1e-4`.  
- **STN LR hiệu dụng**: `~1e-5` (thông qua `LR_MULT_STN` nếu cần).  
- **Focal Gamma**: `γ = 2.0` (cấu hình trong `config.py` nếu hỗ trợ).  
- **Label Smoothing**: `0.1`.  
- **Gradient Clip**: `1.0` (có thể thêm vào `Trainer` nếu chưa có).  
- **Dropout (model)**: `~0.1` cho giai đoạn fine-tune.


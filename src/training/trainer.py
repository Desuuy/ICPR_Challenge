"""Trainer class encapsulating the training and validation loop."""
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.common import seed_everything
from src.utils.postprocess import decode_with_confidence, compute_cer
# GTC/SMTR label encoders từ OpenRec (sử dụng cùng vocab với EN_symbol_dict.txt).
# Hiện tại ta chỉ khởi tạo sẵn, bước sau mới dùng để lắp GTCLoss + GTCDecoder.
from src.openrec.preprocess.smtr_label_encode import SMTRLabelEncode
from src.openrec.preprocess.ctc_label_encode import CTCLabelEncode


class Trainer:
    """Encapsulates training, validation, and inference logic."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config,
        idx2char: Dict[int, str]
    ):
        """
        Args:
            model: The neural network model.
            train_loader: Training data loader.
            val_loader: Validation data loader (can be None).
            config: Configuration object with training parameters.
            idx2char: Index to character mapping for decoding.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.idx2char = idx2char
        self.device = config.DEVICE
        # Model nào có country_emb (MultiFrameSVTRv2) thì sẽ nhận thêm country_ids
        self.use_country_ids = hasattr(model, "country_emb")
        # Gradient accumulation để có effective batch lớn hơn
        self.accum_steps: int = max(1, int(getattr(config, "ACCUM_STEPS", 1)))
        seed_everything(config.SEED, benchmark=config.USE_CUDNN_BENCHMARK)

        # === Giai đoạn 1: Chuẩn bị encoder nhãn cho nhánh GTC/SMTR (chưa thay loss) ===
        # Khi USE_GTC=True, ta dùng lại logic encode của OpenRec: CTCLabelEncode + SMTRLabelEncode.
        # Mục tiêu: tạo ra đúng các tensor label/length/... mà GTCLoss / SMTRLoss và GTCDecoder mong đợi.
        self.use_gtc: bool = bool(getattr(config, "USE_GTC", False))
        self.ctc_encoder: Optional[CTCLabelEncode]
        self.smtr_encoder: Optional[SMTRLabelEncode]
        if self.use_gtc:
            max_len = int(getattr(config, "MAX_TEXT_LENGTH", 25))
            char_dict_path = getattr(config, "CHAR_DICT_PATH", None)
            # Encoder cho nhánh CTC trong GTCLoss (dùng cùng vocab file)
            self.ctc_encoder = CTCLabelEncode(
                max_text_length=max_len,
                character_dict_path=char_dict_path,
                use_space_char=False,
            )
            # Encoder cho nhánh GTC (SMTRLabelEncode): sinh thêm label_subs, label_next, ...
            self.smtr_encoder = SMTRLabelEncode(
                max_text_length=max_len,
                character_dict_path=char_dict_path,
                use_space_char=False,
                sub_str_len=5,
            )
        else:
            # Pipeline CTC hiện tại: chưa dùng GTC, nhưng giữ field None để code phía sau dễ kiểm tra.
            self.ctc_encoder = None
            self.smtr_encoder = None

        # Loss: focal-style CTC (sample-level weighting) or standard mean
        self.use_focal_ctc = getattr(config, 'USE_FOCAL_CTC', False)
        if self.use_focal_ctc:
            self.criterion = nn.CTCLoss(
                blank=0, zero_infinity=True, reduction='none')
        else:
            self.criterion = nn.CTCLoss(
                blank=0, zero_infinity=True, reduction='mean')
        self.criterion_val = nn.CTCLoss(
            blank=0, zero_infinity=True, reduction='mean')

        # Optimizer với 3 nhóm tham số: STN / Backbone / Fusion+Head+Country
        self.optimizer = self._build_optimizer(model, config)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.LEARNING_RATE,
            steps_per_epoch=len(train_loader),
            epochs=config.EPOCHS
        )

        # Train tiếp từ epoch có sẵn best acc
        """
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,  # Restart mỗi 10 epochs
            T_mult=2,  # Tăng gấp đôi chu kỳ
            eta_min=1e-6  # LR min
        )
        """
        self.scaler = GradScaler()

        # Label smoothing
        self.label_smoothing = getattr(config, 'LABEL_SMOOTHING', 0.1)

        # Tracking
        self.best_acc = 0.0
        self.current_epoch = 0
        self.history: List[Dict] = []  # Lưu lịch sử metrics mỗi epoch

    def _build_optimizer(self, model: nn.Module, config):
        """
        Khởi tạo AdamW với 3 nhóm tham số:
          - STN: học nhanh hơn (align tốt hơn)
          - Backbone: LR thấp (giữ ổn định pretrained)
          - Head + Fusion + Country embedding: LR cao hơn (học nhanh)
        """
        stn_params: List[nn.Parameter] = []
        backbone_params: List[nn.Parameter] = []
        head_params: List[nn.Parameter] = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'stn' in name:
                stn_params.append(param)
            elif 'backbone' in name:
                backbone_params.append(param)
            else:
                # fusion, country_emb, head, temp_scaling, ...
                head_params.append(param)

        base_lr = self.config.LEARNING_RATE
        # Có thể override bằng config nếu muốn tinh chỉnh
        lr_mult_stn = getattr(self.config, 'LR_MULT_STN', 1.0)
        lr_mult_backbone = getattr(self.config, 'LR_MULT_BACKBONE', 0.1)
        lr_mult_head = getattr(self.config, 'LR_MULT_HEAD', 1.0)

        param_groups = []
        if stn_params:
            param_groups.append(
                {"params": stn_params, "lr": base_lr * lr_mult_stn})
        if backbone_params:
            param_groups.append(
                {"params": backbone_params, "lr": base_lr * lr_mult_backbone})
        if head_params:
            param_groups.append(
                {"params": head_params, "lr": base_lr * lr_mult_head})

        return optim.AdamW(
            param_groups,
            weight_decay=config.WEIGHT_DECAY,
        )

    def _get_output_path(self, filename: str) -> str:
        """Get full path for output file in configured directory."""
        output_dir = getattr(self.config, 'OUTPUT_DIR', 'results')
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, filename)

    def _get_exp_name(self) -> str:
        """Get experiment name from config."""
        return getattr(self.config, 'EXPERIMENT_NAME', 'baseline')

    def train_one_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        skipped_nan = 0  # Đếm batch bị bỏ qua do loss NaN
        pbar = tqdm(
            self.train_loader,
            desc=f"Ep {self.current_epoch + 1}/{self.config.EPOCHS}",
        )
        accum_counter = 0  # Đếm số batch đã cộng dồn gradient
        # Debug: thống kê độ dài chuỗi / time-steps cho CTC
        min_input_len, max_input_len = None, None
        min_target_len, max_target_len = None, None
        debug_printed = 0

        for images, targets, target_lengths, _, _, _, country_ids in pbar:
            images = images.to(self.device)
            targets = targets.to(self.device)
            country_ids = country_ids.to(self.device)

            # Mixed precision chỉ dùng cho forward; loss/CTC tính ở float32 để tránh NaN
            with autocast('cuda'):
                if self.use_country_ids:
                    preds = self.model(images, country_ids)
                else:
                    preds = self.model(images)

            # Đảm bảo logits cho CTC ở float32 (ổn định hơn so với float16)
            # Expect: preds shape [B, T, Classes]
            preds = preds.float()
            # Chuẩn hoá logits: thay NaN/Inf bằng giá trị hữu hạn và giới hạn biên
            preds = torch.nan_to_num(preds, nan=0.0, posinf=20.0, neginf=-20.0)
            preds = torch.clamp(preds, min=-20.0, max=20.0)

            # Apply label smoothing BEFORE permute
            if self.label_smoothing > 0 and self.model.training:
                num_classes = preds.size(-1)
                # Smooth logits: (1-α)·pred + α/K
                preds = preds * (1 - self.label_smoothing) + \
                    self.label_smoothing / num_classes

            # Chuẩn hoá theo chiều lớp cho CTC: [B, T, Classes] -> log_probs
            preds_logs = preds.log_softmax(2)  # dim=2: Classes
            preds_permuted = preds_logs.permute(1, 0, 2)  # [T, B, Classes]
            input_lengths = torch.full(
                size=(images.size(0),),
                fill_value=preds.size(1),  # T
                dtype=torch.long
            )
            loss_per_sample = self.criterion(
                preds_permuted, targets, input_lengths, target_lengths)

            # Cập nhật thống kê lengths cho debug
            with torch.no_grad():
                batch_min_t = int(target_lengths.min().item())
                batch_max_t = int(target_lengths.max().item())
                t_in = int(input_lengths[0].item())
                min_input_len = t_in if min_input_len is None else min(min_input_len, t_in)
                max_input_len = t_in if max_input_len is None else max(max_input_len, t_in)
                min_target_len = batch_min_t if min_target_len is None else min(min_target_len, batch_min_t)
                max_target_len = batch_max_t if max_target_len is None else max(max_target_len, batch_max_t)

                # In vài ví dụ dự đoán ở những batch đầu epoch đầu để xem model có ra ký tự không
                if self.current_epoch == 0 and debug_printed < 2:
                    beam_width_dbg = getattr(self.config, "CTC_BEAM_WIDTH", 1)
                    decoded_dbg = decode_with_confidence(
                        preds_logs, self.idx2char, beam_width=beam_width_dbg
                    )
                    print("\n[TRAIN DEBUG] Sample predictions (epoch 1):")
                    for i in range(min(3, len(decoded_dbg))):
                        # Không có labels_text trong train loop, chỉ in độ dài target
                        pred_text, conf = decoded_dbg[i]
                        print(
                            f"  - target_len={target_lengths[i].item():2d} | pred='{pred_text}' | conf={conf:.4f}"
                        )
                    debug_printed += 1

            if self.use_focal_ctc:
                # CTC có thể trả inf cho sample lỗi; clamp để tránh nan
                loss_per_sample_safe = torch.nan_to_num(
                    loss_per_sample, nan=20.0, posinf=20.0, neginf=20.0)
                clamped = torch.clamp(loss_per_sample_safe, min=0.0, max=20.0)
                weight = (1 - torch.exp(-clamped)) ** 2
                loss = (loss_per_sample_safe * weight).mean()
            else:
                # Với CTC thường (reduction='mean'): biến NaN/Inf thành số hữu hạn và ép loss >= 0
                loss = torch.nan_to_num(
                    loss_per_sample, nan=20.0, posinf=20.0, neginf=20.0)
                loss = torch.clamp(loss, min=0.0)

            # Gradient accumulation: chia loss theo số bước tích lũy
            effective_loss = loss / self.accum_steps
            # Scale loss & backward
            self.scaler.scale(effective_loss).backward()
            accum_counter += 1

            # Khi đủ accum_steps hoặc là batch cuối cùng, mới step optimizer + scheduler
            if accum_counter % self.accum_steps == 0:
                # Unscale (required before gradient clipping)
                self.scaler.unscale_(self.optimizer)

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.GRAD_CLIP)

                # Step optimizer & update scaler
                scale_before = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.scaler.get_scale() >= scale_before:
                    self.scheduler.step()

                # Chuẩn bị cho nhóm batch tiếp theo
                self.optimizer.zero_grad(set_to_none=True)

            # Lưu epoch_loss theo loss gốc (chưa chia)
            epoch_loss += loss.item()
            pbar.set_postfix(
                {'loss': loss.item(), 'lr': self.scheduler.get_last_lr()[0]})

        # Nếu còn gradient dở dang (không chia hết cho accum_steps), step nốt + scheduler
        if accum_counter % self.accum_steps != 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.GRAD_CLIP)
            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scaler.get_scale() >= scale_before:
                self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

        if skipped_nan > 0:
            print(
                f"   ⚠️ Có {skipped_nan} batch loss ban đầu không hợp lệ (NaN/Inf) nhưng đã được chuẩn hoá bằng nan_to_num.")

        # In thống kê độ dài CTC cho toàn epoch
        if min_input_len is not None:
            print(
                f"   [CTC lengths] input_len: min={min_input_len}, max={max_input_len} | "
                f"target_len: min={min_target_len}, max={max_target_len}"
            )

        total_batches = len(self.train_loader)
        return epoch_loss / max(1, total_batches)

    def validate(self) -> Tuple[Dict[str, float], List[str], List[Tuple[str, str, str, float, str]]]:
        """Run validation and generate submission data.

        Returns:
            Tuple of (metrics_dict, submission_data, wrong_predictions).
            wrong_predictions: list of (track_id, ground_truth, prediction, confidence, img_paths_str).
        """
        if self.val_loader is None:
            return {'loss': 0.0, 'acc': 0.0, 'cer': 0.0}, [], []

        self.model.eval()
        val_loss = 0.0
        val_loss_count = 0  # Số batch có loss hợp lệ (không NaN)
        total_correct = 0
        total_samples = 0
        all_preds: List[str] = []
        all_targets: List[str] = []
        submission_data: List[str] = []
        wrong_predictions: List[Tuple[str, str, str, float, str]] = []

        with torch.no_grad():
            for images, targets, target_lengths, labels_text, track_ids, img_paths_batch, country_ids in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                country_ids = country_ids.to(self.device)
                if self.use_country_ids:
                    preds = self.model(images, country_ids)
                else:
                    preds = self.model(images)


                # Expect: preds [B, T, Classes]
                preds_logs = preds.log_softmax(2)  # dim=2: Classes

                input_lengths = torch.full(
                    (images.size(0),),
                    preds_logs.size(1),  # T
                    dtype=torch.long
                )
                loss = self.criterion_val(
                    preds_logs.permute(1, 0, 2),
                    targets,
                    input_lengths,
                    target_lengths
                )
                # Bỏ qua batch validation nếu loss NaN/inf (tránh val_loss=nan)
                if torch.isfinite(loss):
                    val_loss += loss.item()
                    val_loss_count += 1

                beam_width = getattr(self.config, 'CTC_BEAM_WIDTH', 1)
                decoded_list = decode_with_confidence(
                    preds_logs, self.idx2char, beam_width=beam_width)

                # Thống kê số lượng prediction rỗng để phát hiện collapse-to-blank
                for i, (pred_text, conf) in enumerate(decoded_list):
                    gt_text = labels_text[i]
                    track_id = track_ids[i]
                    img_paths_str = ";".join(list(img_paths_batch[i]))

                    all_preds.append(pred_text)
                    all_targets.append(gt_text)

                    if pred_text == gt_text:
                        total_correct += 1
                    else:
                        wrong_predictions.append(
                            (track_id, gt_text, pred_text, conf, img_paths_str))

                    submission_data.append(
                        f"{track_id},{pred_text};{conf:.4f}")

                total_samples += len(labels_text)

        # Đếm số prediction rỗng để debug
        num_blank = sum(1 for p in all_preds if p == "")
        if total_samples > 0:
            print(
                f"   [VAL DEBUG] blank predictions: {num_blank}/{total_samples} "
                f"({num_blank / total_samples * 100:.2f}%)"
            )

        avg_val_loss = val_loss / \
            val_loss_count if val_loss_count > 0 else float('nan')
        val_acc = (total_correct / total_samples) * \
            100 if total_samples > 0 else 0.0
        val_cer = compute_cer(all_preds, list(all_targets))

        metrics = {
            'loss': avg_val_loss,
            'acc': val_acc,
            'cer': val_cer,
            'correct': total_correct,
            'total': total_samples,
            'preds': all_preds,
            'targets': all_targets,
        }

        return metrics, submission_data, wrong_predictions

    def save_submission(self, submission_data: List[str]) -> None:
        """Save submission file with experiment name."""
        exp_name = self._get_exp_name()
        filename = self._get_output_path(f"submission_{exp_name}.txt")
        with open(filename, 'w') as f:
            f.write("\n".join(submission_data))
        print(f"📝 Saved {len(submission_data)} lines to {filename}")

    def save_wrong_predictions(
        self,
        wrong_predictions: List[Tuple[str, str, str, float, str]],
    ) -> None:
        """Save wrong predictions list (track_id, gt, pred, conf, img_paths) for analysis."""
        if not wrong_predictions:
            return
        exp_name = self._get_exp_name()
        filename = self._get_output_path(f"wrong_predictions_{exp_name}.txt")
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("track_id\tground_truth\tprediction\tconfidence\timg_paths\n")
            for track_id, gt, pred, conf, img_paths_str in wrong_predictions:
                gt_s = gt.replace('\t', ' ').replace('\n', ' ')
                pred_s = pred.replace('\t', ' ').replace('\n', ' ')
                img_s = img_paths_str.replace('\t', ' ').replace('\n', ' ')
                f.write(f"{track_id}\t{gt_s}\t{pred_s}\t{conf:.4f}\t{img_s}\n")
        print(
            f"📋 Saved {len(wrong_predictions)} wrong predictions to {filename}")

    def save_wrong_images(
        self,
        wrong_predictions: List[Tuple[str, str, str, float, str]],
    ) -> None:
        """Copy wrong-prediction images to results/wrong_images/{exp}/{track_id}/ for inspection."""
        if not wrong_predictions:
            return
        import shutil
        exp_name = self._get_exp_name()
        out_dir = self._get_output_path(f"wrong_images_{exp_name}")
        os.makedirs(out_dir, exist_ok=True)
        copied = 0
        for track_id, gt, pred, conf, img_paths_str in wrong_predictions:
            paths = img_paths_str.split(";")
            track_dir = os.path.join(out_dir, f"{track_id}_gt{gt}_pred{pred}")
            os.makedirs(track_dir, exist_ok=True)
            for i, src in enumerate(paths):
                if os.path.exists(src):
                    ext = os.path.splitext(src)[1]
                    dst = os.path.join(track_dir, f"frame_{i}{ext}")
                    shutil.copy2(src, dst)
                    copied += 1
        if copied > 0:
            print(f"📁 Copied {copied} wrong images to {out_dir}")

    def save_model(self, path: str = None) -> None:
        """Save model checkpoint with experiment name."""
        if path is None:
            exp_name = self._get_exp_name()
            path = self._get_output_path(f"{exp_name}_best.pth")
        torch.save(self.model.state_dict(), path)

    def fit(self) -> None:
        """Run the full training loop for specified number of epochs."""
        loss_type = "Focal CTC" if self.use_focal_ctc else "CTC"
        print(
            f"🚀 TRAINING START | Device: {self.device} | Epochs: {self.config.EPOCHS} | Loss: {loss_type}")

        for epoch in range(self.config.EPOCHS):
            self.current_epoch = epoch

            # Training
            avg_train_loss = self.train_one_epoch()

            # Validation
            val_metrics, submission_data, wrong_predictions = self.validate()
            val_loss = val_metrics['loss']
            val_acc = val_metrics['acc']
            val_cer = val_metrics.get('cer', 0.0)
            val_correct = val_metrics.get('correct', 0)
            val_total = val_metrics.get('total', 0)
            current_lr = self.scheduler.get_last_lr()[0]

            # Lưu history
            epoch_record = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss if not (isinstance(avg_train_loss, float) and math.isnan(avg_train_loss)) else None,
                'val_loss': val_loss if not (isinstance(val_loss, float) and math.isnan(val_loss)) else None,
                'val_acc': val_acc,
                'val_cer': val_cer,
                'val_correct': val_correct,
                'val_total': val_total,
                'lr': current_lr,
            }
            self.history.append(epoch_record)

            # Cảnh báo nếu val_total = 0 (có thể val split sai)
            if val_total == 0 and self.val_loader:
                print(f"   ⚠️ Val total=0! Kiểm tra val_tracks.json và DATA_ROOT.")

            # Log results (thêm correct/total để debug val)
            print(f"Epoch {epoch + 1}/{self.config.EPOCHS}: "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.2f}% ({val_correct}/{val_total}) | "
                  f"Val CER: {val_cer:.4f} | "
                  f"LR: {current_lr:.2e}")

            # Save best model (by val accuracy)
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.save_model()
                exp_name = self._get_exp_name()
                model_path = self._get_output_path(f"{exp_name}_best.pth")
                print(f"  ⭐ Saved Best Model: {model_path} ({val_acc:.2f}%)")

            # Always save submission and wrong_predictions every epoch (for analysis, even when val_acc=0%)
            if submission_data:
                self.save_submission(submission_data)
            if getattr(self.config, 'SAVE_WRONG_PREDICTIONS', True) and wrong_predictions:
                self.save_wrong_predictions(wrong_predictions)
                if getattr(self.config, 'SAVE_WRONG_IMAGES', True):
                    self.save_wrong_images(wrong_predictions)

        # Lưu training history
        exp_name = self._get_exp_name()
        history_path = self._get_output_path(
            f"training_history_{exp_name}.json")
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump({
                'experiment': exp_name,
                'best_val_acc': self.best_acc,
                'epochs': self.config.EPOCHS,
                'history': self.history,
            }, f, indent=2, ensure_ascii=False)
        print(f"📊 Saved training history to {history_path}")

        # Luôn lưu .pth khi chạy xong (cả SUBMISSION_MODE True/False)
        if self.val_loader is None:
            # Submission mode: best đã là model cuối, lưu vào _best.pth
            self.save_model()
            model_path = self._get_output_path(f"{exp_name}_best.pth")
            print(f"  💾 Saved model: {model_path}")
        else:
            # Normal mode: lưu thêm bản cuối cùng vào _final.pth
            final_path = self._get_output_path(f"{exp_name}_final.pth")
            torch.save(self.model.state_dict(), final_path)
            print(f"  💾 Saved final model: {final_path}")

        print(f"\n✅ Training complete! Best Val Acc: {self.best_acc:.2f}%")
        if self.history and self.val_loader:
            last = self.history[-1]
            print(
                f"   Val: {last.get('val_correct', 0)}/{last.get('val_total', 0)} correct")

    def predict(self, loader: DataLoader) -> List[Tuple[str, str, float]]:
        """Run inference on a data loader.

        Returns:
            List of (track_id, predicted_text, confidence) tuples.
        """
        self.model.eval()
        results: List[Tuple[str, str, float]] = []

        with torch.no_grad():
            for images, _, _, _, track_ids, _, country_ids in loader:
                images = images.to(self.device)
                country_ids = country_ids.to(self.device)
                if self.use_country_ids:
                    preds = self.model(images, country_ids)
                else:
                    preds = self.model(images)

                preds_logs = preds.log_softmax(2)

                beam_width = getattr(self.config, 'CTC_BEAM_WIDTH', 1)
                decoded_list = decode_with_confidence(
                    preds_logs, self.idx2char, beam_width=beam_width)
                
                for i, (pred_text, conf) in enumerate(decoded_list):
                    results.append((track_ids[i], pred_text, conf))

        return results

    def predict_test(self, test_loader: DataLoader, output_filename: str = "submission_final.txt") -> None:
        """Run inference on test data and save submission file.

        Args:
            test_loader: DataLoader for test data.
            output_filename: Name of the submission file to save.
        """
        print(f"🔮 Running inference on test data...")

        # Use existing predict method
        results = []
        self.model.eval()
        with torch.no_grad():
            for images, _, _, _, track_ids, _, country_ids in tqdm(test_loader, desc="Test Inference"):
                images = images.to(self.device)
                country_ids = country_ids.to(self.device)
                preds = self.model(images, country_ids)
                preds_logs = preds.log_softmax(2)
                beam_width = getattr(self.config, 'CTC_BEAM_WIDTH', 1)
                decoded_list = decode_with_confidence(
                    preds, self.idx2char, beam_width=beam_width)

                for i, (pred_text, conf) in enumerate(decoded_list):
                    results.append((track_ids[i], pred_text, conf))

        # Format and save submission file
        submission_data = [
            f"{track_id},{pred_text};{conf:.4f}" for track_id, pred_text, conf in results]
        output_path = self._get_output_path(output_filename)
        with open(output_path, 'w') as f:
            f.write("\n".join(submission_data))

        print(f"Saved {len(submission_data)} predictions to {output_path}")

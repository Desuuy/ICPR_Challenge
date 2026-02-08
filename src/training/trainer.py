"""Trainer class encapsulating the training and validation loop."""
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
        seed_everything(config.SEED, benchmark=config.USE_CUDNN_BENCHMARK)

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
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.LEARNING_RATE,
            steps_per_epoch=len(train_loader),
            epochs=config.EPOCHS
        )
        self.scaler = GradScaler()

        # Tracking
        self.best_acc = 0.0
        self.current_epoch = 0

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
        pbar = tqdm(self.train_loader,
                    desc=f"Ep {self.current_epoch + 1}/{self.config.EPOCHS}")
        prev_optimizer_stepped = False

        for images, targets, target_lengths, _, _, _ in pbar:
            # Gọi scheduler.step() sau optimizer.step() của batch trước (tránh warning PyTorch)
            if prev_optimizer_stepped:
                self.scheduler.step()
            prev_optimizer_stepped = False

            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast('cuda'):
                preds = self.model(images)
                preds_permuted = preds.permute(1, 0, 2)
                input_lengths = torch.full(
                    size=(images.size(0),),
                    fill_value=preds.size(1),
                    dtype=torch.long
                )
                loss_per_sample = self.criterion(
                    preds_permuted, targets, input_lengths, target_lengths)
                if self.use_focal_ctc:
                    # CTC có thể trả inf cho sample lỗi; clamp để tránh nan
                    loss_per_sample_safe = torch.clamp(
                        loss_per_sample, min=-20.0, max=20.0)
                    loss_per_sample_safe = torch.nan_to_num(
                        loss_per_sample_safe, nan=20.0, posinf=20.0, neginf=-20.0)
                    clamped = loss_per_sample_safe
                    weight = (1 - torch.exp(-clamped)) ** 2
                    loss = (loss_per_sample_safe * weight).mean()
                else:
                    loss = loss_per_sample

            # Bỏ qua batch nếu loss nan/inf (tránh làm hỏng trọng số)
            if not torch.isfinite(loss).all():
                skipped_nan += 1
                pbar.set_postfix(
                    {'loss': 'nan(skip)', 'lr': self.scheduler.get_last_lr()[0]})
                continue

            # Scale loss & backward
            self.scaler.scale(loss).backward()

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
                prev_optimizer_stepped = True

            epoch_loss += loss.item()
            pbar.set_postfix(
                {'loss': loss.item(), 'lr': self.scheduler.get_last_lr()[0]})

        # Scheduler step cho batch cuối cùng (đã gọi optimizer.step() ở trên)
        if prev_optimizer_stepped:
            self.scheduler.step()

        if skipped_nan > 0:
            print(f"   ⚠️ Skipped {skipped_nan}/{len(self.train_loader)} batches (loss NaN). "
                  "Thử --no-pretrained hoặc giảm LEARNING_RATE.")

        valid_batches = len(self.train_loader) - skipped_nan
        return epoch_loss / valid_batches if valid_batches > 0 else float('nan')

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
            for images, targets, target_lengths, labels_text, track_ids, img_paths_batch in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                preds = self.model(images)

                input_lengths = torch.full(
                    (images.size(0),),
                    preds.size(1),
                    dtype=torch.long
                )
                loss = self.criterion_val(
                    preds.permute(1, 0, 2),
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
                    preds, self.idx2char, beam_width=beam_width)

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

        avg_val_loss = val_loss / val_loss_count if val_loss_count > 0 else float('nan')
        val_acc = (total_correct / total_samples) * \
            100 if total_samples > 0 else 0.0
        val_cer = compute_cer(all_preds, list(all_targets))

        metrics = {
            'loss': avg_val_loss,
            'acc': val_acc,
            'cer': val_cer,
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
        print(f"📋 Saved {len(wrong_predictions)} wrong predictions to {filename}")

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
            current_lr = self.scheduler.get_last_lr()[0]

            # Log results
            print(f"Epoch {epoch + 1}/{self.config.EPOCHS}: "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.2f}% | "
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

        # Luôn lưu .pth khi chạy xong (cả SUBMISSION_MODE True/False)
        exp_name = self._get_exp_name()
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

    def predict(self, loader: DataLoader) -> List[Tuple[str, str, float]]:
        """Run inference on a data loader.

        Returns:
            List of (track_id, predicted_text, confidence) tuples.
        """
        self.model.eval()
        results: List[Tuple[str, str, float]] = []

        with torch.no_grad():
            for images, _, _, _, track_ids, _ in loader:
                images = images.to(self.device)
                preds = self.model(images)

                beam_width = getattr(self.config, 'CTC_BEAM_WIDTH', 1)
                decoded_list = decode_with_confidence(
                    preds, self.idx2char, beam_width=beam_width)
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
            for images, _, _, _, track_ids, _ in tqdm(test_loader, desc="Test Inference"):
                images = images.to(self.device)
                preds = self.model(images)
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

        print(f"✅ Saved {len(submission_data)} predictions to {output_path}")

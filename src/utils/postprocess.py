"""Post-processing utilities for OCR decoding."""
import math
from itertools import groupby
from typing import Dict, List, Tuple

import numpy as np
import torch


def _cer_edit_distance(pred: str, target: str) -> int:
    """Levenshtein distance (number of edits: insert, delete, substitute)."""
    m, n = len(pred), len(target)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if pred[i - 1] == target[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] +
                           1, dp[i - 1][j - 1] + cost)
    return dp[m][n]


def compute_cer(predictions: List[str], targets: List[str]) -> float:
    """
    Character Error Rate = total_edits / total_characters_in_targets.
    Returns value in [0, 1+]; 0 = perfect.
    """
    if not targets:
        return 0.0
    total_edits = sum(_cer_edit_distance(p, t)
                      for p, t in zip(predictions, targets))
    total_chars = sum(len(t) for t in targets)
    return total_edits / total_chars if total_chars > 0 else 0.0


def decode_ctc_beam(
    log_probs: torch.Tensor,
    idx2char: Dict[int, str],
    blank: int = 0,
    beam_width: int = 5,
) -> List[Tuple[str, float]]:
    """
    CTC beam search decode. log_probs: [B, T, C] in log space.
    Returns list of (decoded_string, score) where score is mean log-prob of the path.
    """
    B, T, C = log_probs.shape
    log_probs_np = log_probs.detach().cpu().float().numpy()
    results: List[Tuple[str, float]] = []

    for b in range(B):
        beams: Dict[Tuple[int, ...], float] = {(): 0.0}
        for t in range(T):
            next_beams: Dict[Tuple[int, ...], float] = {}
            for prefix, score in beams.items():
                for c in range(C):
                    new_score = score + float(log_probs_np[b, t, c])
                    if c == blank:
                        key = prefix
                    else:
                        key = prefix + (c,)
                    if key not in next_beams:
                        next_beams[key] = new_score
                    else:
                        next_beams[key] = math.log(
                            math.exp(next_beams[key]) + math.exp(new_score))
            beams = dict(sorted(next_beams.items(),
                         key=lambda x: -x[1])[:beam_width])

        best_prefix = max(beams.keys(), key=lambda k: beams[k])
        best_score = beams[best_prefix]
        chars = [idx2char.get(idx, "")
                 for idx in best_prefix if idx != blank and idx in idx2char]
        pred_str = "".join(chars)
        results.append((pred_str, best_score / max(T, 1)))
    return results


def decode_with_confidence(
    preds: torch.Tensor,
    idx2char: Dict[int, str],
    beam_width: int = 1,
) -> List[Tuple[str, float]]:
    """CTC decode: greedy (beam_width=1) or beam search (beam_width>1).

    Args:
        preds: Log-softmax predictions [batch_size, time_steps, num_classes].
        idx2char: Index to character mapping.
        beam_width: 1 = greedy; >1 = beam search for better accuracy.

    Returns:
        List of (predicted_string, confidence_score) tuples.
    """
    if beam_width > 1:
        return decode_ctc_beam(preds, idx2char, blank=0, beam_width=beam_width)

    probs = preds.exp()
    max_probs, indices = probs.max(dim=2)
    indices_np = indices.detach().cpu().numpy()
    max_probs_np = max_probs.detach().cpu().numpy()

    batch_size, time_steps = indices_np.shape
    results: List[Tuple[str, float]] = []

    for batch_idx in range(batch_size):
        path = indices_np[batch_idx]
        probs_b = max_probs_np[batch_idx]

        # Group consecutive identical characters and filter blanks
        # groupby returns (key, group_iterator) pairs
        pred_chars = []
        confidences = []
        time_idx = 0

        for char_idx, group in groupby(path):
            group_list = list(group)
            group_size = len(group_list)

            if char_idx != 0:  # Skip blank
                pred_chars.append(idx2char.get(char_idx, ''))
                # Get maximum probability from this group
                group_probs = probs_b[time_idx:time_idx + group_size]
                confidences.append(float(np.max(group_probs)))

            time_idx += group_size

        pred_str = "".join(pred_chars)
        confidence = float(np.mean(confidences)) if confidences else 0.0
        results.append((pred_str, confidence))

    return results

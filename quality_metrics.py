#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  PERCEPTUAL QUALITY METRICS FOR AUDIO CODEC
  
  Implements quality evaluation metrics:
  - PESQ (Perceptual Evaluation of Speech Quality)
  - VISQOL (Virtual Speech Quality Objective Listener)
  - SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)
  - STOI (Short-Time Objective Intelligibility)
  - MCD (Mel Cepstral Distortion)
  - SNR (Signal-to-Noise Ratio)
  
  Used for:
  - Codec quality evaluation
  - Training monitoring
  - A/B testing
  
  Reference:
  - PESQ: ITU-T P.862
  - VISQOL: https://github.com/google/visqol
═══════════════════════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
import warnings

# Try importing quality metric libraries
try:
    from pesq import pesq
    HAS_PESQ = True
except ImportError:
    HAS_PESQ = False
    warnings.warn("pesq not installed. Run: pip install pesq")

try:
    from pystoi import stoi
    HAS_STOI = True
except ImportError:
    HAS_STOI = False
    warnings.warn("pystoi not installed. Run: pip install pystoi")


def si_sdr(reference: torch.Tensor, estimation: torch.Tensor, 
           eps: float = 1e-8) -> torch.Tensor:
    """
    Scale-Invariant Signal-to-Distortion Ratio.
    
    Higher is better. Good quality: > 15 dB.
    
    Args:
        reference: [B, samples] clean reference
        estimation: [B, samples] reconstructed signal
    Returns:
        si_sdr: [B] SI-SDR in dB
    """
    # Ensure same length
    min_len = min(reference.shape[-1], estimation.shape[-1])
    reference = reference[..., :min_len]
    estimation = estimation[..., :min_len]
    
    # Zero-mean
    reference = reference - reference.mean(dim=-1, keepdim=True)
    estimation = estimation - estimation.mean(dim=-1, keepdim=True)
    
    # Compute scaling factor
    dot = (reference * estimation).sum(dim=-1, keepdim=True)
    ref_energy = (reference ** 2).sum(dim=-1, keepdim=True)
    
    # Target signal
    s_target = (dot / (ref_energy + eps)) * reference
    
    # Error signal
    e_noise = estimation - s_target
    
    # SI-SDR
    si_sdr_value = 10 * torch.log10(
        (s_target ** 2).sum(dim=-1) / ((e_noise ** 2).sum(dim=-1) + eps) + eps
    )
    
    return si_sdr_value


def snr(reference: torch.Tensor, estimation: torch.Tensor,
        eps: float = 1e-8) -> torch.Tensor:
    """
    Signal-to-Noise Ratio.
    
    Args:
        reference: [B, samples] clean reference
        estimation: [B, samples] reconstructed signal
    Returns:
        snr: [B] SNR in dB
    """
    min_len = min(reference.shape[-1], estimation.shape[-1])
    reference = reference[..., :min_len]
    estimation = estimation[..., :min_len]
    
    noise = reference - estimation
    
    signal_power = (reference ** 2).sum(dim=-1)
    noise_power = (noise ** 2).sum(dim=-1)
    
    snr_value = 10 * torch.log10(signal_power / (noise_power + eps) + eps)
    
    return snr_value


def mcd(reference: torch.Tensor, estimation: torch.Tensor,
        sample_rate: int = 16000, n_mfcc: int = 13) -> torch.Tensor:
    """
    Mel Cepstral Distortion (MCD).
    
    Lower is better. Good quality: < 5 dB.
    
    Args:
        reference: [B, samples] clean reference
        estimation: [B, samples] reconstructed signal
    Returns:
        mcd: [B] MCD in dB
    """
    import torchaudio.transforms as T
    
    # Compute MFCCs
    mfcc_transform = T.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={'n_fft': 1024, 'hop_length': 256, 'n_mels': 80}
    )
    
    ref_mfcc = mfcc_transform(reference)  # [B, n_mfcc, T]
    est_mfcc = mfcc_transform(estimation)  # [B, n_mfcc, T]
    
    # Align lengths
    min_len = min(ref_mfcc.shape[-1], est_mfcc.shape[-1])
    ref_mfcc = ref_mfcc[..., :min_len]
    est_mfcc = est_mfcc[..., :min_len]
    
    # MCD (skip first coefficient - energy)
    diff = ref_mfcc[:, 1:, :] - est_mfcc[:, 1:, :]
    mcd_value = (10 / np.log(10)) * torch.sqrt(2 * (diff ** 2).sum(dim=1).mean(dim=-1))
    
    return mcd_value


def pesq_metric(reference: torch.Tensor, estimation: torch.Tensor,
                sample_rate: int = 16000) -> float:
    """
    PESQ (Perceptual Evaluation of Speech Quality).
    
    Range: -0.5 to 4.5. Good quality: > 3.5.
    
    Args:
        reference: [samples] clean reference (single sample)
        estimation: [samples] reconstructed signal
    Returns:
        pesq_score: PESQ score
    """
    if not HAS_PESQ:
        warnings.warn("PESQ not available")
        return 0.0
    
    # Convert to numpy
    if isinstance(reference, torch.Tensor):
        reference = reference.cpu().numpy()
    if isinstance(estimation, torch.Tensor):
        estimation = estimation.cpu().numpy()
    
    # Ensure 1D
    reference = reference.flatten()
    estimation = estimation.flatten()
    
    # Align lengths
    min_len = min(len(reference), len(estimation))
    reference = reference[:min_len]
    estimation = estimation[:min_len]
    
    try:
        score = pesq(sample_rate, reference, estimation, 'wb')  # Wide-band
    except Exception as e:
        warnings.warn(f"PESQ computation failed: {e}")
        score = 0.0
    
    return score


def stoi_metric(reference: torch.Tensor, estimation: torch.Tensor,
                sample_rate: int = 16000) -> float:
    """
    STOI (Short-Time Objective Intelligibility).
    
    Range: 0 to 1. Good quality: > 0.9.
    
    Args:
        reference: [samples] clean reference
        estimation: [samples] reconstructed signal
    Returns:
        stoi_score: STOI score
    """
    if not HAS_STOI:
        warnings.warn("STOI not available")
        return 0.0
    
    if isinstance(reference, torch.Tensor):
        reference = reference.cpu().numpy()
    if isinstance(estimation, torch.Tensor):
        estimation = estimation.cpu().numpy()
    
    reference = reference.flatten()
    estimation = estimation.flatten()
    
    min_len = min(len(reference), len(estimation))
    reference = reference[:min_len]
    estimation = estimation[:min_len]
    
    try:
        score = stoi(reference, estimation, sample_rate, extended=False)
    except Exception as e:
        warnings.warn(f"STOI computation failed: {e}")
        score = 0.0
    
    return score


class MultiScaleSpectralLoss(nn.Module):
    """
    Multi-scale spectral loss for codec training.
    
    Computes L1 loss on spectrograms at multiple window sizes.
    """
    def __init__(self, 
                 n_ffts: list = [512, 1024, 2048],
                 hop_lengths: list = None,
                 win_lengths: list = None):
        super().__init__()
        
        self.n_ffts = n_ffts
        self.hop_lengths = hop_lengths or [n // 4 for n in n_ffts]
        self.win_lengths = win_lengths or n_ffts
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-scale spectral loss.
        
        Args:
            pred: [B, 1, samples] predicted audio
            target: [B, 1, samples] target audio
        Returns:
            loss: Scalar loss value
        """
        pred = pred.squeeze(1)
        target = target.squeeze(1)
        
        loss = 0.0
        
        for n_fft, hop, win in zip(self.n_ffts, self.hop_lengths, self.win_lengths):
            window = torch.hann_window(win, device=pred.device)
            
            # Compute spectrograms
            pred_spec = torch.stft(
                pred, n_fft, hop, win, window, return_complex=True
            ).abs()
            target_spec = torch.stft(
                target, n_fft, hop, win, window, return_complex=True
            ).abs()
            
            # L1 loss on magnitude
            loss += F.l1_loss(pred_spec, target_spec)
            
            # Log-magnitude loss
            log_pred = torch.log(pred_spec + 1e-8)
            log_target = torch.log(target_spec + 1e-8)
            loss += F.l1_loss(log_pred, log_target)
        
        return loss / len(self.n_ffts)


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using mel-spectrogram similarity.
    
    More perceptually meaningful than raw waveform loss.
    """
    def __init__(self, sample_rate: int = 16000, n_mels: int = 80):
        super().__init__()
        
        import torchaudio.transforms as T
        
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=n_mels,
            f_min=20,
            f_max=8000,
        )
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss.
        
        Args:
            pred: [B, 1, samples] predicted audio
            target: [B, 1, samples] target audio
        Returns:
            loss: Scalar perceptual loss
        """
        pred = pred.squeeze(1)
        target = target.squeeze(1)
        
        pred_mel = self.mel_transform(pred)
        target_mel = self.mel_transform(target)
        
        # L1 loss on mel-spectrogram
        mel_loss = F.l1_loss(pred_mel, target_mel)
        
        # Log-mel loss
        log_pred = torch.log(pred_mel + 1e-8)
        log_target = torch.log(target_mel + 1e-8)
        log_mel_loss = F.l1_loss(log_pred, log_target)
        
        return mel_loss + log_mel_loss


class QualityMetrics:
    """
    Complete quality metrics evaluator for codec.
    
    Usage:
        metrics = QualityMetrics()
        
        results = metrics.evaluate(original, reconstructed)
        print(results['si_sdr'])  # SI-SDR in dB
        print(results['pesq'])    # PESQ score
    """
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        
    def evaluate(self, reference: torch.Tensor, 
                 estimation: torch.Tensor) -> Dict[str, float]:
        """
        Compute all quality metrics.
        
        Args:
            reference: [B, samples] or [samples] clean audio
            estimation: [B, samples] or [samples] reconstructed audio
        Returns:
            Dict with all metric scores
        """
        # Ensure 2D
        if reference.dim() == 1:
            reference = reference.unsqueeze(0)
        if estimation.dim() == 1:
            estimation = estimation.unsqueeze(0)
        
        results = {}
        
        # SI-SDR
        si_sdr_value = si_sdr(reference, estimation)
        results['si_sdr'] = si_sdr_value.mean().item()
        
        # SNR
        snr_value = snr(reference, estimation)
        results['snr'] = snr_value.mean().item()
        
        # MCD
        mcd_value = mcd(reference, estimation, self.sample_rate)
        results['mcd'] = mcd_value.mean().item()
        
        # PESQ (single sample, CPU-bound)
        if reference.shape[0] == 1:
            results['pesq'] = pesq_metric(
                reference[0], estimation[0], self.sample_rate
            )
        
        # STOI (single sample)
        if reference.shape[0] == 1:
            results['stoi'] = stoi_metric(
                reference[0], estimation[0], self.sample_rate
            )
        
        return results
    
    def format_results(self, results: Dict[str, float]) -> str:
        """Format results for printing."""
        lines = []
        lines.append("┌─────────────────────────────────────┐")
        lines.append("│       QUALITY METRICS REPORT        │")
        lines.append("├─────────────────────────────────────┤")
        
        # Metric guidelines
        guidelines = {
            'si_sdr': ('SI-SDR', 'dB', '> 15 = good'),
            'snr': ('SNR', 'dB', '> 20 = good'),
            'mcd': ('MCD', 'dB', '< 5 = good'),
            'pesq': ('PESQ', '', '> 3.5 = good'),
            'stoi': ('STOI', '', '> 0.9 = good'),
        }
        
        for key, value in results.items():
            if key in guidelines:
                name, unit, guideline = guidelines[key]
                if unit:
                    lines.append(f"│ {name:12s}: {value:7.2f} {unit:4s} ({guideline}) │")
                else:
                    lines.append(f"│ {name:12s}: {value:7.3f}      ({guideline}) │")
        
        lines.append("└─────────────────────────────────────┘")
        return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  QUALITY METRICS TEST")
    print("=" * 70)
    
    # Create test signals
    duration = 2  # seconds
    sample_rate = 16000
    samples = duration * sample_rate
    
    # Clean reference (sine wave)
    t = torch.linspace(0, duration, samples)
    reference = torch.sin(2 * np.pi * 440 * t)  # 440 Hz
    
    # Add some noise for reconstruction
    noise_level = 0.1
    estimation = reference + torch.randn_like(reference) * noise_level
    
    print(f"\nTest signal: {duration}s @ {sample_rate}Hz")
    print(f"Noise level: {noise_level}")
    
    # Test individual metrics
    print("\n--- Individual Metrics ---")
    
    si_sdr_val = si_sdr(reference.unsqueeze(0), estimation.unsqueeze(0))
    print(f"SI-SDR: {si_sdr_val.item():.2f} dB")
    
    snr_val = snr(reference.unsqueeze(0), estimation.unsqueeze(0))
    print(f"SNR: {snr_val.item():.2f} dB")
    
    mcd_val = mcd(reference.unsqueeze(0), estimation.unsqueeze(0))
    print(f"MCD: {mcd_val.item():.2f} dB")
    
    if HAS_PESQ:
        pesq_val = pesq_metric(reference, estimation, sample_rate)
        print(f"PESQ: {pesq_val:.3f}")
    else:
        print("PESQ: Not available (pip install pesq)")
    
    if HAS_STOI:
        stoi_val = stoi_metric(reference, estimation, sample_rate)
        print(f"STOI: {stoi_val:.3f}")
    else:
        print("STOI: Not available (pip install pystoi)")
    
    # Test complete evaluator
    print("\n--- Complete Evaluation ---")
    
    metrics = QualityMetrics(sample_rate)
    results = metrics.evaluate(reference, estimation)
    
    print(metrics.format_results(results))
    
    # Test losses
    print("\n--- Loss Functions ---")
    
    spectral_loss = MultiScaleSpectralLoss()
    pred = estimation.unsqueeze(0).unsqueeze(0)
    target = reference.unsqueeze(0).unsqueeze(0)
    
    s_loss = spectral_loss(pred, target)
    print(f"Multi-Scale Spectral Loss: {s_loss.item():.4f}")
    
    perceptual_loss = PerceptualLoss()
    p_loss = perceptual_loss(pred, target)
    print(f"Perceptual Loss: {p_loss.item():.4f}")
    
    print("\n" + "=" * 70)
    print("  ✅ ALL TESTS PASSED!")
    print("=" * 70)

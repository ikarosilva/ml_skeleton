"""Audio augmentation pipeline for SimSiam self-supervised learning.

This module provides audio augmentations that create different "views" of the
same audio for contrastive learning. SimSiam learns by maximizing agreement
between two augmented views of the same sample.

Augmentation Types:
1. Waveform-level: Time shift, time stretch, pitch shift, noise injection
2. Spectrogram-level (SpecAugment): Frequency masking, time masking

Usage:
    from ml_skeleton.music.augmentations import create_audio_augmentor

    augmentor = create_audio_augmentor(config, sample_rate=16000)
    view1 = augmentor(waveform)  # First augmented view
    view2 = augmentor(waveform)  # Second augmented view (different random params)
"""

import random
from typing import Optional
import torch
import torchaudio
import torchaudio.transforms as T


class AudioAugmentor:
    """Audio augmentation pipeline for SimSiam training.

    Creates augmented views of audio waveforms for self-supervised learning.
    Each call produces a different random augmentation.

    Args:
        sample_rate: Audio sample rate in Hz
        time_stretch_range: Range for time stretching [min, max] (e.g., [0.8, 1.2])
        pitch_shift_range: Range for pitch shifting in semitones [min, max]
        noise_snr_range: Range for noise SNR in dB [min, max]
        volume_change_db: Maximum volume change in dB (applied ±)
        time_shift_prob: Probability of applying time shift
        time_stretch_prob: Probability of applying time stretch
        pitch_shift_prob: Probability of applying pitch shift
        noise_prob: Probability of adding noise
        volume_prob: Probability of changing volume
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        time_stretch_range: tuple[float, float] = (0.8, 1.2),
        pitch_shift_range: tuple[float, float] = (-2, 2),
        noise_snr_range: tuple[float, float] = (15, 30),
        volume_change_db: float = 6.0,
        time_shift_prob: float = 1.0,
        time_stretch_prob: float = 0.5,
        pitch_shift_prob: float = 0.5,
        noise_prob: float = 0.3,
        volume_prob: float = 0.8
    ):
        self.sample_rate = sample_rate
        self.time_stretch_range = time_stretch_range
        self.pitch_shift_range = pitch_shift_range
        self.noise_snr_range = noise_snr_range
        self.volume_change_db = volume_change_db

        self.time_shift_prob = time_shift_prob
        self.time_stretch_prob = time_stretch_prob
        self.pitch_shift_prob = pitch_shift_prob
        self.noise_prob = noise_prob
        self.volume_prob = volume_prob

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations to waveform.

        Args:
            waveform: Input waveform tensor, shape (num_samples,) or (1, num_samples)

        Returns:
            Augmented waveform tensor with same shape as input
        """
        # Ensure 2D: (1, num_samples)
        squeeze_output = False
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            squeeze_output = True

        original_length = waveform.shape[1]

        # Time shift (circular shift)
        if random.random() < self.time_shift_prob:
            waveform = self._time_shift(waveform)

        # Time stretch (changes tempo without pitch)
        if random.random() < self.time_stretch_prob:
            waveform = self._time_stretch(waveform)

        # Pitch shift
        if random.random() < self.pitch_shift_prob:
            waveform = self._pitch_shift(waveform)

        # Add noise
        if random.random() < self.noise_prob:
            waveform = self._add_noise(waveform)

        # Volume change
        if random.random() < self.volume_prob:
            waveform = self._volume_change(waveform)

        # Ensure output has same length as input
        waveform = self._adjust_length(waveform, original_length)

        if squeeze_output:
            waveform = waveform.squeeze(0)

        return waveform

    def _time_shift(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply random circular time shift."""
        shift_amount = random.randint(0, waveform.shape[1] - 1)
        return torch.roll(waveform, shifts=shift_amount, dims=1)

    def _time_stretch(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply time stretching (speed change without pitch change)."""
        rate = random.uniform(*self.time_stretch_range)

        # Use torchaudio's resample for time stretching effect
        # Stretch by resampling: new_rate = sample_rate * rate
        stretched = torchaudio.functional.resample(
            waveform,
            orig_freq=self.sample_rate,
            new_freq=int(self.sample_rate * rate)
        )

        return stretched

    def _pitch_shift(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply pitch shifting."""
        n_steps = random.uniform(*self.pitch_shift_range)

        try:
            # Use torchaudio's pitch_shift (requires sox backend or newer torchaudio)
            shifted = torchaudio.functional.pitch_shift(
                waveform,
                sample_rate=self.sample_rate,
                n_steps=n_steps
            )
            return shifted
        except Exception:
            # Fallback: simple resampling-based pitch shift (less accurate)
            # This changes both pitch and tempo, but is always available
            ratio = 2 ** (n_steps / 12)
            resampled = torchaudio.functional.resample(
                waveform,
                orig_freq=self.sample_rate,
                new_freq=int(self.sample_rate / ratio)
            )
            return resampled

    def _add_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise at random SNR."""
        snr_db = random.uniform(*self.noise_snr_range)

        # Calculate signal power
        signal_power = waveform.pow(2).mean()

        # Calculate noise power for desired SNR
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear

        # Generate and add noise
        noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
        return waveform + noise

    def _volume_change(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply random volume change."""
        gain_db = random.uniform(-self.volume_change_db, self.volume_change_db)
        gain_linear = 10 ** (gain_db / 20)
        return waveform * gain_linear

    def _adjust_length(
        self,
        waveform: torch.Tensor,
        target_length: int
    ) -> torch.Tensor:
        """Adjust waveform to target length by padding or cropping."""
        current_length = waveform.shape[1]

        if current_length == target_length:
            return waveform
        elif current_length > target_length:
            # Random crop
            start = random.randint(0, current_length - target_length)
            return waveform[:, start:start + target_length]
        else:
            # Pad with zeros
            padding = target_length - current_length
            return torch.nn.functional.pad(waveform, (0, padding))


class SpecAugment:
    """SpecAugment: Spectrogram-level augmentation.

    Applies frequency masking and time masking to spectrograms,
    as described in "SpecAugment: A Simple Data Augmentation Method
    for Automatic Speech Recognition" (Park et al., 2019).

    Args:
        frequency_mask_param: Maximum number of frequency bins to mask (F)
        time_mask_param: Maximum number of time frames to mask (T)
        num_frequency_masks: Number of frequency masks to apply
        num_time_masks: Number of time masks to apply
    """

    def __init__(
        self,
        frequency_mask_param: int = 27,
        time_mask_param: int = 100,
        num_frequency_masks: int = 1,
        num_time_masks: int = 1
    ):
        self.freq_mask = T.FrequencyMasking(freq_mask_param=frequency_mask_param)
        self.time_mask = T.TimeMasking(time_mask_param=time_mask_param)
        self.num_frequency_masks = num_frequency_masks
        self.num_time_masks = num_time_masks

    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment to spectrogram.

        Args:
            spectrogram: Input spectrogram, shape (channels, freq, time)

        Returns:
            Augmented spectrogram with same shape
        """
        # Apply frequency masks
        for _ in range(self.num_frequency_masks):
            spectrogram = self.freq_mask(spectrogram)

        # Apply time masks
        for _ in range(self.num_time_masks):
            spectrogram = self.time_mask(spectrogram)

        return spectrogram


class SimSiamAugmentor:
    """Combined augmentation pipeline for SimSiam.

    Combines waveform-level augmentations with spectrogram conversion
    and SpecAugment for comprehensive data augmentation.

    Args:
        audio_augmentor: Waveform-level augmentor
        spec_augment: Optional SpecAugment instance
        mel_spectrogram: MelSpectrogram transform
    """

    def __init__(
        self,
        audio_augmentor: AudioAugmentor,
        mel_spectrogram: T.MelSpectrogram,
        spec_augment: Optional[SpecAugment] = None
    ):
        self.audio_augmentor = audio_augmentor
        self.mel_spectrogram = mel_spectrogram
        self.spec_augment = spec_augment

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply full augmentation pipeline.

        Args:
            waveform: Input waveform, shape (num_samples,) or (1, num_samples)

        Returns:
            Augmented mel-spectrogram, shape (1, n_mels, time) or (3, n_mels, time)
        """
        # Step 1: Waveform augmentation
        augmented_waveform = self.audio_augmentor(waveform)

        # Ensure 2D for mel spectrogram
        if augmented_waveform.dim() == 1:
            augmented_waveform = augmented_waveform.unsqueeze(0)

        # Step 2: Convert to mel-spectrogram
        mel_spec = self.mel_spectrogram(augmented_waveform)

        # Step 3: Log scale (more perceptually meaningful)
        mel_spec = torch.log(mel_spec + 1e-9)

        # Step 4: SpecAugment (frequency and time masking)
        if self.spec_augment is not None:
            mel_spec = self.spec_augment(mel_spec)

        return mel_spec


def create_audio_augmentor(
    config: dict,
    sample_rate: int = 16000
) -> AudioAugmentor:
    """Factory function to create audio augmentor from config.

    Args:
        config: Augmentation configuration dictionary
        sample_rate: Audio sample rate in Hz

    Returns:
        AudioAugmentor instance configured according to config
    """
    return AudioAugmentor(
        sample_rate=sample_rate,
        time_stretch_range=tuple(config.get('time_stretch_range', [0.8, 1.2])),
        pitch_shift_range=tuple(config.get('pitch_shift_range', [-2, 2])),
        noise_snr_range=tuple(config.get('noise_snr_range', [15, 30])),
        volume_change_db=config.get('volume_change_db', 6.0),
        time_shift_prob=1.0,
        time_stretch_prob=0.5,
        pitch_shift_prob=0.5,
        noise_prob=0.3,
        volume_prob=0.8
    )


def create_simsiam_augmentor(
    config: dict,
    sample_rate: int = 16000,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512
) -> SimSiamAugmentor:
    """Factory function to create full SimSiam augmentor.

    Args:
        config: Augmentation configuration dictionary
        sample_rate: Audio sample rate in Hz
        n_mels: Number of mel frequency bins
        n_fft: FFT window size
        hop_length: Hop length for STFT

    Returns:
        SimSiamAugmentor instance with waveform + spectrogram augmentations
    """
    # Create waveform augmentor
    audio_augmentor = create_audio_augmentor(config, sample_rate)

    # Create mel spectrogram transform
    mel_spectrogram = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )

    # Create SpecAugment if enabled
    spec_augment = None
    if config.get('spec_augment', True):
        spec_augment = SpecAugment(
            frequency_mask_param=config.get('frequency_mask_param', 27),
            time_mask_param=config.get('time_mask_param', 100)
        )

    return SimSiamAugmentor(
        audio_augmentor=audio_augmentor,
        mel_spectrogram=mel_spectrogram,
        spec_augment=spec_augment
    )

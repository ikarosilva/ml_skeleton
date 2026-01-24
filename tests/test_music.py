
import torch
import torchaudio
import numpy as np
from pathlib import Path
import pytest
from unittest.mock import MagicMock

from ml_skeleton.music.dataset import SimSiamMusicDataset, simsiam_collate_fn
from ml_skeleton.music.simsiam_encoder import SimSiamEncoder
from ml_skeleton.music.augmentations import AudioAugmentor
from ml_skeleton.music.clementine_db import Song

@pytest.fixture
def dummy_song(tmp_path):
    sr = 16000
    duration = 2
    dummy_wav_path = tmp_path / "dummy.wav"
    waveform = torch.randn(1, sr * duration)
    torchaudio.save(str(dummy_wav_path), waveform, sr)
    
    song = MagicMock(spec=Song)
    song.filename = str(dummy_wav_path)
    song.filepath = dummy_wav_path
    song.artist = "artist"
    song.album = "album"
    song.title = "title"
    song.year = 2024
    return song

def test_simsiam_dataset_returns_waveforms(dummy_song):
    dataset = SimSiamMusicDataset(
        songs=[dummy_song],
        sample_rate=16000,
        duration=1.0,
        augmentor=AudioAugmentor(sample_rate=16000)
    )
    item = dataset[0]
    assert "view1" in item
    assert "view2" in item
    assert isinstance(item["view1"], torch.Tensor)
    assert item["view1"].shape == (16000,)

def test_simsiam_collate_fn_stacks_waveforms(dummy_song):
    dataset = SimSiamMusicDataset(
        songs=[dummy_song, dummy_song],
        sample_rate=16000,
        duration=1.0,
        augmentor=AudioAugmentor(sample_rate=16000)
    )
    batch = [dataset[0], dataset[1]]
    collated = simsiam_collate_fn(batch)
    assert "view1" in collated
    assert "view2" in collated
    assert collated["view1"].shape == (2, 16000)
    assert collated["view2"].shape == (2, 16000)

def test_encoder_with_spec_augment(dummy_song):
    encoder = SimSiamEncoder(
        sample_rate=16000,
        duration=1.0,
        spec_augment=True
    )
    encoder.train()

    dataset = SimSiamMusicDataset(
        songs=[dummy_song, dummy_song],
        sample_rate=16000,
        duration=1.0,
        augmentor=AudioAugmentor(sample_rate=16000)
    )
    batch = simsiam_collate_fn([dataset[0], dataset[1]])
    
    # Mock spec_augment to check if it's called
    encoder.spec_augment = MagicMock(wraps=encoder.spec_augment)

    p1, p2, z1, z2 = encoder.forward_simsiam(batch["view1"], batch["view2"])

    assert encoder.spec_augment.call_count > 0
    assert p1.shape == (2, encoder.projection_dim)
    assert z1.shape == (2, encoder.projection_dim)


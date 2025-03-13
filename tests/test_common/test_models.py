from jatic_ri._common.models import set_device
import pytest
import torch

def test_set_device_default_behavior():
    """ Ensure default behavior returns CPU if no GPU available. """
    if torch.cuda.is_available():
        expected_device = "cuda"
    elif torch.backends.mps.is_available():
        expected_device = "mps"
    else:
        expected_device = "cpu"
    device = set_device(None)
    assert device.type == expected_device
    
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_set_device_cuda():
    """ Test CUDA selection if available. """
    assert set_device(None).type == "cuda"
    
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_set_device_mps():
    """ Test MPS selection if available (for Apple Silicon). """
    assert set_device(None).type == "mps"

def test_set_device_idempotency():
    """ Ensure calling set_device multiple times doesn't change its output. """
    device1 = set_device("cpu")
    device2 = set_device(device1)
    assert device1 == device2

def test_set_device_with_torch_device():
    """Ensure passing an existing torch.device does nothing."""
    for dev in ["cpu", "cuda", "mps"]:
        expected_device = torch.device(dev)
        actual_device = set_device(expected_device)
        assert actual_device == expected_device

# NPU/GPU Device Management for AReaL Framework
"""
Unified device management for NPU/GPU support in AReaL framework.
This module provides unified interfaces for device detection, setup, and configuration.
"""

import os
import torch
from typing import Optional, Union, Dict, Any

from .npu import is_npu_available, set_npu_environment, prepare_npu_for_vllm


def get_device_type(device: Optional[Union[str, torch.device]] = None) -> str:
    """
    Get standardized device type string.
    
    Args:
        device: Device specification (cuda, npu, cpu, or torch.device)
        
    Returns:
        Standardized device type: 'cuda', 'npu', or 'cpu'
    """
    if device is None:
        # Auto-detect available device
        if is_npu_available():
            return 'npu'
        elif torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'
    
    if isinstance(device, torch.device):
        device = device.type
    
    device = str(device).lower()
    
    if device.startswith('npu'):
        return 'npu'
    elif device.startswith('cuda'):
        return 'cuda'
    else:
        return 'cpu'


def setup_device_environment(device_type: str, device_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Setup device environment and return configuration.
    
    Args:
        device_type: Device type ('cuda', 'npu', 'cpu')
        device_id: Optional device ID for multi-device systems
        
    Returns:
        Configuration dictionary with device setup info
    """
    config = {
        'device_type': device_type,
        'device_id': device_id,
        'available': False,
        'environment_set': False
    }
    
    if device_type == 'npu':
        if is_npu_available():
            set_npu_environment(device_id)
            config['available'] = True
            config['environment_set'] = True
        else:
            raise RuntimeError("NPU device requested but torch_npu not available")
    
    elif device_type == 'cuda':
        if torch.cuda.is_available():
            if device_id is not None:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
            config['available'] = True
            config['environment_set'] = True
        else:
            raise RuntimeError("CUDA device requested but not available")
    
    elif device_type == 'cpu':
        # CPU is always available
        config['available'] = True
        config['environment_set'] = True
    
    else:
        raise ValueError(f"Unsupported device type: {device_type}")
    
    return config


def prepare_inference_device(device: str) -> Dict[str, Any]:
    """
    Prepare device for inference tasks (vLLM, etc.).
    
    Args:
        device: Device specification string
        
    Returns:
        Device configuration dictionary
    """
    device_type = get_device_type(device)
    config = setup_device_environment(device_type)
    
    if device_type == 'npu':
        prepare_npu_for_vllm()
    
    return config


def get_torch_device(device_type: str, device_id: Optional[int] = None) -> torch.device:
    """
    Get torch.device object for the specified device type.
    
    Args:
        device_type: Device type ('cuda', 'npu', 'cpu')
        device_id: Optional device ID
        
    Returns:
        torch.device object
    """
    if device_type == 'npu':
        if device_id is not None:
            return torch.device(f'npu:{device_id}')
        else:
            return torch.device('npu')
    elif device_type == 'cuda':
        if device_id is not None:
            return torch.device(f'cuda:{device_id}')
        else:
            return torch.device('cuda')
    else:
        return torch.device('cpu')


def validate_device_compatibility(device: str) -> bool:
    """
    Validate if the requested device is compatible with current environment.
    
    Args:
        device: Device specification
        
    Returns:
        True if device is compatible, False otherwise
    """
    device_type = get_device_type(device)
    
    if device_type == 'npu':
        return is_npu_available()
    elif device_type == 'cuda':
        return torch.cuda.is_available()
    else:
        return True  # CPU is always available

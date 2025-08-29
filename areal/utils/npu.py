# Copyright 2025 Ant Group Inc.

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def is_npu_available() -> bool:
    """Check if NPU is available in the system."""
    try:
        import torch_npu
        return torch_npu.is_available()
    except ImportError:
        return False

def get_npu_device_count() -> int:
    """Get the number of available NPU devices."""
    try:
        import torch_npu
        return torch_npu.device_count()
    except ImportError:
        return 0

def set_npu_environment():
    """Set up NPU environment variables for optimal performance."""
    if not is_npu_available():
        logger.warning("NPU not available, skipping NPU environment setup")
        return
    
    # NPU specific environment variables
    npu_env_vars = {
        # Enable NPU compilation optimization
        "ASCEND_LAUNCH_BLOCKING": "0",
        # Enable NPU memory optimization
        "NPU_FUZZY_COMPILE_BLACKLIST": "Reshape",
        # Set default NPU device for vLLM
        "CUDA_VISIBLE_DEVICES": "",  # Clear CUDA devices
        "NPU_VISIBLE_DEVICES": os.environ.get("NPU_VISIBLE_DEVICES", "0"),
    }
    
    for key, value in npu_env_vars.items():
        if key not in os.environ:
            os.environ[key] = value
            logger.info(f"Set {key}={value}")

def prepare_npu_for_vllm(device_ids: Optional[str] = None):
    """Prepare NPU environment specifically for vLLM."""
    if not is_npu_available():
        logger.warning("NPU not available for vLLM")
        return False
    
    set_npu_environment()
    
    if device_ids:
        os.environ["NPU_VISIBLE_DEVICES"] = device_ids
        logger.info(f"Set NPU_VISIBLE_DEVICES={device_ids}")
    
    # Import torch_npu to initialize NPU runtime
    try:
        import torch_npu
        logger.info(f"NPU initialized successfully, {get_npu_device_count()} devices available")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize NPU: {e}")
        return False

# Copyright 2025 Ant Group Inc.

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def is_npu_available() -> bool:
    """Check if NPU is available in the system."""
    try:
        import torch
        # 使用标准的 torch.npu.is_available() API
        available = torch.npu.is_available()
        print(f"🔍 [NPU] NPU availability check: {available}")
        return available
    except Exception as e:
        print(f"❌ [NPU] NPU check failed: {e}")
        return False

def get_npu_device_count() -> int:
    """Get the number of available NPU devices."""
    try:
        import torch
        return torch.npu.device_count()
    except Exception:
        return 0

def set_npu_environment():
    """Set up NPU environment variables for optimal performance."""
    if not is_npu_available():
        logger.warning("NPU not available, skipping NPU environment setup")
        return
    
    # NPU specific environment variables
    npu_env_vars = {
        # Enable NPU compilation optimization
        "ASCEND_LAUNCH_BLOCKING": "1",  # 启用阻塞模式，便于调试
        # Enable NPU memory optimization
        "NPU_FUZZY_COMPILE_BLACKLIST": "Reshape",
        # Set default NPU device for vLLM
        "CUDA_VISIBLE_DEVICES": "",  # Clear CUDA devices
        "NPU_VISIBLE_DEVICES": os.environ.get("NPU_VISIBLE_DEVICES", "0"),
        # 启用NPU调试和性能优化
        "ASCEND_GLOBAL_LOG_LEVEL": "1",  # 启用详细日志
        "ASCEND_SLOG_PRINT_TO_STDOUT": "1",  # 日志输出到stdout
        # 禁用torch.compile以避免NPU兼容性问题
        "TORCHDYNAMO_DISABLE": "1",
        "TORCH_COMPILE_DISABLE": "1",
    }
    
    for key, value in npu_env_vars.items():
        if key not in os.environ:
            os.environ[key] = value
            logger.info(f"Set {key}={value}")
    
    # 配置torch compilation设置
    configure_npu_compilation()


def configure_npu_compilation():
    """Configure torch compilation settings for NPU compatibility."""
    try:
        import torch
        
        # 禁用torch.compile的动态编译
        if hasattr(torch, '_dynamo'):
            torch._dynamo.config.suppress_errors = True
            torch._dynamo.config.disable = True
        
        logger.info("NPU compilation settings configured: disabled torch.compile")
        
    except Exception as e:
        logger.warning(f"Failed to configure NPU compilation settings: {e}")


def init_npu_environment():
    """Initialize NPU environment and apply all necessary configurations."""
    if is_npu_available():
        set_npu_environment()
        return True
    return False


def device_synchronize():
    """Device-agnostic synchronization: NPU or CUDA."""
    try:
        if is_npu_available():
            import torch_npu
            torch.npu.synchronize()
        else:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    except Exception as e:
        logger.warning(f"Device synchronization failed: {e}")


def is_device_npu(device):
    """Check if the given device is an NPU device."""
    if hasattr(device, 'type'):
        return device.type == 'npu'
    return 'npu' in str(device)

def prepare_npu_for_vllm(device_ids: Optional[str] = None):
    """Prepare NPU environment specifically for vLLM."""
    if not is_npu_available():
        logger.warning("NPU not available for vLLM")
        return False
    
    set_npu_environment()
    
    if device_ids:
        os.environ["NPU_VISIBLE_DEVICES"] = device_ids
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = device_ids  # vLLM-Ascend使用的环境变量
        logger.info(f"Set NPU_VISIBLE_DEVICES={device_ids}, ASCEND_RT_VISIBLE_DEVICES={device_ids}")
    
    # Import torch_npu to initialize NPU runtime
    try:
        import torch_npu
        logger.info(f"NPU initialized successfully, {get_npu_device_count()} devices available")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize NPU: {e}")
        return False


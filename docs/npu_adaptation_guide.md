# AReaL NPU Adaptation Guide

本文档说明了如何在Huawei Ascend NPU上运行AReaL框架。

## NPU支持概览

AReaL框架现已支持华为昇腾NPU，实现了以下核心功能的NPU适配：

### ✅ 已完成的NPU适配

1. **FlashAttention NPU实现**
   - `flash_attn_varlen_func` → `torch_npu.npu_fusion_attention` 
   - `_kvcacheflash_attn_with` → `torch_npu.npu_incre_flash_attention`
   - 统一的GPU/NPU FlashAttention接口

2. **设备管理和环境配置**
   - NPU设备检测和初始化
   - 环境变量自动配置
   - 设备类型自动识别

3. **vLLM Remote Engine NPU支持**
   - NPU环境自动准备
   - 设备参数透传
   - NPU兼容性检查

4. **配置系统NPU集成**
   - `vLLMConfig.device` 支持 "npu" 选项
   - 设备兼容性验证
   - 自动设备检测

## 环境要求

### 硬件要求
- 华为昇腾NPU (Ascend 910/310P等)
- 支持的操作系统 (通常为Linux)

### 软件依赖
```bash
# 基础依赖
torch >= 2.0.0
torch_npu >= 2.0.0  # 华为官方NPU支持包

# 环境变量
export ASCEND_LAUNCH_BLOCKING=1
export NPU_VISIBLE_DEVICES=0,1,2,3
```

## 快速开始

### 1. 环境检查
```bash
python scripts/check_npu_status.py
```

### 2. NPU训练启动
```bash
# 参考示例脚本
bash examples/npu_training_example.sh
```

### 3. 配置文件修改
```yaml
inference_engine:
  type: vllm
  vllm_config:
    device: npu  # 指定使用NPU
    dtype: bfloat16
    enforce_eager: true
```

## 核心代码修改说明

### FlashAttention NPU适配

**文件**: `realhf/impl/model/modules/attn.py`

```python
# NPU FlashAttention统一接口
def flash_attn_with_kvcache(...):
    if HAS_NPU and str(q.device).startswith('npu'):
        # 使用NPU实现
        return torch_npu.npu_incre_flash_attention(...)
    else:
        # 使用GPU实现
        return _kvcacheflash_attn_with(...)

def flash_attn_varlen_func_unified(...):
    if HAS_NPU and str(q.device).startswith('npu'):
        # 使用NPU实现
        return torch_npu.npu_fusion_attention(...)
    else:
        # 使用GPU实现  
        return flash_attn_varlen_func(...)
```

### NPU环境管理

**文件**: `areal/utils/npu.py`

```python
def is_npu_available() -> bool:
    """检查NPU是否可用"""
    
def set_npu_environment(device_id: Optional[int] = None):
    """设置NPU环境变量"""
    
def prepare_npu_for_vllm():
    """为vLLM准备NPU环境"""
```

### 设备管理器

**文件**: `areal/utils/device_manager.py`

```python
def get_device_type(device=None) -> str:
    """获取标准化设备类型"""
    
def setup_device_environment(device_type: str) -> Dict:
    """设置设备环境"""
    
def prepare_inference_device(device: str) -> Dict:
    """为推理任务准备设备"""
```

## NPU vs GPU FlashAttention参数映射

| GPU参数 | NPU参数 | 说明 |
|---------|---------|------|
| `dropout_p` | `keep_prob` | dropout保持概率 = 1 - dropout_p |
| `causal` | `sparse_mode` | 因果掩码模式 |
| `softmax_scale` | `scale` | 注意力缩放因子 |
| `cu_seqlens` | `actual_seq_qlen`/`actual_seq_kvlen` | 序列长度信息 |

## 故障排除

### 常见问题

1. **torch_npu未找到**
   ```bash
   pip install torch_npu
   # 或按华为官方文档安装
   ```

2. **NPU设备不可用**
   ```bash
   # 检查设备状态
   npu-smi info
   
   # 检查驱动
   cat /usr/local/Ascend/driver/version.info
   ```

3. **环境变量配置**
   ```bash
   export ASCEND_LAUNCH_BLOCKING=1
   export NPU_VISIBLE_DEVICES=0
   unset CUDA_VISIBLE_DEVICES  # 清除CUDA设置
   ```

### 性能优化建议

1. **内存优化**
   - 使用 `bfloat16` 数据类型
   - 调整 `max_num_seqs` 参数

2. **并行配置**
   - 根据NPU数量设置 `NPU_VISIBLE_DEVICES`
   - 配置分布式训练参数

3. **算子优化**
   - 启用 `enforce_eager=True` 避免图模式问题
   - 使用NPU优化的数据类型

## 验证NPU适配

运行以下命令验证NPU支持是否正常：

```bash
# 1. 环境检查
python scripts/check_npu_status.py

# 2. 简单推理测试
python -c "
from areal.api.cli_args import vLLMConfig
config = vLLMConfig(device='npu')
print(f'NPU config created: {config.device}')
"

# 3. 注意力机制测试
python -c "
from realhf.impl.model.modules.attn import flash_attn_with_kvcache
print('NPU FlashAttention imported successfully')
"
```

## 已知限制

1. **算子兼容性**: 部分高级算子可能需要进一步适配
2. **性能差异**: NPU和GPU的性能特性不同，需要针对性优化
3. **调试支持**: NPU调试工具相对GPU生态较少

## 下一步计划

1. **性能优化**: 针对NPU特性进行深度优化
2. **算子补全**: 适配更多专用算子
3. **多卡支持**: 完善NPU多卡并行训练
4. **混合精度**: 优化NPU混合精度训练策略

---

**注意**: 这两个实现（torch_npu.npu_fusion_attention 和 torch_npu.npu_incre_flash_attention）能够解决当前缺失的算子问题，为AReaL框架提供完整的NPU支持。

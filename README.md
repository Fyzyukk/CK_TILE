# CK_TILE - 从 AITER 转换的 Composable Kernel 框架

## 概述

这个项目是将 AITER 代码库转换为 CK_TILE 的示例实现。主要变化包括：

- 将所有 `aiter` 相关的命名更改为 `ck_tile`
- 更新环境变量和配置
- 保持原有的功能和架构

## 主要转换内容

### 1. 包名和模块名
- `aiter` → `ck_tile`
- `aiter_` → `ck_tile_`
- `aiter_lib` → `ck_tile_lib`

### 2. 环境变量
- `AITER_REBUILD` → `CK_TILE_REBUILD`
- `AITER_LOG_MORE` → `CK_TILE_LOG_MORE`
- `AITER_JIT_DIR` → `CK_TILE_JIT_DIR`
- `AITER_ASM_DIR` → `CK_TILE_ASM_DIR`
- `AITER_FP4x2` → `CK_TILE_FP4x2`

### 3. 目录结构
- `AITER_ROOT_DIR` → `CK_TILE_ROOT_DIR`
- `AITER_META_DIR` → `CK_TILE_META_DIR`
- `AITER_CSRC_DIR` → `CK_TILE_CSRC_DIR`
- `AITER_GRADLIB_DIR` → `CK_TILE_GRADLIB_DIR`
- `AITER_ASM_DIR` → `CK_TILE_ASM_DIR`

### 4. 函数和类名
- `module_aiter_enum` → `module_ck_tile_enum`
- `rebuilded_list` 中的模块名更新

### 5. 导入语句
- `from aiter import*` → `from ck_tile import*`
- `torch.ops.aiter` → `torch.ops.ck_tile`

## 文件结构

```
CK_TILE/
├── setup.py              # 原始安装配置
├── setup_rocm.py         # ROCm 专用安装配置
├── build_rocm.sh         # ROCm 构建脚本
├── ck_tile/
│   ├── __init__.py       # 包初始化文件
│   └── core.py           # 核心功能模块（转换后的）
├── csrc/
│   ├── jit/              # JIT 编译相关文件
│   └── python_api.cpp    # C++ Python API
├── example_usage.py      # 基本使用示例
├── example_rocm.py       # ROCm 使用示例
└── README.md            # 本文件
```

## 安装和使用

### 1. 环境要求
- Python >= 3.8
- PyTorch
- ROCm 工具链
- Composable Kernel 依赖

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. ROCm 环境设置
```bash
# 设置 ROCm 环境变量
export ROCM_HOME=/opt/rocm
export HIP_HOME=$ROCM_HOME
export PATH=$HIP_HOME/bin:$PATH

# 设置 CK_TILE 环境变量
export CK_TILE_REBUILD=0
export CK_TILE_LOG_MORE=1
export GPU_ARCHS=gfx90a  # 根据您的 GPU 调整
export MAX_JOBS=4
```

### 4. 构建 C++ 扩展
```bash
# 使用构建脚本（推荐）
chmod +x build_rocm.sh
./build_rocm.sh

# 或手动构建
python setup_rocm.py build_ext --inplace
```

### 5. 运行示例
```bash
# 基本示例
python example_usage.py

# ROCm 特定示例
python example_rocm.py
```

## 核心功能

### 1. JIT 编译系统
- 支持动态编译 CUDA/HIP 内核
- 自动管理构建缓存
- 多进程安全的编译锁

### 2. 操作装饰器
```python
@compile_ops("module_name")
def my_operation(tensor: torch.Tensor) -> torch.Tensor:
    # 操作实现
    return result
```

### 3. 构建配置
- 支持自定义编译标志
- GPU 架构检测和验证
- 内存和 CPU 核心数优化

### 4. ROCm 特定功能
- AMD HIP 内核支持
- rocBLAS 集成
- 自动 GPU 架构检测
- HIP 版本特定优化

## 注意事项

1. **依赖项**: 需要正确安装 Composable Kernel 和相关依赖
2. **硬件支持**: 目前仅支持 ROCm 平台
3. **CUDA 内核**: 实际使用需要实现具体的 CUDA 内核
4. **配置调整**: 可能需要根据具体硬件调整 GPU_ARCHS 等设置

## ROCm 特定注意事项

1. **ROCm 安装**: 确保已正确安装 AMD ROCm 工具链
2. **GPU 架构**: 根据您的 AMD GPU 调整 GPU_ARCHS 设置
   - gfx90a: MI250X
   - gfx940: MI300A
   - gfx941: MI300X
   - gfx942: MI300X
   - gfx1100: RX 7900 XTX
3. **HIP 版本**: 不同 HIP 版本可能需要不同的编译标志
4. **内存管理**: 注意 GPU 内存使用，避免 OOM 错误

## 转换检查清单

- [x] 包名和模块名更新
- [x] 环境变量重命名
- [x] 目录路径更新
- [x] 函数和类名更新
- [x] 导入语句修改
- [x] 配置文件更新
- [x] 示例代码创建
- [x] 文档更新

## 故障排除

### 常见问题

1. **导入错误**: 确保所有依赖都已正确安装
2. **编译失败**: 检查 ROCm 工具链和 GPU 架构设置
3. **内存不足**: 调整 MAX_JOBS 环境变量
4. **权限问题**: 确保对构建目录有写权限

### 调试技巧

- 设置 `CK_TILE_LOG_MORE=2` 获取详细日志
- 使用 `CK_TILE_REBUILD=1` 强制重新编译
- 检查 `~/.ck_tile/` 目录下的缓存文件

## 贡献

欢迎提交问题和改进建议！

## 许可证

MIT License

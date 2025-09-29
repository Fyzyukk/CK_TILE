## 文件结构

```
CK_TILE/
├── setup.py              # 原始安装配置
├── ck_tile/  
│   └── include/ck_tile   #ck_tile 头文件     
├── csrc/
│   ├── kernels/          # kernel文件
│   ├── apis/             # 传递参数
│   └── python_api.cpp    # PYBIND 绑定
├── test.py               # 测试
└── README.md             # 本文件
```

## 安装步骤
python3 setup.py install
python3 setup.py install > build.log 2>&1

## 测试步骤
python3 test_gemm.py
python3 test_gemm.py > test_gemm.log 2>&1

## Example
import ck_tile_python

ck_tile_python.gemm_api(A_tensor, B_tensor, args)
ck_tile_python.batched_gemm_api(A_tensor, B_tensor, args)
ck_tile_python.flatmm_api(A_tensor, B_tensor, args)
ck_tile_python.grouped_gemm_api(A_tensors, B_tensors, args)
.....

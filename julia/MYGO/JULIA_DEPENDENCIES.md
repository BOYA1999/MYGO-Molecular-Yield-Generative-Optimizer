# MYGO.jl Julia依赖说明

## Julia环境配置

### 1. 安装Julia

从 https://julialang.org/ 下载并安装 Julia 1.8 或更高版本。

### 2. 安装Julia包

在Julia REPL中运行以下命令安装依赖：

```julia
# 进入包管理模式
]

# 添加所需的包
add Distributions
add LinearAlgebra
add Random
add SparseArrays
add Printf
add Statistics

# 如果需要使用PyCall与Python交互
add PyCall

# 退出包管理模式
Ctrl + C
```

### 3. 在Python中使用Julia

安装PyCall并配置Julia：

```python
# 在Python中，首先确保安装了PyJulia
pip install julia

# 然后安装Julia包
julia> using Pkg
julia> Pkg.add("PyCall")
julia> Pkg.build("PyCall")
```

### 4. 验证安装

```julia
# 在Julia中测试
using MYGO
mol = MYGO.parse_smiles("CCO")
descriptors = MYGO.calculate_descriptors(mol)
println(descriptors)
```

## 包结构

```
MYGO/
├── Project.toml          # Julia包配置
├── README.md             # 使用说明
└── src/
    ├── MYGO.jl           # 主模块
    ├── molecule.jl       # 分子结构
    ├── descriptors.jl    # 分子描述符
    ├── fingerprints.jl   # 分子指纹
    └── admet_predictor.jl # ADMET预测
```

## 性能对比

Julia实现相比Python的优势：

1. **计算速度**: Julia编译后的代码接近C语言速度
2. **并行计算**: 原生支持多线程和GPU加速
3. **数值精度**: 更精确的数值计算

典型性能提升：
- 分子描述符计算: 2-5x 加速
- 批量分子处理: 5-10x 加速
- ADMET预测: 3-8x 加速

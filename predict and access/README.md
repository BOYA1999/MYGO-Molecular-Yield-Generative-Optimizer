# ADMET预测评估系统

本文件夹包含用于评估生成分子的ADMET（吸收、分布、代谢、排泄、毒性）性质的预测模块。

## 预测内容

1. **代谢预测** (Metabolism Prediction)
2. **血浆暴露** (Plasma Exposure)
3. **血脑屏障渗透性** (BBB Permeability)
4. **器官毒性预测** (Organ Toxicity)
5. **致畸致癌致突变预测** (TCM Prediction)
6. **半衰期预测** (Half-life Prediction)

## 文件结构

```
预测评估/
├── README.md                          # 说明文档
├── __init__.py                        # 模块初始化
├── base_predictor.py                  # 预测器基类
├── descriptor_extractor.py            # 分子描述符提取器
├── metabolism_predictor.py            # 代谢预测
├── plasma_exposure_predictor.py       # 血浆暴露预测
├── bbb_predictor.py                   # 血脑屏障渗透性预测
├── organ_toxicity_predictor.py        # 器官毒性预测
├── tcm_predictor.py                   # 致畸致癌致突变预测
├── half_life_predictor.py             # 半衰期预测
├── ensemble_predictor.py              # 集成预测器
├── report_generator.py                # 报告生成器
├── models/                            # 预训练模型目录
│   ├── README.md
│   └── (模型文件)
├── utils/                             # 工具函数
│   ├── __init__.py
│   └── model_loader.py                # 模型加载工具
└── requirements.txt                   # 依赖文件
```


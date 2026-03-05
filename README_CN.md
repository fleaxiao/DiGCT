# DiGCT：面向压接式IGCT表面热管理的物理-数据协同约束扩散模型

![IEEE](https://img.shields.io/badge/IEEE-Reviewing-blue?style=for-the-badge&logo=ieee)
![Techrxiv](https://img.shields.io/badge/TechRxiv-2312.16476-8A2BE2?style=for-the-badge&logo=arxiv) [![项目主页](https://img.shields.io/badge/网站-项目主页-green?style=for-the-badge&logo=github)](https://github.com/fleaxiao/DiGCT)
[![数据集 IGCT X](https://img.shields.io/badge/数据集-IGCT_x-yellow?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/fleaxiao/IGCTX)

本项目是论文《面向压接式IGCT表面热管理的物理-数据协同约束扩散模型》的官方实现。该项目通过基于扩散模型的数字孪生，实现对压接式IGCT表面温度分布的监测、评估与优化。

<img src="./assets/figure3.png" alt="flowchart" width="800">

## 这是什么？

**DiGCT** 是一个专为压接式绝缘栅换向晶闸管（IGCT）表面热管理设计的深度学习框架。IGCT是大功率电力电子系统（如高压直流输电、电机驱动等）中的关键器件，其表面温度分布直接影响器件的可靠性与寿命。

本项目的核心思路是：
1. 将物理解析预测与实时温度测量进行插值融合，构建IGCT表面温度参考场；
2. 通过扩散模型迭代精化参考场的残差误差，生成符合物理一致性的高保真表面温度分布；
3. 支持基于梯度的在线温度优化，可调控最大值、均值和空间方差等多种指标。

## ✨ 核心亮点

* **物理-数据协同融合**：通过插值解析预测与实时温度测量，将物理机理和可观测信息压缩为几何表征，形成表面温度参考场。
* **启发式物理约束精化**：扩散模型在保证物理一致性的前提下，对参考场的残差误差进行迭代精化，生成高保真的IGCT表面温度分布。
* **基于梯度的温度优化**：在线优化策略可调节IGCT表面温度分布，支持最大值、均值、空间方差等多种优化指标。
* **专属数据集 _`IGCT X`_**：首个面向压接式IGCT表面热管理的开源数据集，包含多物理耦合效应和多种系统参数下的GCT表面与侧面温度配对数据。

## 🧩 环境配置

请确保满足 `assets/requirement.yaml` 中的依赖要求：

```bash
conda env create -n DiGCT -f requirement.yml
```

主要依赖：
* Python >= 3.12
* PyTorch >= 1.6.0

## 🔥 快速开始

### 🗂️ 数据准备

* 创建空文件夹 `data`：
```bash
mkdir -p data
```

* 下载开源数据集 [IGCTX](https://huggingface.co/datasets/fleaxiao/IGCTX) 至 `data` 文件夹

* 在 `configs/config_data.yml` 中调整温度预处理和解析模型的关键参数：

  - `surface`：裁剪表面温度目标
  - `side`：裁剪侧面温度测量
  - `L2S`：将侧面温度测量线转换为表面视角参考
  - `P2S`：将侧面温度测量点转换为表面视角参考
  - `PA2S`：将侧面温度测量点与解析模型结果转换为表面视角参考
  - `G`：计算表面温度目标与温度参考之间的差距

* 预处理DiGCT训练数据（预处理结果将保存至 `dataset` 文件夹）：
```bash
python data.py -config configs/config_data.yml
```

### 💪 模型训练

* 在 `configs/config_model.yml` 中调整模型训练关键参数：

  - `training`：训练开关
  - `generate_sample`：训练后样本生成开关
  - `physics_constraint`：物理约束去噪精化开关

* 训练模型（训练结果保存至 `results` 文件夹）：
```bash
python model.py -config configs/config_model.yml
```

### ✍️ 模型测试

* 在 `configs/config_model.yml` 中调整模型测试关键参数：

  - `testing`：测试开关
  - `test_path`：结果文件夹路径
  - `calculate_metric`：基于生成样本评估模型性能
  - `sample_metric`：通过采样过程评估模型性能

* 测试模型（测试结果保存至对应测试文件夹）：
```bash
python model.py -config configs/config_model.yml
```

## 🙏 致谢

本项目基于以下开源工作构建：
- [hojonathanho/diffusion](https://github.com/hojonathanho/diffusion)
- [ximinng/LLM4SVG](https://github.com/ximinng/LLM4SVG)

感谢上述作者的杰出贡献。

## 📋 引用

如果本代码对您的研究有所帮助，请引用以下工作：

```

```

## ☎️ 联系方式

如有任何问题，请联系作者：x.yang2@tue.nl

## ©️ 许可证

本项目采用 MIT 许可证。

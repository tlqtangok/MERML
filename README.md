# MERML: 基于数据驱动的递归动力学建模

## 项目概述

MERML (Machine learning Enhanced Recursive Modeling for kinetics Learning) 是一个基于机器学习的递归动力学建模框架，用于预测化学反应的动力学行为。本项目实现了论文《Data-Driven Recursive Kinetic Modeling for Fenton Reaction》中提出的方法，并将其应用于多个化学反应系统。

## 论文信息

### 主要论文
- **标题**: Data-Driven Recursive Kinetic Modeling for Fenton Reaction
- **位置**: 根目录下的 `data-driven-recursive-kinetic-modeling-for-fenton-reaction.pdf`
- **补充材料**: `SI---Data-Driven Recursive Kinetic Modeling for Fenton Reaction.pdf`

### 核心方法论
本项目提出了一种**机器学习增强的递归建模（MERML）**方法，通过以下特点实现高精度的化学反应动力学预测：

1. **递归预测机制**: 利用前一时刻的状态预测下一时刻的状态
2. **多时间尺度建模**: 支持不同时间步长（lag）的预测
3. **特征工程**: 结合分子描述符和反应条件进行综合建模
4. **多模型对比**: 实现了多种机器学习算法的对比评估

## 项目结构

```
MERML/
├── Fenton/                                      # Fenton反应系统
│   ├── Fenton-data.csv                          # Fenton反应数据集
│   ├── main.py                                  # 主程序入口
│   ├── globalvaribale.py                        # 全局变量管理
│   ├── init_func.py                             # 初始化函数
│   ├── preprocess.py                            # 数据预处理和特征工程
│   ├── exploratory_data_analysis.py             # 探索性数据分析
│   ├── plot.py                                  # 结果可视化
│   ├── reference model.py                       # 参考模型（传统动力学模型）
│   ├── descriptors.txt                          # 分子描述符
│   ├── cluster_label.txt                        # 聚类标签
│   ├── cluster_result.txt                       # 聚类结果
│   └── title.txt                                # 标题信息
│
├── Dushman reaction/                            # Dushman反应系统
│   ├── Dushman reaction.xlsx                    # 原始数据
│   ├── Dushman test.txt                         # 测试集数据
│   ├── Dushman test-io3.txt                     # IO3测试数据
│   ├── Dushman test-io3-train.txt               # IO3训练数据
│   ├── Dushman-i-test.txt                       # I离子测试数据
│   ├── Dushman-i-train.txt                      # I离子训练数据
│   ├── main.py                                  # 主程序
│   ├── globalvaribale.py                        # 全局变量
│   ├── plot.py                                  # 可视化脚本
│   └── reference model.py                       # 参考模型
│
├── PMAR (secondary amine-mediated aldol) reaction/  # PMAR反应系统
│   ├── PMAR-origin.xlsx                         # 原始数据
│   ├── PMAR-merged.xlsx                         # 合并后的数据
│   ├── PMAR.txt                                 # 处理后的数据
│   └── main.py                                  # 主程序
│
├── RCM (ring-closing metathesis) reaction/      # RCM反应系统
│   ├── RCM.xlsx                                 # 反应数据
│   ├── RCM-Condition.xlsx                       # 反应条件
│   └── main.py                                  # 主程序
│
├── Photocatalytic reaction/                     # 光催化反应系统
│   ├── dataset/                                 # 数据集目录
│   ├── dataset_generation_photocatalytic_reaction.py  # 数据生成
│   └── model_photocatalytic_reaction.py         # 模型实现
│
├── Simulation cases/                            # 模拟案例
│   ├── dataset/                                 # 模拟数据集
│   ├── dataset_generation_simulation_cases.py   # 数据生成
│   └── model_simulation_cases.py                # 模型实现
│
├── requirement.yml                              # Conda环境配置文件
├── data-driven-recursive-kinetic-modeling-for-fenton-reaction.pdf  # 主论文
└── SI---Data-Driven Recursive Kinetic Modeling for Fenton Reaction.pdf  # 补充材料
```

## 环境配置

### 1. 使用Conda创建环境（推荐）

```bash
# 使用提供的环境配置文件创建环境
conda env create -f requirement.yml

# 激活环境
conda activate Fenton
```

### 2. 主要依赖包

- **Python**: 3.9.16
- **核心机器学习库**:
  - XGBoost: 1.5.1
  - scikit-learn: 1.1.3
  - NumPy: 1.22.3
  - Pandas: 1.4.1
  - SciPy: 1.9.1
  
- **化学信息学**:
  - RDKit: 2022.03.2 (用于分子描述符计算)
  
- **可视化**:
  - Matplotlib: 3.5.2
  - Seaborn: 0.11.0
  - Mayavi: 4.8.1 (3D可视化)
  
- **模型解释**:
  - SHAP: 0.41.0
  - ALE Python: 0.1.dev17
  
- **其他工具**:
  - SymPy: 1.12 (符号计算)
  - Symfit: 0.5.6 (拟合工具)

## 反应系统详细说明

### 1. Fenton反应系统

Fenton反应是本项目的核心案例，用于降解有机污染物。

#### 数据集特点
- **污染物种类**: 12种不同的有机污染物（酚类化合物）
- **反应条件**: Fe²⁺浓度、H₂O₂浓度、污染物初始浓度
- **时间序列**: 6个采样时间点 (2, 5, 9, 15, 22, 30分钟)
- **样本数**: 288个反应条件

#### 污染物列表
1. phenol (苯酚)
2. chlorophenol (氯苯酚)
3. nitrophenol (硝基苯酚)
4. p-hydroxybenzoic acid (对羟基苯甲酸)
5. quinol (对苯二酚)
6. p-Hydroxybenzaldehyde (对羟基苯甲醛)
7. p-Hydroxyacetophenone (对羟基苯乙酮)
8. p-hydroxyanisole (对羟基苯甲醚)
9. Methyl 4-hydroxybenzoate (对羟基苯甲酸甲酯)
10. p-Hydroxybenzyl Alcohol (对羟基苄醇)
11. 4-acetamidophenol (对乙酰氨基苯酚)
12. p-methylphenol (对甲基苯酚)

#### 运行步骤

```bash
cd Fenton

# 1. 探索性数据分析
python exploratory_data_analysis.py

# 2. 运行MERML模型
python main.py

# 3. 运行参考模型（传统动力学模型）进行对比
python "reference model.py"

# 4. 生成可视化结果
python plot.py
```

#### 模型配置选项

在 `main.py` 中可以配置以下参数：

- **数据划分方式**:
  - `'evaluate'`: 随机划分训练集和测试集（80/20）
  - `'pollutant'`: 按污染物类型划分（留一法交叉验证）
  - `'concentration'`: 按浓度条件划分
  - `'learning curve'`: 学习曲线分析

- **分子描述符类型**:
  - `'RDKit'`: RDKit指纹
  - `'Morgan/Circular'`: Morgan指纹
  - `'MACCS keys'`: MACCS键
  - `'Morgan_WHIM'`: Morgan指纹 + WHIM描述符（推荐）
  - `'Morgan_Morse'`: Morgan指纹 + Morse描述符

- **时间滞后参数** (`lag`):
  - `lag=1`: 仅使用前一时刻信息
  - `lag=2`: 使用前两个时刻信息（推荐）

### 2. Dushman反应系统

Dushman反应是一个经典的化学振荡反应，用于研究复杂反应动力学。

#### 反应特点
- 涉及IO₃⁻、I⁻、H⁺等多种离子
- 非线性动力学行为
- pH依赖性强

#### 运行步骤

```bash
cd "Dushman reaction"

# 1. 运行MERML模型
python main.py

# 2. 运行参考模型对比
python "reference model.py"

# 3. 生成结果图表
python plot.py
```

#### 数据文件说明
- `Dushman reaction.xlsx`: 完整的实验数据
- `Dushman test.txt`: 测试集（完整动力学曲线）
- `Dushman test-io3.txt`: IO₃⁻离子浓度测试数据
- `Dushman-i-test.txt`: I⁻离子浓度测试数据

### 3. PMAR反应系统

PMAR (Secondary Amine-Mediated Aldol Reaction) 是一种有机合成反应。

#### 运行步骤

```bash
cd "PMAR (secondary amine-mediated aldol) reaction"

# 运行模型
python main.py
```

#### 数据处理
- 原始数据在 `PMAR-origin.xlsx`
- 预处理后的数据在 `PMAR-merged.xlsx`
- 训练数据在 `PMAR.txt`

### 4. RCM反应系统

RCM (Ring-Closing Metathesis) 是一种环化反应，广泛用于有机合成。

#### 运行步骤

```bash
cd "RCM (ring-closing metathesis) reaction"

# 运行模型
python main.py
```

#### 数据文件
- `RCM.xlsx`: 反应产率数据
- `RCM-Condition.xlsx`: 详细的反应条件

### 5. 光催化反应系统

研究光催化降解反应的动力学行为。

#### 运行步骤

```bash
cd "Photocatalytic reaction"

# 1. 生成训练数据集
python dataset_generation_photocatalytic_reaction.py

# 2. 训练和评估模型
python model_photocatalytic_reaction.py
```

#### 特点
- 时间序列: [0, 10, 30, 60, 100, 150, 210] 分钟
- 考虑光照条件的影响

### 6. 模拟案例

使用合成数据验证MERML方法的有效性。

#### 运行步骤

```bash
cd "Simulation cases"

# 1. 生成模拟数据
python dataset_generation_simulation_cases.py

# 2. 运行模型验证
python model_simulation_cases.py
```

#### 特点
- 时间序列: [0, 50, 100, 200, 400, 600, 800, 1000] 时间单位
- 可控的"真实"动力学参数
- 用于方法验证和误差分析

## 核心算法说明

### MERML递归建模框架

#### 1. 特征构建

对于时刻 t 的预测，特征包括：
- **静态特征**: 分子描述符、初始反应条件
- **动态特征**: 
  - `t_previous`: 前一采样时刻
  - `t_current`: 当前采样时刻
  - `Δt`: 时间间隔
  - `C_previous`: 前一时刻的浓度

#### 2. 训练数据构建

```python
# lag=1: 使用前一时刻
for j in range(time_points-1):
    features = [molecular_descriptors, initial_conditions, 
                t[j], t[j+1], t[j+1]-t[j], C[j]]
    label = C[j+1]

# lag=2: 额外使用前两个时刻
for k in range(time_points-2):
    features = [molecular_descriptors, initial_conditions,
                t[k], t[k+2], t[k+2]-t[k], C[k]]
    label = C[k+2]
```

#### 3. 递归预测

测试时，模型递归地预测每个时间点：
```
C(t₁) → 预测 C(t₂)
C(t₂) → 预测 C(t₃)
...
```

### XGBoost模型配置

项目使用XGBoost作为基础学习器，并采用Huber损失函数增强鲁棒性：

```python
params = {
    'max_depth': 7,
    'eta': 0.1,
    'objective': 'reg:squarederror',  # 或使用自定义Huber损失
    'eval_metric': 'rmse',
}
```

### 损失函数

Huber损失函数（对异常值更鲁棒）：

```python
def huber(gap):
    sigma = SIGMA  # 控制参数
    grad = sign(sigma) * sigma * gap / sqrt(sigma² + gap²)
    hess = sign(sigma) * sigma³ / (sigma² + gap²)^1.5
    return grad, hess
```

## 分子描述符详解

### 支持的描述符类型

1. **RDKit指纹**: 基于分子拓扑结构的二进制指纹
2. **Atom Pairs**: 原子对指纹
3. **Topological Torsions**: 拓扑扭转指纹
4. **MACCS keys**: 166位的结构键指纹
5. **Morgan/Circular**: 环形指纹（类似ECFP）
6. **2D Pharmacophore**: 药效团指纹
7. **Pattern**: 模式指纹
8. **3D-Morse**: 3D Morse描述符
9. **WHIM**: 加权整体不变分子描述符
10. **Morgan_WHIM**: Morgan指纹 + WHIM描述符（推荐）
11. **Morgan_Morse**: Morgan指纹 + Morse描述符

### 描述符生成流程

```python
# 1. 将污染物名称映射到SMILES
smiles = MAP_POLLUTANT_TO_SMILES[pollutant_name]

# 2. 生成分子描述符
descriptors = encoder(data, fp='Morgan_WHIM')

# 3. 过滤常量特征
# 自动删除所有样本中值相同的描述符
```

## 模型评估指标

项目使用以下指标评估模型性能：

1. **R²分数**: 决定系数，衡量模型拟合优度
2. **MAE**: 平均绝对误差
3. **RMSE**: 均方根误差

```python
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

r2 = r2_score(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
```

## 模型可解释性

### SHAP值分析

项目使用SHAP (SHapley Additive exPlanations) 分析特征重要性：

```python
import shap

# 计算SHAP值
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# 可视化
shap.summary_plot(shap_values, X)
```

### ALE图

累积局部效应（ALE）图用于分析特征与预测的关系：

```python
from alepython import ale_plot

ale_plot(model, X, feature_name)
```

## 与参考模型对比

每个反应系统都包含参考模型（`reference model.py`），基于传统的化学动力学方程：

### Fenton反应参考模型

使用ODEModel和Lambert W函数求解微分方程：

```python
# 动力学方程
dS/dt = -k * k1 * [Fe²⁺] * [H₂O₂] * [S] / (k[S] + k3[H₂O₂] + k7[Fe²⁺])
```

### Dushman反应参考模型

```python
# 动力学方程
dIO₃⁻/dt = -k₁[IO₃⁻]ᵃ[H⁺]ᵇ[I⁻]ᶜ
```

## 交叉验证策略

### 1. Leave-One-Pollutant-Out (LOPO)

```python
# 评估对新污染物的外推能力
type = 'pollutant'
for ID in range(12):  # 12种污染物
    train, test, data = data_init(type, ID)
```

### 2. Leave-One-Concentration-Out (LOCO)

```python
# 评估对新浓度条件的外推能力
type = 'concentration'
for ID in range(10):  # 10次重复
    train, test, data = data_init(type, ID)
```

### 3. 随机划分

```python
# 标准的80/20划分
type = 'evaluate'
train, test, data = data_init(type)
```

## 结果可视化

### 1. 实验值-预测值散点图

```python
# EP图（Experimental vs Predicted）
plt.scatter(y_true, y_pred)
plt.plot([0, 1], [0, 1], 'r--')  # 对角线
plt.xlabel('Measured Conversion')
plt.ylabel('Predicted Conversion')
```

### 2. 动力学曲线对比

```python
# 时间序列曲线
plt.plot(time, y_true, 'o-', label='Experimental')
plt.plot(time, y_pred, 's--', label='MERML')
plt.xlabel('Time (min)')
plt.ylabel('C/C₀')
```

### 3. 特征重要性图

```python
# XGBoost特征重要性
xgb.plot_importance(model)
```

## 常见问题与解决方案

### 1. 环境安装问题

**问题**: RDKit安装失败

**解决方案**:
```bash
# 使用conda安装（推荐）
conda install -c conda-forge rdkit

# 或使用pip
pip install rdkit-pypi
```

### 2. 内存不足

**问题**: 大数据集训练时内存不足

**解决方案**:
- 减少XGBoost的`max_depth`
- 使用更少的分子描述符
- 分批处理数据

### 3. 收敛问题

**问题**: 模型训练不收敛或性能差

**解决方案**:
- 调整学习率`eta`
- 增加训练轮数`num_boost_round`
- 尝试不同的损失函数
- 检查数据预处理和特征缩放

### 4. SHAP计算慢

**问题**: SHAP值计算耗时过长

**解决方案**:
```python
# 使用树模型加速
explainer = shap.TreeExplainer(model)

# 或采样部分数据
X_sample = X.sample(n=100)
shap_values = explainer.shap_values(X_sample)
```

## 复现论文结果的完整流程

### 步骤1: 环境准备

```bash
# 克隆仓库
git clone https://github.com/tlqtangok/MERML.git
cd MERML

# 创建conda环境
conda env create -f requirement.yml
conda activate Fenton
```

### 步骤2: Fenton反应系统复现

```bash
cd Fenton

# A. 数据探索
python exploratory_data_analysis.py
# 输出: 数据分布图、聚类分析结果

# B. MERML模型训练与评估
python main.py
# 输出: 
# - 训练集和测试集的R²、MAE、RMSE
# - 特征重要性图
# - SHAP值分析
# - 预测结果文件

# C. 参考模型对比
python "reference model.py"
# 输出: 传统动力学模型的预测结果

# D. 结果可视化
python plot.py
# 输出: EP图、动力学曲线对比图
```

### 步骤3: 其他反应系统验证

```bash
# Dushman反应
cd "../Dushman reaction"
python main.py
python "reference model.py"
python plot.py

# PMAR反应
cd "../PMAR (secondary amine-mediated aldol) reaction"
python main.py

# RCM反应
cd "../RCM (ring-closing metathesis) reaction"
python main.py

# 光催化反应
cd "../Photocatalytic reaction"
python dataset_generation_photocatalytic_reaction.py
python model_photocatalytic_reaction.py

# 模拟案例
cd "../Simulation cases"
python dataset_generation_simulation_cases.py
python model_simulation_cases.py
```

### 步骤4: 交叉验证分析

在`Fenton/main.py`中修改配置进行不同的交叉验证：

```python
# 污染物留一法交叉验证
type = 'pollutant'
for ID in range(12):
    train, test, data = data_init(type, ID)
    # ... 训练和评估

# 浓度条件交叉验证
type = 'concentration'
for ID in range(10):
    train, test, data = data_init(type, ID)
    # ... 训练和评估

# 学习曲线分析
type = 'learning curve'
for ratio in [0.2, 0.4, 0.6, 0.8, 1.0]:
    train, test, data = data_init(type, ratio)
    # ... 训练和评估
```

### 步骤5: 特征工程对比

测试不同分子描述符的性能：

```python
descriptors = ['RDKit', 'Morgan/Circular', 'MACCS keys', 
               'Morgan_WHIM', 'Morgan_Morse']

for fp in descriptors:
    data_encoded = encoder(data, fp)
    # ... 训练和评估
    # 比较不同描述符的性能
```

### 步骤6: 超参数优化

```python
# XGBoost超参数网格搜索
param_grid = {
    'max_depth': [5, 7, 9],
    'eta': [0.05, 0.1, 0.15],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2]
}

# 使用交叉验证寻找最优参数
best_params = grid_search(param_grid, train_data)
```

## 预期结果

根据论文，在Fenton反应系统上应获得以下性能：

### 整体性能（随机划分）
- **R²**: > 0.95
- **MAE**: < 0.05
- **RMSE**: < 0.08

### 污染物外推（LOPO）
- **R²**: > 0.85
- 对结构相似的污染物预测更准确

### 浓度外推（LOCO）
- **R²**: > 0.90
- 在已知污染物上的浓度外推性能良好

### 与参考模型对比
- MERML模型显著优于传统动力学模型
- 递归预测策略优于独立预测

## 文件输出说明

运行程序后会生成以下文件：

### Fenton反应系统
- `MCB`: 训练好的XGBoost模型（joblib格式）
- `descriptors.npy`: 分子描述符矩阵
- `cluster_result.txt`: 聚类分析结果
- 各种图表文件（.svg或.png格式）

### 模型文件
- 使用`joblib`保存和加载模型：
```python
from joblib import dump, load

# 保存模型
dump(model, 'MCB')

# 加载模型
model = load('MCB')
```

## 性能优化建议

### 1. 计算加速
- 使用GPU版本的XGBoost
- 减少SHAP计算的样本数
- 并行化交叉验证

### 2. 内存优化
- 使用`float32`代替`float64`
- 批量处理大数据集
- 及时释放不用的变量

### 3. 模型优化
- 使用网格搜索或贝叶斯优化调参
- 尝试集成多个模型
- 特征选择减少冗余特征

## 扩展应用

MERML框架可扩展到其他化学反应系统：

### 1. 新反应系统
- 准备类似格式的数据（时间序列 + 反应条件）
- 生成合适的分子描述符
- 调整时间采样点列表
- 运行MERML模型

### 2. 自定义特征
- 在`preprocess.py`中添加新的描述符函数
- 更新`switch`字典
- 在`encoder`函数中使用新描述符

### 3. 其他机器学习算法
- 替换XGBoost为其他算法（Random Forest, Neural Network等）
- 保持递归预测框架不变
- 对比不同算法的性能

## 引用

如果您使用本代码或方法，请引用相关论文：

```bibtex
@article{MERML2024,
  title={Data-Driven Recursive Kinetic Modeling for Fenton Reaction},
  author={作者信息},
  journal={期刊名称},
  year={2024}
}
```

## 许可证

请参考项目根目录下的LICENSE文件。

## 联系方式

如有问题或建议，请通过以下方式联系：
- GitHub Issues: [提交问题](https://github.com/tlqtangok/MERML/issues)
- Email: [根据论文填写]

## 更新日志

### 版本信息
- **当前版本**: 根据论文发表时间确定
- **最后更新**: 2024年

### 已知限制
1. 仅支持单一产物或单一反应物的动力学预测
2. 时间序列数据需要均匀或规律采样
3. 对于高度非线性的反应可能需要更多训练数据
4. 分子描述符的质量直接影响预测性能

## 致谢

感谢所有为本项目提供数据和建议的研究人员。

---

**注意**: 本README基于仓库代码和论文内容编写。具体的运行结果和性能指标需要实际运行代码后确认。如发现文档与代码不一致的地方，请以代码为准，并欢迎提交Issue或PR进行更正。

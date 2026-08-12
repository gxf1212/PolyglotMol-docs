# 指标语义表

> **本文件是 MolBlender 全包指标语义的唯一真相来源。**
> 任何 `src/molblender/metrics/`、`src/molblender/models/api/screening_engine/`、
> 仪表盘、CLI 引用指标前，必须先核对本表。
> 字段含义冲突时，以本表为准；不一致的代码视为缺陷。

## 0. 命名约束

- **指标 key（小写蛇形）** 是稳定 API，不允许别名。
- **数学含义只能由 key 唯一决定**。同名 key 不允许在回归 / 分类下表达不同含义。
- **方向（higher/lower-better）由 key 唯一决定**，禁止同一 key 在不同路径下方向不同。
- **显示名** 用于 UI，由 catalog 集中翻译，不在调用点散落。
- **计算函数** 在 `molblender/metrics/core.py`，名称与 key 一一对应；旧名只允许留 deprecated alias。

## 1. 回归指标

| key | 任务 | 数学含义 | 取值范围 | 方向 | 显示名 | 描述 | core 计算函数 | sklearn 对应 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `r2` | regression | **Pearson 决定系数（pearson_r²）** | $[0, 1]$ | 越高越好 | Pearson R² | Squared Pearson correlation coefficient (higher is better) | `calculate_pearson_r2` | 无内置（= `pearsonr(y, ŷ)[0] ** 2`） |
| `r2_score` | regression | **sklearn coefficient of determination** | $(-\infty, 1]$ | 越高越好 | R² Score | Coefficient of determination; can be negative when model is worse than mean baseline (higher is better) | `calculate_r2_score` | `sklearn.metrics.r2_score` |
| `pearson_r` | regression | Pearson 相关系数 | $[-1, 1]$ | 越高越好 | Pearson r | Pearson correlation coefficient, range [-1, 1] | `calculate_pearson_r` | `scipy.stats.pearsonr(y, ŷ)[0]` |
| `pearson_r2` | regression | Pearson 相关系数平方 | $[0, 1]$ | 越高越好 | Pearson R² | Squared Pearson correlation coefficient (higher is better) | `calculate_pearson_r2` | 无内置 |
| `rmse` | regression | 均方根误差 | $[0, \infty)$ | 越低越好 | RMSE | Root Mean Square Error (lower is better) | `calculate_rmse` | `sklearn.metrics.mean_squared_error(squared=False)` |
| `mae` | regression | 平均绝对误差 | $[0, \infty)$ | 越低越好 | MAE | Mean Absolute Error (lower is better) | `calculate_mae` | `sklearn.metrics.mean_absolute_error` |
| `mse` | regression | 均方误差 | $[0, \infty)$ | 越低越好 | MSE | Mean Square Error (lower is better) | `calculate_mse` | `sklearn.metrics.mean_squared_error` |
| `mape` | regression | 平均绝对百分比误差 | $[0, \infty)$ | 越低越好 | MAPE | Mean Absolute Percentage Error (lower is better) | `calculate_mape` | 无内置 |
| `medae` | regression | 中位绝对误差 | $[0, \infty)$ | 越低越好 | MedAE | Median Absolute Error (lower is better) | `calculate_medae` | `sklearn.metrics.median_absolute_error` |
| `max_error` | regression | 最大绝对误差 | $[0, \infty)$ | 越低越好 | Max Error | Maximum absolute error (lower is better) | `calculate_max_error` | `sklearn.metrics.max_error` |
| `msle` | regression | 均方对数误差 | $[0, \infty)$ | 越低越好 | MSLE | Mean Squared Logarithmic Error (lower is better) | `calculate_mean_squared_log_error` | `sklearn.metrics.mean_squared_log_error` |
| `rmsle` | regression | 均方根对数误差 | $[0, \infty)$ | 越低越好 | RMSLE | Root Mean Squared Logarithmic Error (lower is better) | — | `sklearn.metrics.mean_squared_log_error(squared=False)` |
| `normalized_rmse` | regression | 归一化 RMSE | $[0, \infty)$ | 越低越好 | Normalized RMSE | Normalized RMSE (lower is better) | — | — |
| `cv_rmse` | regression | 变异系数 RMSE | $[0, \infty)$ | 越低越好 | CV RMSE | Coefficient of variation RMSE (lower is better) | — | — |
| `mean_percentage_error` | regression | 平均百分比误差 | $(-\infty, \infty)$ | 越低越好 | MPE (%) | Mean Percentage Error (lower is better) | — | — |
| `spearman_rho` | regression | Spearman 秩相关 | $[-1, 1]$ | 越高越好 | Spearman ρ | Spearman rank correlation coefficient (higher is better) | — | `scipy.stats.spearmanr` |
| `kendall_tau` | regression | Kendall τ | $[-1, 1]$ | 越高越好 | Kendall's τ | Kendall tau rank correlation coefficient (higher is better) | — | `scipy.stats.kendalltau` |
| `explained_variance_score` | regression | 解释方差 | $(-\infty, 1]$ | 越高越好 | Explained Variance | Explained variance regression score | — | `sklearn.metrics.explained_variance_score` |

## 2. 分类指标

| key | 任务 | 含义 | 范围 | 方向 | 显示名 | 描述 |
| --- | --- | --- | --- | --- | --- | --- |
| `accuracy` / `accuracy_score` | classification | 正确率 | $[0, 1]$ | 越高越好 | Accuracy | Classification accuracy (higher is better) |
| `f1` / `f1_score` | classification | F1 | $[0, 1]$ | 越高越好 | F1 Score | F1 score (higher is better) |
| `f1_macro` | classification | 宏平均 F1 | $[0, 1]$ | 越高越好 | F1 Score (Macro) | Macro-averaged F1 |
| `f1_micro` | classification | 微平均 F1 | $[0, 1]$ | 越高越好 | F1 Score (Micro) | Micro-averaged F1 |
| `f1_weighted` | classification | 加权 F1 | $[0, 1]$ | 越高越好 | F1 Score (Weighted) | Weighted-averaged F1 |
| `precision` | classification | 精确率 | $[0, 1]$ | 越高越好 | Precision | Precision (higher is better) |
| `recall` | classification | 召回率 | $[0, 1]$ | 越高越好 | Recall | Recall (higher is better) |
| `roc_auc` / `roc_auc_score` | classification | ROC AUC | $[0, 1]$ | 越高越好 | ROC AUC | Area under ROC curve (higher is better) |
| `pr_auc` | classification | PR AUC | $[0, 1]$ | 越高越好 | Precision-Recall AUC | Average precision (higher is better) |
| `matthews_corrcoef` / `mcc` | classification | Matthews 相关系数 | $[-1, 1]$ | 越高越好 | Matthews Correlation Coefficient (MCC) | MCC |
| `cohen_kappa` | classification | Cohen κ | $[-1, 1]$ | 越高越好 | Cohen's κ | Cohen kappa score |
| `log_loss` | classification | 对数损失 | $[0, \infty)$ | 越低越好 | Log Loss | Logistic regression loss |

## 3. 元 key（不参与回归 / 分类评估，但进入 result 字典）

| key | 含义 | 取值类型 | 方向 | 显示名 |
| --- | --- | --- | --- | --- |
| `cv_mean` | CV 平均 | float | 跟随主指标 | CV Mean Score |
| `cv_std` | CV 标准差 | float | 越低越好（基于主指标） | CV Std Dev |
| `cv_scores` / `cv_fold_scores` | 各折分数 | list[float] | 跟随主指标 | CV Scores |
| `primary_metric` | 主指标名 | str | — | Metric |
| `composite_score` | 复合分 | float | 越高越好 | Composite Score |
| `efficiency_ratio` | 效率比 | float | 越高越好 | Efficiency Ratio |
| `robustness_score` | 鲁棒性分 | float | 越高越好 | Robustness Score |
| `confidence_score` | 置信度 | float | 越高越好 | Confidence Score |
| `evaluation_mode` | 评估模式 | str ("regression" / "classification") | — | — |
| `train_r2` / `train_rmse` / `train_mae` / `train_pearson_r` / `train_spearman_rho` | 训练集指标 | float | 与对应 test 指标方向一致 | — |

## 4. 失败值（必须统一）

| 指标族 | 失败返回 | 备注 |
| --- | --- | --- |
| 所有 `r2` / `pearson_r*` / `*_score` 高优指标 | `0.0` | 数学无定义时，0 表示"完全无信号" |
| 所有 error 指标（`rmse` / `mae` / `mse` / `medae` / `max_error` / `msle` / `rmsle` / `normalized_rmse` / `cv_rmse` / `mean_percentage_error`） | `float("inf")` | 越大越差，无穷大对应"彻底失败" |

`NaN` 不允许出现在 `result_processor` 落盘前的 `all_metrics` 中；
若计算过程中产出 `NaN`，统一替换为上表失败值。

### 4.1 常量 target 与单样本（与 sklearn 的对齐与分歧）

`calculate_r2_score` 的核心原则是**健壮性 > 严格等价**：筛选链路里退化输入（常量 target、单样本、全 NaN 行）不应让整次评估崩溃。

**常量 target（≥2 样本）** — 遵循 sklearn `force_finite=True` 语义：

| 情形 | 期望值 | 说明 |
| --- | --- | --- |
| 常量 target + 完美预测（`ss_res == 0`） | `1.0` | 等价 sklearn `r2_score(..., force_finite=True)` |
| 常量 target + 任意误差（`ss_res > 0`） | `0.0` | 等价 sklearn `r2_score(..., force_finite=True)` |

**单样本输入（`size < 2`）** — 返回 `0.0`。sklearn 返回 `NaN` 并发出 `UndefinedMetricWarning`；
本实现选择返回确定的失败值（符合 §4 的失败值契约），而非让调用方处理异常。

**多元素常量 input**（如 `[5, 5] -> [5, 5]`）走常量 target 路径返回 `1.0`，与 sklearn 一致。

### 4.2 输入净化（与 sklearn 的有意分歧）

`molblender/metrics/core.py:_prepare_numeric_pair` 会静默丢弃 `NaN` / `±inf` 元素后做配对；
而 sklearn 在同一输入上会抛 `ValueError`。

这是有意的"健壮性 > 严格等价"选择，理由是筛选链路里 NaN/inf 是退化输入，不应让整次评估崩溃。
测试用例 `tests/metrics/test_core.py::TestCalculateR2Score::test_matches_sklearn_r2_score_on_finite_inputs` 仅校验**有限输入、≥2 样本**上的 sklearn parity；NaN/inf 和单样本输入不要求相等。

`calculate_r2`（旧名）已发 `DeprecationWarning`，警告内容明确指向 `calculate_r2_score` / `calculate_pearson_r2`；新代码不应再使用。

## 5. 关键不变量（机器可校验）

下列不变量由 `tests/metrics/test_core.py` 守护，任意一条失败即视为回归：

1. **key 唯一性**：`r2` 与 `r2_score` 的核心计算结果在 *相同* `y_true, y_pred` 上**不相等**（sklearn 的 `r2_score` 在仅偏移非零的预测上会偏离 Pearson R²）。
2. **范围不变量**：`r2 ∈ [0, 1]`；`r2_score ∈ (-∞, 1]`。
3. **方向不变量**：`r2` 与 `r2_score` 都属于 *越高越好*，均不在 `LOWER_IS_BETTER_METRICS`。
4. **任务不变量**：`r2` / `r2_score` / `pearson_r*` 都在 `REGRESSION_ONLY_METRICS` 中。
5. **函数不变量**：`calculate_pearson_r2(y, ŷ) == calculate_r2(y, ŷ)`（旧 `calculate_r2` 是 deprecated alias 指向 `calculate_pearson_r2`，每次调用都会发 `DeprecationWarning`）。
6. **失败不变量**：传入两个全为常量的数组（≥2 样本）时，回归指标返回 §4 失败值（`r2_score` 常量+完美 → 1.0，常量+误差 → 0.0；其余高优指标 → 0.0；error 指标 → `+inf`），不抛异常。单样本输入 → `0.0`。

## 6. 反向预测检验（必须通过的回归测试）

用 `y_pred = -y_true + const` 跑回归指标：

| 指标 | 期望值 | 含义 |
| --- | --- | --- |
| `pearson_r` | $-1.0$ | 完美反向 |
| `pearson_r2` | $1.0$ | 平方后无方向性 |
| `r2` | $1.0$ | 等同 `pearson_r2` |
| `r2_score` | $\ll 0$（如 $\approx -3.0$） | 远差于均值基线 |

此用例可以同时证明 `r2` 与 `r2_score` 的语义分离。`core.calculate_r2` 当前是 `calculate_pearson_r2` 的 deprecated alias，在此用例下正确返回 1.0。

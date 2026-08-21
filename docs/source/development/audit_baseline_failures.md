# Baseline Test Failure Audit

上次 HPO 双线收敛完成后，全量 `tests/models/api/` 仍有 18 个与本次迁移无关的失败。以下每个都经过 `git stash` 基线复测（HEAD `0cef184` 同失败），确认为预存在问题，不是本轮 HPO 改动引入。按根因归类，供后续拆 ticket。

复测命令：

```bash
git stash                        # 暂存工作区
PYTHONPATH=src python -m pytest tests/models/api/ -q --tb=line   # 观察同组失败
git stash pop                    # 恢复工作区
```

## 1. 可选依赖缺失（需环境安装或加 skip）

| 测试 | 根因 |
|------|------|
| `test_fusion_hpo_resume.py::test_fusion_cross_process_reconstruction_numerical_parity` | `PLIPInteractionFingerprint` 缺可选依赖 `plip`（`DependencyNotFoundError`） |

处理建议：这些测试应 `pytest.importorskip("plip")` 或标记 `@pytest.mark.optional_deps` 在 CI 跳过；生产代码行为正确。

## 2. DB schema 迁移缺口（stage1_result_links 表）

| 测试 | 根因 |
|------|------|
| `test_multimodal_base_helpers.py::test_sqlite_*`（4 个） | **已修复（2026-08-20）**：测试 fixture `_seed_sqlite_db` 未建 `stage1_result_links` 表、`model_results` 缺 `stage1_compatibility_key` 列，且两个 expected-model 测试用旧参数名 `expected_model_names` 与旧键序 `(model, repr)`。已补建表/列、改 `expected_candidates`（键序 `(repr, model)`）、seed 补 key。30 passed |

处理建议：schema bootstrap（`schema_bootstrap.py`）需为 `stage1_result_links` 补建表，或测试 fixture 显式跑 bootstrap；确认旧 DB 升级路径。

## 3. 重构残留的方法引用

| 测试 | 根因 |
|------|------|
| `test_standard_execution_parallel_policy.py::test_legacy_adapter_all_aligned_with_mapping` / `test_import_star_resolves_all_declared_names` | `standard_execution.py:94` 引用 `screening_runtime.preparation._derive_repr_output_type`，该函数已不存在（重构遗留） |
| `test_lead_sensitivity_eval.py::test_private_train_wrapper_delegates` | `lead_sensitivity` 模块无 `train_and_evaluate_combination_impl`（旧接口已改） |
| `test_lead_sensitivity_runtime_config.py`（2 个映射测试） | lead sensitivity 的 n_jobs→max_cpu_cores 映射与当前 runtime 契约不一致 |

处理建议：三处都是"代码仍引用已删除/改名符号"，直接修正引用或补齐别名；`_derive_repr_output_type` 若不再需要应删除对应广告声明。

## 4. 导出/边界一致性

| 测试 | 根因 |
|------|------|
| `test_multimodal_service_boundary.py::TestServiceBoundary::test_services_all_exports_only_factories` | `services.py` 导出集合含 `create_session` 等非工厂符号，与"只导出工厂"契约不符 |
| `test_classification.py`（3 个：registry multiclass / permissive concrete→generic / data_handler split） | registry 对 multiclass/generic 兼容的声明与实际解析不一致 |

处理建议：service boundary 决定是否扩契约或删导出；classification 三测需按 registry 语义收敛。

## 5. fusion CV-only 数值路径

| 测试 | 根因 |
|------|------|
| `test_cv_only_fusion_ba.py::test_cv_only_fusion_two_stage_mask_projection` | CV-only fusion 评估在 3 折中 pearson 非有限（`EvaluationError`），两阶段 mask 投影数据量太小导致 fold 内常数目标 |

处理建议：这是数值合理失败（fold 内目标常数 → pearson 未定义），应允许 evaluation 层跳过非有限 fold（现有 `non-finite` 分支）并在测试中放大数据或断言补充帧跳过。

## 处理方式

本轮 HPO 迁移不触碰上述任一模块的改动加载面；它们应作为独立 ticket 按类修复，而不是与 HPO 收敛混在同一批。
## 2026-08-20 收口追加（HPO 收敛第二提交后）

### 已完成

- **Stage-1 effective-result SQL 收口**：新增 `persistence/stage1_results.stage1_exists_effective_result()` 作为 Stage-1 native+linked dedup exists 的唯一 owner（fail-closed on 缺 `stage1_compatibility_key`，可选 best_params 精确匹配）。`persistence/result_queries.py` 两个 exists 查询的 Stage-1 分支改为委托；`utils/database/sessions.py::check_existing_result_record`（无生产调用方的旧 facade）删除内联 SQL 改为统一委托。历史 SQL 从 3 份减为 1 份。
- **测试收集冲突**：`tests/models/api/screening_engine/test_combination_contract.py` 与 `multimodal/routing/` 同名导致 pytest import 冲突（`import file mismatch`）；已重命名为 `test_combination_canonical_owner.py`。
- **NestedCV HPO 无效分数测试适配**：`test_nested_cv_hpo_typed_contract.py` 的 nan/inf/非数值构造改为断言 `OptimizationResult` 构造层抛 `ValueError（finite or None）`；skipped case 补 `legacy_scorer_space=True` 标记。

### 本轮指标语义收敛（metrics 多源 → canonical bundle）

- `models/api/lead/lead_sensitivity_eval.py::_compute_regression_metrics`：弃用 scipy.pearsonr / sklearn r² 直算，改调 `molblender.metrics.core` 的 `calculate_pearson_r/rmse/mae/r2_score`。
- `drawings/model/metrics.py::calculate_regression_metrics`：R²/RMSE/MAE/MSE/Pearson/MAPE/residual 全部走 `metrics.core`，不再直接 `np.corrcoef` / sklearn。
- `drawings/model/regression.py`（两处 `plot_regression_results` / publication 绘制）：同一 canonical bundle。
- 保留：`dashboard/components/model_inspection/export.py` 内 3 处 scipy/sklearn 片段是**导出给用户的复现代码模板**（代码字符串，非运行时计算），维持 sklearn 教学风格；`dashboard/components/utils/calculations.py` 早已 re-export `metrics.core`。
- 覆盖：`test_lead_sensitivity_eval.py`、`tests/drawings/` 全过（`test_private_train_wrapper_delegates` 除外，见 #3 预存在）。

## 2026-08-20 收口追加（第三轮：metrics 与 Stage-1 exists 语义修复）

- **`check_existing_result_record()` 恢复旧按名语义**：旧 facade 无 `stage1_compatibility_key`，此前委托后 Stage-1 永远返回未命中。现 `stage1_exists_effective_result()` 增加 `allow_name_only` 参数——canonical 调用默认 fail-closed（缺 key 返回 False），旧 facade 显式传 `allow_name_only=True` 恢复历史按名跳过行为；新生产调用必须传 key。
- **engine scorer 未定义→NaN**：`evaluation/utilities.py::pearson_r_score/pearson_r2_score` 空数组/长度不匹配/全 NaN 由 `0.0` 改为 `float("nan")`，计算委托 `metrics.core`；移除 `scipy.stats.pearsonr` 导入（`evaluator.py` 残留导入一并删除）。
- **multimodal 训练指标 canonical bundle**：`modality_handlers/base_helpers.py` 的 train pearson/r2/rmse/mae 直算改为 `metrics.core`，`np.isnan(pearsonr) → 0.0` 的强制归零删除（未定义保留 NaN）。
- **`models/api/export.py::_calculate_metrics`**：真实运行时计算改为 core bundle（其第 233 行 scipy 片段为复现代码模板字符串，保留）。
- **测试断言修正**：`test_trial_execution.py` 的 scorer-space Pearson 断言范围从 `[0,1]` 放宽到 `[-1.000001, 1.000001]`（Pearson 可负，允许浮点噪声）；此前 `-1.02e-18` 被误判失败。
- **验证**：HPO + persistence + multimodal + drawings + lead 761 passed，`test_private_train_wrapper_delegates` 仍为 #3 预存在。

### 追加：fusion DB resume + multimodal base_helpers 修复（2026-08-20 第三轮）

- **`test_multimodal_base_helpers.py`（4 个）全部修复**：fixture `_seed_sqlite_db` 补建 `stage1_result_links` 表 + `model_results.stage1_compatibility_key` 列；两个 expected-model 测试改用新参数名 `expected_candidates`（键序 `(repr, model)`）并让 seed 写入匹配 key。30 passed。
- **`test_fusion_db_resume_smoke.py`（2 个）修复**：Stage-1 skip-existing 依据 fail-closed 契约需要 `stage1_compatibility_key`。`_make_fusion_result` 补 `stage1_compatibility_key="fusion-compat-key-1"`、`check_existing_result` 传 key、`_filter_models_to_evaluate` 注入 `_stage1_candidate_keys`（生产路径由候选规划注入）。35 passed。

## 2026-08-20 收口追加（第四轮：codex 复审两项整改）

### allow_name_only 撤销 → Stage-1 exists 全部 fail-closed

- **`stage1_exists_effective_result` 删除 `allow_name_only` 参数**：缺 `stage1_compatibility_key` 一律返回 False（不按名 skip）。同名不同 config/split 的行再也不会误命中——这正是 key 机制要消灭的语义，不再存在按名逃生口。
- **`utils/database/sessions.py::check_existing_result_record`（旧 facade，无生产调用方）**：新增可选 `stage1_compatibility_key` 参数。无 key 的 Stage-1 查询发 `DeprecationWarning` 并返回 False；带 key 时委托 canonical exists。Stage 2+ / check_all_sessions 行为不变。
- **反例测试** `tests/models/api/persistence/test_result_queries.py::TestLegacyFacadeFailClosed`（3 个）：native/linked 同名不同 key 行在无 key 时不 skip、带匹配 key skip；Stage 2+ 无 key 不受影响。
- 逐项对照 codex 方案 1-4 完成。

### metrics 继续收口（dashboard + engine scorer → core）

- **`metrics.core` 新增 `calculate_explained_variance_score`**：population variance，零方差目标返回 NaN（未定义，不返回 0.0），完美拟合常数目标返回 1.0。随 `metrics.__init__` 导出。
- **`dashboard/data/metrics.py::calculate_additional_metrics`**：R²/RMSE/MAE/MSE/MAPE/MedAE/max_error/Pearson r/r²/explained variance 全部改委托 `metrics.core`，删除手写 `np.corrcoef`/`np.median`/mask-MAPE 实现。MAPE 语义随之从"跳过零真值"改为 core 的 epsilon 保护（*100 百分比保持一致）。
- **`evaluation/metrics.py::get_scoring_function` 回归分支**：rmse/mae/mse/medae/max_error/neg_* scorer 从 sklearn 直算 + 手写 `np.sqrt`/`ravel` 改为 `core` 的 `calculate_*`（负号包装维持 scorer 空间）；`r2_score` key 显式绑定 `calculate_r2_score`（与 `r2`/`pearson_r2` 的 squared-Pearson 语义区分）。移除 `mean_squared_error/mean_absolute_error/median_absolute_error/max_error/r2_score` 的 sklearn 导入。
- 验证：`evaluation/` 159 passed；`test_classification.py`（3 个 baseline 失败修复：`_supports_task`→模块级 `supports_task`、DataSplitter patch 目标改 data_handler 命名空间、split 返回 keys 对齐 `X_train/X_test/...` 真实契约）31 passed；`test_cv_only_fusion_ba.py` 4 passed（#5 fusion 数值路径随迁移已过）。

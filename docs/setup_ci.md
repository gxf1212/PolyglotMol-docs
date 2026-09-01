# CI/CD 设置指南 (CI Setup)

本文件说明如何在本地搭建与 GitHub Actions 一致的 CI 验证环境，以及如何解读 CI
工作流。CI 冒烟测试 (`tests/ci/test_ci_smoke.py`) 会校验本文件与
`scripts/run_ci_checks.sh` 的存在性，因此它们属于仓库 CI 契约的一部分。

## 前置条件 (Prerequisites)

- Python 3.9 / 3.10 / 3.11（见 `pyproject.toml` 的 `requires-python = ">=3.9"`）
- `pip` 与可写的虚拟环境（推荐 `venv` 或 `conda`）
- 可选：用于 DL 表征（UniMol / PLM / ODDT）或 GPU 任务的外部模型权重与环境变量

## 本地安装 (Local install)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

> 历史说明：CI 曾使用 `.[dev,test]`，现以 `pyproject.toml` 中声明的
> optional-dependencies 为准；以 `.[dev]` 安装即可获得测试与质量工具。

## 运行全部检查 (Run all checks)

```bash
./scripts/run_ci_checks.sh          # 执行 smoke 测试 + ruff + mypy
./scripts/run_ci_checks.sh --help   # 打印用法
```

该脚本等价于 `.github/workflows/ci.yml` 中的 fast-feedback 与 code-quality
两步，便于推送前在本地复现。

## 测试分类 (Test categories)

`pyproject.toml` 在 `[tool.pytest.ini_options]` 中声明了以下标记，可用 `-m` 或
`-k` 精确选取：

- `smoke` —— CI 门禁最先运行的健康检查（导入、结构、关键脚本）
- `unit` —— 单元级测试
- `integration` —— 需要数据库/外部组件的集成测试
- `slow` —— 耗时较长的测试（评估、HPO、多模态）

示例：

```bash
pytest tests/ -k "smoke" -v
pytest tests/models/api/persistence/ -m "unit or integration" -v
```

## 分层门禁 (Layering gate)

架构分层（foundation / domain / leaf）由 `tests/ci/check_layer_dependencies.py`
强制。运行它可在合并前发现 domain 层反向依赖：

```bash
pytest tests/ci/check_layer_dependencies.py -q
```

## 目标验收套件 (Target suites)

完成架构迁移验收时，至少运行以下套件并读取 passed/failed/skipped：

| 套件 | 命令 |
|---|---|
| 分层门禁 | `pytest tests/ci/check_layer_dependencies.py` |
| HPO | `pytest tests/models/api/screening_engine/hpo/` |
| evaluation | `pytest tests/models/api/screening_engine/evaluation/` |
| multimodal HPO | `pytest tests/models/api/multimodal/processors/hpo/` |
| persistence | `pytest tests/models/api/persistence/` |
| cache 安全 | `pytest tests/data/test_multimodal_cache_security.py` |
| 兼容迁移 | `pytest tests/models/api/test_compatibility_migration.py` |
| splitting | `pytest tests/data/splitting/` |

## 可选依赖与跳过 (Optional deps & skips)

缺少 GPU、UniMol/PLM/ODDT 权重或真实 RdRp 资产时，相关测试必须以结构化
`skip` reason 跳过，而非报错；资产存在但流程失败视为回归。

## 故障排查 (Troubleshooting)

- **smoke 失败 `test_registry_import`**：`MolBlenderRegistry` 兼容别名必须从
  canonical owner 导入（如 `FeaturizerCatalog` 已迁至 `.featurizer_catalog`）。
- **layering gate 失败**：domain 包不应运行时导入 `molblender.models` 以外的
  leaf 层；缓存不应反向依赖筛选实现。
- **cache 安全测试失败**：未信任缓存读取 pickle 必须抛 `CacheTrustError`，
  越出 trusted root 的路径必须被拒绝。
- **persistence 测试失败**：DB 损坏/锁冲突/SQL 编程错误必须可见，不得静默降级；
  仅已知旧 schema 才允许兼容降级。
- **mypy 非零**：CI 中以 `|| true` 非阻塞运行，不阻断合并；本地修复可逐步收敛。

# Docs

```
docs/
└── source/
    ├── _static/
    ├── _templates/
    ├── api/                      # API 参考 (RST + autodoc)
    │   ├── index.rst             # API 根索引
    │   ├── data.rst              # data 模块 API
    │   ├── models.rst            # models 模块 API
    │   └── representations/      # representations 包 API
    │       ├── index.rst         # representations 根索引
    │       ├── fingerprints.rst  # fingerprints 子模块 API
    │       ├── graph.rst         # graph 子模块 API
    │       ├── image.rst         # image 子模块 API
    │       ├── protein/          # protein 子包 API
    │       │   ├── index.rst
    │       │   └── sequence.rst
    │       ├── sequential/       # sequential 子包 API
    │       │   ├── index.rst
    │       │   ├── language_model.rst
    │       │   └── tokenizer.rst
    │       ├── spatial.rst       # spatial 子模块 API
    │       ├── temporal.rst      # temporal 子模块 API
    │       ├── topological.rst   # topological 子模块 API
    │       └── utils.rst         # utils 子模块 API
    ├── contributing.md           # 贡献指南
    ├── installation.md           # 安装指南
    ├── introduction.md           # 介绍
    ├── quickstart.md             # 快速入门
    ├── usage/                    # 用法指南 (Markdown)
    │   ├── index.md              # 用法 根索引
    │   ├── data.md               # data 模块用法
    │   ├── models.md             # models 模块用法
    │   └── representations/      # representations 包用法
    │       ├── index.md          # representations 用法索引
    │       ├── fingerprints.md   # fingerprints 子模块用法
    │       ├── graph.md          # graph 子模块用法
    │       ├── image.md          # image 子模块用法
    │       ├── protein/          # protein 子包用法
    │       │   ├── index.md
    │       │   └── sequence.md
    │       ├── sequential/       # sequential 子包用法
    │       │   ├── index.md
    │       │   ├── language_model.md
    │       │   └── tokenizer.md
    │       ├── spatial.md        # spatial 子模块用法
    │       ├── temporal.md       # temporal 子模块用法
    │       ├── topological.md    # topological 子模块用法
    │       └── utils.md          # utils 子模块用法 (如果需要)
    ├── conf.py                   # Sphinx 配置
    ├── index.rst                 # 文档主入口
    └── locales/                  # 翻译文件
        └── ...
```

To build the docs, run:

```bash
make html
# make clean
```


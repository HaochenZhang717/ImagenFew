# AI-READI Dataset Utils Summary

## 1. 当前 repo 里已经存在的 AI-READI 相关 utility code

### 1.1 Dataset loader

- `data_provider/datasets/aireadi.py`
  - 提供了 `AIREADIGlucose(**config)` 工厂函数。
  - 核心类是 `AIREADIGlucoseDataset`。
  - 它会读取已经处理好的 parquet 文件，并把每个时间序列切成长度为 `seq_len` 的滑动窗口。
  - 返回格式和 repo 里其他 dataset loader 一致：`__getitem__` 默认返回单个 sample tensor，shape 是 `(seq_len, 1)`。
  - 关键可配项：
    - `rel_path`，默认 `AI-READI-processed`
    - `flag`，支持 `train/val(valid)/test`
    - `seq_len`
    - `scale`
    - `window_stride`
    - `min_seq_len`
    - `drop_nan`
    - `return_metadata`

### 1.2 底层 preprocessing / windowing utility

- `utils/utils_data.py`
  - 已经有一套专门给 AI-READI glucose 用的辅助函数：
    - `resolve_aireadi_glucose_path(...)`
    - `read_aireadi_glucose_frame(...)`
    - `fit_aireadi_glucose_scaler(...)`
    - `normalize_aireadi_glucose_values(...)`
    - `load_aireadi_glucose_windows(...)`
  - 这套逻辑做了几件事：
    - 把 split 映射到 parquet 文件名：
      - `train -> glucose_train.parquet`
      - `val/valid -> glucose_valid.parquet`
      - `test -> glucose_test.parquet`
    - 在 train split 上拟合 `StandardScaler`
    - 对每一行的 `glucose` 数组做缺失值过滤和标准化
    - 用滑动窗口生成样本
    - 为每个窗口记录 metadata：
      - `row_idx`
      - `patient_id`
      - `start`
      - `end`

### 1.3 已经接入统一 data provider

- `data_provider/datasets/__init__.py`
  - 已经导出了 `AIREADIGlucose`

- `data_provider/data_provider.py`
  - 已经把 `AIREADIGlucose` 注册进 `data_dict`
  - 这意味着只要 config 里写：
    - `name: AIREADIGlucose`
    - `data: AIREADIGlucose`
  - 就可以走现有的 `get_train(...)` / `dataset_to_tensor(...)` / `data_provider(...)` 主流程

### 1.4 已有测试

- `tests/test_utils_data.py`
  - 已经覆盖了 AI-READI glucose 的关键 utility：
    - split path 解析
    - scaler 拟合
    - NaN 处理
    - `min_seq_len` 过滤
    - window 输出与 metadata


## 2. 当前 AI-READI 支持的真实范围

### 2.1 现在只支持 glucose，不支持完整 AI-READI

虽然 repo 根目录下有：

- `AI-READI-processed/glucose_train.parquet`
- `AI-READI-processed/glucose_valid.parquet`
- `AI-READI-processed/glucose_test.parquet`
- `AI-READI-processed/calorie_train.parquet`
- `AI-READI-processed/calorie_valid.parquet`
- `AI-READI-processed/calorie_test.parquet`

但代码里真正被读到的只有 glucose。

我没有在 repo 中找到任何 calorie loader、calorie preprocessing utility、或者统一的 `AIREADI` 多变量 loader。

### 2.2 当前输入数据格式是假设“已经处理好”的 parquet

现有代码不处理 raw AI-READI 原始数据。它直接假设下面这个目录已经存在：

- `/Users/zhc/Documents/PhD/projects/ImagenFew/AI-READI-processed`

并且 parquet 内部已经是“每一行一个 episode / patient sequence，每个字段都是 array”的结构。

从实际 parquet 看，`glucose_*.parquet` 的 schema 大致是：

- `glucose`
- `unit`
- `event_type`
- `source_device_id`
- `transmitter_id`
- `transmitter_time`
- `patient_id`
- `time_utc`
- `time_local`

其中每行都是 array-like 序列，当前 loader 只使用 `glucose`，以及 metadata 中的 `patient_id`。

### 2.3 还没有完整融入“其他 dataset 一样的统一训练名单”

虽然 `AIREADIGlucose` 已被注册到 `data_provider.data_dict`，但还没有被加入：

- `data_provider/combined_datasets.py` 里的 `dataset_list`

这会影响：

- dataset token / class label 的统一索引
- 多数据集联合训练时的 dataset name 映射
- conditional / visualization / sampling 等依赖 `dataset_list` 的流程

### 2.4 还没有对应 config 模板

我没有在 `configs/` 下面找到任何：

- `AIREADIGlucose.yaml`
- `AIREADI*.yaml`

也就是说它虽然“底层能加载”，但还没有像 `AirQuality`、`ETTh2`、`mujoco` 那样形成可直接运行的配置模板。


## 3. 对照你的目标，目前缺了哪些 utility code

你的目标是：

> 写一些用来处理 AI-READI dataset 的 code，让它能和其他 `data` 文件夹里面的文件被读取成一样格式的数据集。

按这个目标看，目前还缺下面这些部分。

### 3.1 缺 raw AI-READI -> repo 标准格式 的 preprocessing 脚本

这是最大缺口。

现在 repo 没有看到任何脚本负责：

- 读取 raw AI-READI 原始文件
- 按 patient / episode / session 分组
- 对齐时间轴
- 重采样到固定频率
- 处理缺失值
- 切分 train / valid / test
- 导出成当前 repo 可直接消费的格式

建议补一个类似下面职责的脚本：

- `scripts/prepare_aireadi.py`

建议输出两类目标格式中的一种：

1. 直接输出当前 `aireadi.py` 能读的 parquet 格式
2. 或者输出到 `data/AIREADI/...`，格式尽量和其他 dataset 一致，再由专门 loader 读取

### 3.2 缺统一的 AIREADI loader，而不是只支持 glucose

现在只有：

- `AIREADIGlucose`

但 AI-READI 目录里已经能看到 calorie parquet，说明后续很可能不止一种信号。

建议至少补以下其中一种：

- `AIREADICalorie`
- `AIREADI` 通用 loader，可通过 `target_col=glucose/calorie/...` 配置

如果目标是和 repo 里其他多变量 dataset 更一致，更推荐做通用版：

- 支持选择一个或多个 signal column
- 输出 shape 统一为 `(seq_len, channels)`
- glucose 单变量时 channels=1
- glucose+calorie 时 channels=2

### 3.3 缺 config 层接入

建议补至少一套最小可运行配置，例如：

- `configs/finetune/AIREADIGlucose.yaml`
- `configs/conditional_imagen_few/AIREADIGlucose.yaml`
- 如果要走 self-conditioned / simple_vae，也补对应版本

核心字段大概需要：

- `name: AIREADIGlucose`
- `data: AIREADIGlucose`
- `rel_path: AI-READI-processed`
- `window_stride`
- `min_seq_len`
- `drop_nan`
- `scale`

### 3.4 缺 `combined_datasets.py` 的 dataset_list 注册

如果你希望它和其他 dataset 一样参与：

- 统一 class label
- conditional generation
- sampling / visualize

那就需要把它加入：

- `data_provider/combined_datasets.py`

否则很多基于 dataset name -> index 的路径不会完整支持它。

### 3.5 缺对 AI-READI 特有 metadata 的更完整利用

当前 metadata 只回传：

- `row_idx`
- `patient_id`
- `start`
- `end`

但 parquet 里还包含：

- `time_utc`
- `time_local`
- `event_type`
- `source_device_id`

如果后面要做：

- patient-level split 检查
- 时间对齐
- 多模态拼接
- 可解释性分析

这些字段最好保留到更规范的 sample metadata 里，而不是只在 parquet 中存在。

### 3.6 缺 calorie 的 utility 和测试

当前 repo 里有 calorie parquet 文件，但没有看到：

- `resolve_aireadi_calorie_path(...)`
- `read_aireadi_calorie_frame(...)`
- `load_aireadi_calorie_windows(...)`
- `AIREADICalorieDataset`
- calorie 对应测试

这说明 calorie 这条支线目前基本还没写。

### 3.7 缺 “与其他 data 目录下数据完全同构” 的目录规范

目前 AI-READI 数据放在 repo 根目录：

- `/Users/zhc/Documents/PhD/projects/ImagenFew/AI-READI-processed`

但其他 dataset 更常见的是放在：

- `/Users/zhc/Documents/PhD/projects/ImagenFew/data/...`

如果你的目标是“和其他 data 文件夹里面的文件被读取成一样格式”，那最好统一到：

- `data/AIREADI/processed/...`

这样 config 里的 `datasets_dir` 和 `rel_path` 也会更一致。


## 4. 一个更贴近你预期的落地方案

如果目标是把 AI-READI 真正接成和现有 dataset 一样的标准数据源，我建议按下面优先级补：

### Phase 1: 最小可用

1. 保留现有 `AIREADIGlucoseDataset`
2. 把 AI-READI 数据目录迁到 `data/AIREADI/processed/`
3. 增加 `configs/finetune/AIREADIGlucose.yaml`
4. 把 `AIREADIGlucose` 加进 `combined_datasets.dataset_list`

这样可以先让 glucose 单变量跑通训练和采样流程。

### Phase 2: 完整 utility 化

1. 新增 raw preprocessing 脚本
2. 支持 calorie
3. 支持通用 `AIREADI` loader
4. 支持多变量输出 `(seq_len, channels)`
5. 增加更多测试

### Phase 3: 完全对齐现有数据体系

1. 统一目录结构到 `data/AIREADI/...`
2. 统一 config 命名方式
3. 接入 conditional / self-conditioned / eval 全流程
4. 补全 visualization / sample script 的 dataset 名称支持


## 5. 结论

结论很直接：

- 这个 repo 里已经有一部分 AI-READI utility code
- 但它目前只覆盖了 `glucose` 的“processed parquet -> window tensor”这一步
- 距离“像其他 `data/` 里的 dataset 一样完整接入”还差不少

最关键还没写的，是这三类：

1. raw AI-READI 数据预处理脚本
2. config + dataset_list 层面的正式接入
3. 除 glucose 外的更通用 AI-READI loader（尤其是 calorie / multi-signal）

如果你愿意，下一步我可以继续直接帮你把这套缺口补起来，优先做一个最小可用版本：

- 把 AI-READI 目录规范化
- 加 `AIREADIGlucose` config
- 把它接进 `combined_datasets.py`
- 再补一个通用版 `AIREADI` loader 骨架

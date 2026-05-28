# limit_grad_lora 说明文档

本模块实现的是基于 K-FAC 曲率信息的 LoRA 梯度投影方法。当前版本在原始
“高曲率方向投影保护”的基础上，新增了三类机制：

1. Newton 风格的逐方向投影强度。
2. 面向持续学习的 K-FAC 记忆融合。
3. 可选的泄露率和在线动态投影放松。

整体目标是：在模型编辑或下游任务适配时，尽量保护旧知识的重要方向，同时给新任务保留足够的可塑性。

## 整体流程

```text
预训练/基础 K-FAC
        +
可选的任务 K-FAC
        |
        v
按权重融合 K-FAC
        |
        v
对 A/B 做特征值分解
        |
        v
根据 energy_threshold 选择高曲率方向
        |
        v
对 LoRA 梯度做 Newton 加权投影
```

对于 LoRA 参数：

```text
lora_A：在输入/activation 方向上做投影。
lora_B：在输出/gradient 方向上做投影。
```

## K-FAC 记忆相关参数

这些参数定义在 `MyLoRAHyperParams` 中，并由 `build_lora_projection_cache` 使用。

### `base_kfac_cache_path`

上一轮或已有的 K-FAC 缓存路径，用作当前阶段的基础记忆。

在持续学习场景中，如果你已经有前几个任务融合后的 K-FAC 缓存，就应该把它作为
`base_kfac_cache_path`。如果不设置该参数，代码会回到默认逻辑：使用
`mom2_dataset` 计算或加载预训练 K-FAC。

示例：

```yaml
base_kfac_cache_path: "llama3-8b/merged_kfac/task1.pt"
```

含义：

```text
使用 task1 的累计 K-FAC 记忆作为旧知识保护锚点。
```

### `task_kfac_cache_path`

当前下游任务的 K-FAC 缓存路径。

这个缓存应该和预训练 K-FAC 分开计算。这样做很重要，因为下游任务可能是分类、
排序、多标签任务或其它非语言建模目标，不能简单地和预训练文本数据混在同一个
dataloader 中。

期望的缓存格式：

```python
{
    layer_name: {
        "A": A_tensor,
        "B": B_tensor,
        "N": token_or_sample_count,
    }
}
```

也支持：

```python
{
    "stats": {
        layer_name: {
            "A": A_tensor,
            "B": B_tensor,
            "N": token_or_sample_count,
        }
    }
}
```

### `task_kfac_weight`

当前任务 K-FAC 融合进基础 K-FAC 的权重。

融合公式：

```text
A_new = (1 - w) * A_base + w * A_task
B_new = (1 - w) * B_base + w * B_task
```

其中：

```text
w = task_kfac_weight
```

推荐初始值：

```yaml
task_kfac_weight: 0.05
```

或者：

```yaml
task_kfac_weight: 0.1
```

该值越大，投影矩阵越偏向当前任务；但过大时可能削弱对旧知识的保护。

### `task_kfac_tag`

任务名称标签，用于自动生成融合 K-FAC 缓存文件名。

示例：

```yaml
task_kfac_tag: "classification"
```

### `merged_kfac_cache_path`

融合后的 K-FAC 缓存保存路径。

如果该文件已经存在，并且 `force_recompute=False`，代码会直接加载这个融合缓存，
避免重复融合和重复计算。

示例：

```yaml
merged_kfac_cache_path: "llama3-8b/merged_kfac/wiki_cls_w0.1.pt"
```

持续学习中推荐的使用方式：

```text
任务 1：
base = 预训练 K-FAC
保存 -> task1.pt

任务 2：
base_kfac_cache_path = task1.pt
task_kfac_cache_path = task2.pt
保存 -> task1_task2.pt

任务 3：
base_kfac_cache_path = task1_task2.pt
task_kfac_cache_path = task3.pt
保存 -> task1_task2_task3.pt
```

这样做的意义是：不要用最新任务的投影矩阵直接替换旧投影矩阵，而是维护一个不断累积的
K-FAC 记忆，从而让旧任务的重要方向继续受到保护。

## Newton 投影参数

### `newton_damping`

控制 Newton 风格投影权重的平滑程度。

第 `i` 个方向的删除强度为：

```text
delete_i = (1 - leak) * lambda_i / (lambda_i + damping * lambda_max)
```

其中：

```text
lambda_i   = 被选中 K-FAC 方向的特征值
lambda_max = 被选中特征值中的最大值
damping    = newton_damping
```

含义：

```text
特征值大的方向     -> 删除更强
特征值小的方向     -> 删除更柔和
newton_damping 越大 -> 整体投影越保守
newton_damping = 0  -> 退化为硬投影
```

推荐搜索范围：

```yaml
newton_damping: 0.0001
newton_damping: 0.001
newton_damping: 0.01
newton_damping: 0.1
```

建议第一个实验使用：

```yaml
newton_damping: 0.001
```

## 泄露率参数

### `use_leak`

是否启用泄露率机制。

如果关闭：

```text
leak = 0
```

如果开启：

```text
actual_leak = sigmoid(leak_rate_param) * leak_rate
```

其中 `leak_rate_param` 默认初始化为 `-4.0`，所以：

```text
sigmoid(-4.0) ~= 0.018
```

因此，如果配置为：

```yaml
leak_rate: 0.2
```

那么初始实际泄露率大约只有：

```text
0.018 * 0.2 ~= 0.0036
```

### `leak_rate`

泄露率的上限缩放系数。

泄露率允许高曲率方向有一小部分梯度通过。它可能提高编辑成功率，但也可能削弱
locality、generality 或旧知识保持能力。

推荐第一个实验先关闭：

```yaml
use_leak: false
```

如果确实需要提高新任务可塑性，可以尝试：

```yaml
use_leak: true
leak_rate: 0.05
```

不建议在已经启用 Newton 投影和任务 K-FAC 融合时，一开始就使用较大的泄露率。

## 动态投影参数

动态投影是一种持续学习式的在线放松机制。它会统计当前任务梯度在每个受保护方向上的使用强度。
如果某个方向被当前任务反复使用，就临时降低该方向的删除强度，让新任务更容易学习。

### `use_dynamic_projection`

是否启用在线任务自适应投影强度。

推荐第一个基线实验关闭：

```yaml
use_dynamic_projection: false
```

建议在 Newton-only 结果稳定后，再尝试打开。

### `dynamic_projection_beta`

用于跟踪当前任务方向使用强度的 EMA 系数。

公式：

```text
ema_t = beta * ema_{t-1} + (1 - beta) * current_direction_energy
```

该值越大，动态适应越慢，但更稳定。

推荐：

```yaml
dynamic_projection_beta: 0.95
```

### `dynamic_projection_strength`

当前任务使用某个方向时，对该方向投影删除强度的放松幅度。

该值越大，新任务可塑性越强，但对旧知识的保护可能越弱。

推荐保守设置：

```yaml
dynamic_projection_strength: 0.1
```

### `dynamic_projection_min_scale`

动态投影缩放的下限。

它用于防止某个方向因为被当前任务频繁使用而完全放开保护。

推荐：

```yaml
dynamic_projection_min_scale: 0.5
```

## 推荐实验顺序

### 1. 只使用 Newton 投影

这是最干净的基线。

```yaml
use_leak: false
newton_damping: 0.001
use_dynamic_projection: false
task_kfac_weight: 0.0
```

### 2. Newton + 当前任务 K-FAC 记忆

当你已经单独计算好了下游任务 K-FAC 缓存时，使用该设置。

```yaml
task_kfac_cache_path: "llama3-8b/task_kfac/classification.pt"
task_kfac_weight: 0.05
task_kfac_tag: "classification"
merged_kfac_cache_path: "llama3-8b/merged_kfac/wiki_cls_w0.05.pt"
use_leak: false
newton_damping: 0.001
use_dynamic_projection: false
```

### 3. 持续学习式 K-FAC 记忆

把上一轮融合后的缓存作为当前阶段的基础记忆。

```yaml
base_kfac_cache_path: "llama3-8b/merged_kfac/task1.pt"
task_kfac_cache_path: "llama3-8b/task_kfac/task2.pt"
task_kfac_weight: 0.05
merged_kfac_cache_path: "llama3-8b/merged_kfac/task1_task2.pt"
```

### 4. 加入轻量动态投影

只建议在前面实验已经稳定后再尝试。

```yaml
use_dynamic_projection: true
dynamic_projection_beta: 0.95
dynamic_projection_strength: 0.1
dynamic_projection_min_scale: 0.5
```

## 与 `decograd_newton.py` 的关系

两者都使用同一个 Newton 风格的特征值加权思想：

```text
lambda / (lambda + damping * lambda_max)
```

但算法语义不同：

```text
decograd_newton.py：
    在移除 retain 子空间后，对 forget 方向使用 Newton 权重做软删除。

limit_grad_lora：
    在 K-FAC 高曲率方向上使用 Newton 权重，目的是在 LoRA 编辑时保护旧知识。
```

因此，两者核心数学形式一致，但本模块使用它来做高曲率保护，而不是显式的
retain/forget unlearning。


# MoeEP 接收容量 API 的设计方向

> 状态：面向 TE 集成的设计提案。本文描述的 API 修改尚未发布。

## 容量术语

- `R`：由并行拓扑和输入上限推导出的原始路由数，通常为
  `ep_size * max_tokens_per_rank * top_k`。
- `L`：逻辑接收路由上限。路由溢出判断和上游 kernel preset 使用该值。
- `P`：经过 padding 的物理接收池行数。Kernel 根据 `L`、本地 expert 数量
  和 padding 粒度计算该值；workspace 和 WGrad operand 的 shape 使用该值。

`L` 与 `P` 有意保持不同：每个 expert 的 token 段分别进行 padding，因此
容量为 `L` 条逻辑路由时，可能需要超过 `L` 行的物理存储。

## 当前行为

公共参数 `MoeEpParallelConfig.max_recv_size_per_rank` 表示物理行数 `P`，
而 vendored kernel 中的同名参数表示逻辑上限 `L`。

因此，常规 backend 路径会：

1. 接受调用方提供的 `P`，或者计算默认的 `P`；
2. 重复实现 kernel 的 padding 规则，将 `P` 反解为 `L`；
3. 通过名为 `max_recv_size_per_rank` 的 kernel 参数传入 `L`；
4. 由 kernel 再次计算 `P`，并验证结果与请求的物理池大小一致。

部分上游优化 preset 会直接约束 `L`。由于当前公共 API 表达的是 `P`，
支持这些 preset 需要额外的 preset 专用处理。

调用方不负责分配 MoeEP 私有 workspace。调用 `prepare_training()` 后，
调用方根据返回的 `(shape, stride, dtype, alignment)` 合约分配其持有的
forward state 和 WGrad operand。

## 建议行为

显式暴露逻辑容量，例如命名为 `logical_recv_route_limit`，并将 prepared
kernel 作为物理容量的唯一权威来源：

1. 调用方提供 `L`，或者由 MoeEP 根据拓扑推导无裁剪的默认值；
2. MoeEP 将 `L` 原样传给 kernel；
3. Kernel 仅计算一次 `P`；
4. `prepare_training()` 返回由 prepared kernel 推导出的完整分配合约；
5. 调用方按照这些合约分配输出 bundle，然后依次执行 forward 和
   backward。

MoeEP 还可以额外暴露只读的 `resolved_physical_pool_rows`，用于日志记录
和显存规划。调用方仍应根据完整合约进行分配，而不应仅从 `P` 推导 tensor
shape，因为 scale layout、stride、dtype 和 alignment 同样属于 ABI。

为保持兼容，现有 `max_recv_size_per_rank` 可在弃用周期内继续保留物理
`P` 的含义。同时设置旧物理容量字段和新逻辑容量字段时，应直接报错。

## 示例

假设某个上游 preset 要求：

```text
L = 131072
本地 expert 数量 = 8
padding block = 128
```

Kernel 推导得到：

```text
P = 131968
```

当前 TE 调用方需要表达物理值 `131968`，再由 MoeEP 恢复逻辑值
`131072`。采用建议 API 后，调用方直接表达 `L=131072`；
`prepare_training()` 随后返回 pool 维度为 `P=131968` 的 WGrad operand
shape。

## 预期收益

- 公共容量参数与路由溢出语义及上游优化 preset 保持一致。
- 消除常规路径中的 `P -> L -> P` 往返转换及重复 padding 逻辑。
- 减少 preset 专用适配代码和本地 vendored-code overlay。
- 在 MoeEP 内部管理私有 workspace 的同时，为 TE 提供由 kernel 推导的
  精确分配合约。
- 让 compile key、诊断信息和容量错误都使用含义明确的单位。
- 未来 kernel 修改 padding 或 layout 时，TE 无需重新实现容量公式。

预期的职责边界为：TE 选择逻辑工作负载容量 `L`；kernel 负责物理布局
`P`；MoeEP 发布连接两者的精确分配 ABI。

# NB-Recall：面向具身大脑的可校准经验记忆——存什么、何时信任、何时重观察

> **Calibrated, Version-Aware Experience Memory for Embodied Brains: Knowing What to Store, When to Trust It, and When to Re-Observe.**
>
> **一句话（TL;DR）**：把一套 typed、带 provenance/版本/支持域的经验记忆，配上一个**显式区分"证据不足"与"记忆过期"的可校准信任闸门**；**论文重心（scientific claim）在具身侧**——在 FindingDory 上把闸门的第三支落为**低信任触发真实重观察动作**，闭合"记忆不确定→去看一眼→更新记忆"的感知-动作环，这是文本 RAG 结构上做不到的。LongMemEval-v2 **仅作受控文本 sanity check**（在固定 reader 下验证同一套记忆的检索质量），**不作论文成败的承重点**：榜上唯二可用杠杆（把信任信号注入固定 reader 上下文、按 τ 早停省延迟）偏薄且需先经 pilot 验证（§5.0），故即便 LAFS 结果平平也不拖垮全文。

---

## 1. Motivation（本方案的根基）

### 1.1 具身大脑的瓶颈不是推理容量，而是记忆的"可信度"

双系统（大脑 System-2 + 小脑 System-1）已成具身 AI 的默认架构：VLM/MLLM 负责慢速语言推理与规划，action-head 负责高频动作。大脑要做好规划，前提是它**记得对的事**：某物体上次在哪、某容器是否已清空、某前置条件是否满足、某操作顺序是否有坑。但"记得"在长程、跨 episode、部分可观的真实部署里有三个结构性难点，而现有系统基本回避：

1. **代价**：长上下文把全部历史塞进 context，代价随历史线性/平方增长。web-agent 轨迹（LongMemEval-v2 单条可达 115M token 级）或终身具身任务下不可承受。
2. **过期**：记忆中的事实会随时间失效——物体被移动、状态被更新、任务配置被改。一条"昨天正确"的记忆今天可能是错的。
3. **不可信而不自知**：相似检索返回的 top-k 直接进 reader，模型**没有显式判断"这条记忆现在是否还成立"**。最伤的恰恰是"高相似但错"的近邻（semantically adjacent but wrong），阈值 gap 预警会把它们直接放行。

一句话：**相似度不是置信度**（similarity ≠ confidence）。文档 QA 的 RAG 优化的是排序；而 agent/具身记忆真正要优化的是**可校准的弃答与复核**——知道自己何时不知道，并在能复核时去复核。

### 1.2 这是 SAGE 式"有证据支持才行动"原则在大脑层的投影

本组一贯主张"只在有证据支持的范围内改变行为，证据不足就主动获取"。在真机低层它表现为 Bayes-risk 的 observe/inspect。在大脑记忆层，等价问题是：**面对一个查询，记忆给出的答案是否落在它被证据支持（且未过期）的范围内？若否，应弃答或触发重观察，而不是自信地答错。** 这个问题便宜（文本/sim）、可刷榜、却几乎没人在"具身大脑"语境里正面做。

### 1.3 为什么是现在、为什么便宜

- **FindingDory（科学主场）**（arXiv `2506.15635`，ICLR 2026）：Habitat 真实感室内仿真里的具身记忆 benchmark，60 个导航任务、82,174 训练 / 5,876 验证实例，覆盖 spatial / temporal / multi-goal；官方指标含 HL-SR/HL-SPL、LL-SR/LL-SPL、SC-SR 及 DTG-SR。冻结大模型（Gemini-2.0-Flash、GPT-4o）几乎做不动，Qwen-SFT 在高层任务也仅 ~50%，multi-goal 近 0。**无 eval server**，我们可定义 SOTA 条目。这里"低信任→真实重观察"有用武之地，是本文承重的试验台。
- **LongMemEval-v2（受控文本 sanity check）**（xiaowu0162.github.io/longmemeval-v2；arXiv `2605.12493`）把 agent 记忆做成干净的 `Insert(trajectory)` / `Query(question)→context` 接口：轨迹顺序流入记忆，Query 返回的上下文被截断到固定 token 预算，再由**固定 reader** 作答。5 个能力维度（`static` / `dynamic` / `procedure` / `gotchas` / `overall_full_set`），两档（`small` / `medium`），榜单指标是 **LAFS Gain**，**外部榜目前基本空置**。榜规核验：reader 必须含 `qwen3.5-9b`，judge 必须含 `gpt-5.2`，提交需 web+enterprise 两域按样本数加权、打包 `.tar.gz` 经 Google Form 提交。**定位**：在固定 reader 下验证同一套记忆的检索质量与版本化机制；是便宜、可复现的文本对照，**不是论文成败承重点**（理由见 §5.0 的榜上杠杆限制）。

### 1.4 为什么必须是"我们这套"——可辩护的新颖性边界（经核验后收窄）

**重要前提**：截至 2026，"给检索记忆打校准置信 + 低置信弃答"本身**已拥挤**：

- **MMA**（Multimodal Memory Agent，arXiv `2602.16493`）：Confidence Module 对每条记忆给 `C(Mᵢ)∈[0,1]`，高置信优先、低置信标记待弃答；配套 MMA-Bench 在 source-reliability 先验与图文冲突下压测长程 belief revision，打分奖励校准弃答、惩罚过度自信——**几乎与朴素版 headline 重合**。
- **SABER**（Trust or Abstain?，arXiv `2605.18792`）：免微调的 Self-Aware Belief Estimator，4 格决策（信参数知识 / 信上下文 / 都信 / 弃答），其校准弃答 Pareto 优于 prompt-based abstainer。
- **Conformal Abstention for RAG**：分布无关、有限样本地保证"输出答案中不忠实比例 ≤ 指定风险"，阈值由数据按 upper confidence bound 选取。
- **Magnitude Mirage**（arXiv `2609.15578`）：指出推理型查询下原始检索分数的校准性退化。

因此**单纯"校准 trust + 弃答"不能作为贡献**。NB-Recall 必须把新颖性收窄到三点残差，并在 §9 用 prior-art 表逐条划清：

1. **具身物理重观察（核心差异）**：文本 RAG 的"低置信"至多触发**再检索一跳**；本文把闸门第三支落为**真实的重观察动作**——在 FindingDory 里移动/靠近/重新交互以获取新证据，闭合"记忆不确定 → 去看一眼 → 更新记忆"的感知-动作环。这是上述所有文本/多模态弃答工作**结构上不具备**的决策分支，也是把问题从"RAG 弃答"抬到"具身大脑"的关键。
2. **动态事实版本化 × 校准耦合**：对随时间变化的事实做显式版本化（`t_valid_from/to`），查询按参考时刻解析到**时间正确**的版本；信任头**把"过期（stale）"从"证据不足（low-evidence）"里拆出来单独校准**，而非笼统打一个置信分。MMA 做冲突下的 belief revision，但不做"查询时刻 → 版本解析"与"过期 vs 不足"的分离校准。
3. **单一大脑 text→embodied 迁移**：同一套记忆大脑既在 LongMemEval-v2 上拼 LAFS，又迁到 FindingDory 的空间/时序记忆，证明是**通用大脑子系统**而非榜单特化。

一句话定位：**榜上（LongMemEval-v2）拼 LAFS，可辩护的科学贡献在具身侧**——正是现有 RAG 弃答工作触不到的地方。

---

## 2. 问题定式

### 2.1 记忆接口与目标

记忆系统实现两个接口：

- `Insert(x_t)`：消费一条到来的经验（文本轨迹片段 / 具身观测帧），更新内部记忆 `M`。
- `Query(q, t_q) → (ctx, τ, decision)`：对查询 `q`（参考时刻 `t_q`）返回：检索上下文 `ctx`（截断到固定预算 `B_tok`）、信任标量 `τ∈[0,1]`、以及三元决策 `decision ∈ {answer, re-observe, abstain}`。

固定 reader `R`（LongMemEval-v2 为 `qwen3.5-9b`）对 `ctx` 作答 `a = R(q, ctx)`；judge `J`（`gpt-5.2`）判对错。

**两类目标分开，且主次明确**（这是全文的工程纪律）：

- **主（具身科学目标）**：FindingDory 上"低信任→真实重观察"的成功率/路径效率；risk–coverage（选择性准确率 vs 覆盖率）、abstention 质量、dynamic 事实随 fact-change 率的准确率。**论文成败由此承重**。
- **次（文本 sanity check / 可选榜单）**：LongMemEval-v2 的 **LAFS Gain**——以 `accuracy = overall_full_set×100`、`latency = memory_query_avg_seconds` 为两轴，相对固定参考前沿的前沿增益。**关键限制**：榜上没有原生 abstain 动作、reader 固定，对"可答题"弃答即计错；唯二杠杆是**(a) 把 τ/冲突信号写进 `ctx`** 诱导固定 reader 在该弃答的题（gotchas/premise 类）上弃答、**(b) 按 τ 早停**省延迟。杠杆 (a) 是否生效**取决于固定 9B reader 是否服从注入提示**，这一点未经证实，故设 §5.0 pilot 作 go/no-go；LAFS 不达标**不**判全文失败。

### 2.2 staleness / conflict 模型（供理论与数据生成共用）

- 每个事实 `f` 有真值版本序列 `{(v_k, [s_k, e_k))}`，`[s_k,e_k)` 为该版本的有效区间。查询 `(q,t_q)` 命中 `f` 时，**正确答案 =** `t_q∈[s_k,e_k)` **的那个版本** `v_k`。
- **版本盲检索**（按相似度）命中过期版本的概率记 `p_stale`（因为相似度不 condition on 时间）。命中过期 → 答错。
- **冲突**：两条观测对同一 `(entity, attr)` 给出不一致值，来源可靠性先验不同。
- **重观察信息量** `ρ∈(0,1]`：在具身侧，对处于"中间带"的查询执行一次重观察，能把版本解析正确的概率 ≥ ρ，代价 `c_obs`（导航步数/时间）。

---

## 3. 方法

### 3.1 Typed 经验记忆数据结构

记忆为一张带类型节点的图 `M=(V,E)`：

```yaml
MemoryNode:
  id: str
  type: Entity | Fact | Event | Procedure
  content: value | text | embedding | frame_ref
  provenance:
    source: trajectory_id | sensor
    time: observed_at            # 观测时刻
    episode: str
    action: optional             # 由什么动作获得（具身侧）
    observation_ref: uri_or_hash # 可复核指针（高影响证据保原始帧/哈希）
  validity:
    state: valid | stale | disputed
    t_valid_from: time
    t_valid_to: time | null      # null = 当前仍有效
  support:                       # 该记忆被证据支持的上下文域
    scope: {entity, scene, task_condition}
  trust: float                   # 学习得到的校准信任（见 3.4）
Edges: generated-by | before/after | supports | occludes | same-entity | version-of
```

- **Entity**：持久实体（物体/人/地点），携实例身份。
- **Fact**：`(entity, attr, value)` 三元组，**版本化**（`version-of` 边串起同一 fact 的多版本）。
- **Event**：带时间戳的交互/观察事件，`before/after` 边支撑时序/顺序问题。
- **Procedure**：workflow 知识（步骤/前置条件/坑），服务 `procedure`/`gotchas` 能力维。

### 3.2 写入策略（存什么）：DROP / WRITE / MERGE / UPDATE

每条 `Insert(x_t)` 先抽取候选 typed 节点，写入控制器输出四类操作之一：

- **DROP**：冗余或低效用（与已存高 trust 节点语义重合且无新信息）。
- **WRITE**：新实体/新事实/新事件。
- **MERGE**：同 acquisition group、同实体概率高、likelihood 兼容 → 合并（保留 provenance 分叉，不覆盖、不平均）。
- **UPDATE**：检测到某 Fact 的**新版本** → 新增 `(v_{k+1}, [t,null))` 并把旧版本 `t_valid_to` 置为 `t`。**这是动态事实的核心**：UPDATE 不是覆盖，而是版本演进。

写入控制器以一个小模型（或规则 + 轻量分类器）实现，损失含 cost-sensitive 操作分类 + 效用排序（hindsight：固定 future policy 下 leave-one-node-out 的 cost-to-go 边际）+ 预算/因果一致性正则。冲突时写 `disputed` 并保留 mixture，交由检索期的校准传播，而非提前压成单值。

### 3.3 版本感知的动态事实解析

`Query(q, t_q)` 命中 Fact `f` 时：

1. 抽取 `q` 的**参考时刻** `t_q`（显式日期 / "现在" / "上次……之后" 等相对表达，用小解析器 + reader 零样本兜底）。
2. 在 `f` 的版本链中选 `t_q∈[s_k,e_k)` 的版本；若 `t_q` 落在多版本边界或版本缺失 → 标记为需复核（进中间带）。
3. 把**选中版本 + 其有效区间 + 是否存在更新版本**一并写入 `ctx`，让固定 reader 看到"这条事实的时间边界"。

### 3.4 可校准信任头（何时信任）——headline 机制

对每个检索候选，信任头输出 `τ = g(语义匹配, provenance 新鲜度, 版本一致性, 冲突度, 支持域覆盖, 来源可靠性先验)`。关键设计：

- **"过期"与"证据不足"分头建模**：`τ = σ(w_e·e_evidence + w_s·s_staleness + ...)`，其中 `s_staleness` 专门吃版本/时间特征。输出两个可解释子分：`τ_evidence`（证据是否充分）与 `τ_fresh`（是否未过期），整体 `τ = τ_evidence · τ_fresh`。这让"相似但过期"被 `τ_fresh` 压低，而非被整体相似度放行。
- **校准**：用 held-out 校准集做 temperature scaling 得初步 `τ`；再用 **conformal** 在每批审计切片上给分布无关覆盖保证（对标 Conformal Abstention，但我们校准的是**版本感知**的 τ，不是 raw 相似度）。数据足够时对比 isotonic。
- **报告 u-ordering 先于一切**：先验证 `τ` 与实际正确性的 rank-correlation 显著 >0（见 K1），否则整套防线不成立，先修 τ。

### 3.5 三元决策：answer / re-observe / abstain

阈值 `τ_lo < τ_hi` 按业务代价（答错 vs 弃答 vs 复核）预注册：

- `τ ≥ τ_hi` → **answer**（把选中版本 + 边界写进 `ctx`）。
- `τ_lo ≤ τ < τ_hi` → **re-observe**：
  - *文本侧（LongMemEval-v2）*：退化为"再检索一跳 / 扩展邻域 / 调出原始证据指针"，并把"证据有冲突/可能过期"的提示写进 `ctx`——**诱导固定 reader 在该弃答的题上弃答**（这是榜上唯一能影响 abstention 的合法杠杆）。
  - *具身侧（FindingDory）*：触发**真实重观察动作**（导航靠近 / 换视角 / 重新交互），用新观测 `Insert` 后重解析。**这是本文独有的分支**。
- `τ < τ_lo` → **abstain**（文本侧写入"无足够证据"提示；具身侧选择 no-op / 请求澄清）。

### 3.6 迁移到具身（FindingDory）

- egocentric 观测流作为 `Insert` 源；节点加空间字段（object id、pose、room）；Event 的 `before/after` 边服务 temporal/multi-goal 顺序问题。
- 大脑结构、写入策略、信任头、三元决策**完全复用**，只换感知前端与节点 schema——证明通用性。
- FindingDory 为两段式（oracle 收集经验 → agent 据记忆行动）；我们在"据记忆行动"段把低 τ 的子目标接到重观察动作。

### 3.7 训练流程（分阶段）

1. **Stage A 记忆抽取/写入**：仿真/文本 privileged 标注训练抽取器与写入控制器。
2. **Stage B 信任头 + 校准**：在独立 validation 上拟合 τ 的特征权重，temperature + conformal 校准；冻结校准集不入 test。
3. **Stage C 具身接入**：FindingDory 训练集（82k 实例）上对 Qwen2.5-VL-3B 做 LoRA，把"低 τ→重观察"策略蒸馏进高层策略。
4. **Stage D 冻结评测**：阈值/校准全部冻结，test 实例不参与任何调参。

---

## 4. 理论

### 4.1 Proposition（版本感知校准闸门：不劣，且在 (A1)+(A2) 下严格更优）

**设定.** 事实版本化如 §2.2；查询正确 = 命中 `t_q` 有效的版本。令 `s` 为相似度统计量，`z=(s, time/version 特征)` 为版本感知统计量，`p^*(·)=P(correct | ·)`。三元闸门阈值 `τ_lo<τ_hi`，重观察信息量 `ρ>0`、代价 `c_obs`。

**明确假设 / 引理.**

- **(A1) 校准单调性.** 信任头 `τ` 是 `p^*(z)` 的单调变换，校准误差 `|E[correct|τ=c]−c| ≤ η`（Lemma 1：由此 τ 的 risk–coverage 曲线落在"按 `p^*(z)` 真排序"曲线的 `O(η)` 邻域内）。
- **(A2) 时间特征非冗余.** 给定相似度 `s`，时间/版本特征对正确性仍有信息：`∃` 可测集上 `p^*(z) ≠ E[p^*(z)\mid s]`（等价地 `I(correct; time | s) > 0`，在 `p_stale>0` 时成立）。
- **(Lemma 2，充分性.)** 由 (A2)，`s` 不是 `correct` 的充分统计量；"仅按 `s` 阈值"的任何规则其可达 risk–coverage 前沿被"按 `p^*(z)` 排序"的前沿支配（Blackwell / 数据处理：粗化统计量不增信息），当 (A2) 成立时为**严格**支配。

**命题.** 对任意目标选择性错误 `ε`（在被 answer 的查询上）：

1. **（不劣，且 (A1)+(A2) 下严格更优）** 版本感知校准闸门的覆盖率 `cov_cal ≥ cov_simthr − O(η)`；在 (A1)(A2) 下该式去掉 `O(η)` 后为**严格**不等（`cov_cal > cov_simthr`）。**不声称无条件 "domination"**——仅主张"不劣，且在 (A1)+(A2) 下严格更优"。
2. **（重观察改善前沿）** 对中间带执行重观察，把其中 `≥ρ` 比例的 would-be-error 转为 correct，使 risk–coverage 前沿在固定覆盖下下移 `Ω(ρ · mass_mid · 𝟙[p_stale>0])`，净收益为正当 `收益 > λ·c_obs`。

**证明思路.** (1) 由 Lemma 1（τ≈`p^*(z)` 的单调变换）+ Lemma 2（`p^*(z)` 排序支配任何 `s`-only 阈值，(A2) 下严格），组合得结论；`O(η)` 项来自校准误差。(2) 重观察把中间带版本解析正确率抬到 `≥ρ`，等于把概率质量从 error 区搬到 correct 区，固定覆盖下错误下降 `Ω(ρ·mass_mid)`；仅当收益超过 `λ·c_obs` 才触发。∎

**定位（诚实）**：这是**算法设计依据**，依赖 (A1)(A2) 与"重观察有信息量"；不是对任意记忆分布的全局保证。(A2) 失效（`p_stale→0`，无动态事实）时命题退化为等号——版本化不带来增益，正是 K2 要检验的；(A1) 失效（τ 未校准）由 K1 拦截。

---

## 5. 实验设计

### 5.0 前置 Pilot（预注册 go/no-go，承载 K4 的前提）

**Pilot-A（固定 reader 是否服从注入提示）.** 榜上杠杆 (a) 成立的**全部前提**是：冻结的 `qwen3.5-9b` reader 在 `ctx` 中看到我们注入的"证据冲突 / 可能过期 / 无足够证据"提示后，**确实会在该弃答的题上弃答、在该答的题上正常作答**，而不是无视提示。做法：构造一小批（~60 题，含 answerable 与 premise-false 两类）对照——注入提示 vs 不注入——测弃答率与正确率的受控变化。

- **Go**：注入提示显著提升 premise/gotchas 题的正确弃答率，且不显著拉低 answerable 题正确率 → 保留榜上杠杆 (a) 与 LAFS headline。
- **No-go**：reader 基本无视注入提示 → **放弃榜上 abstention 机制**，LongMemEval-v2 退为纯检索-质量 sanity check（仅看 `overall_full_set`/latency），全文重心完全落到具身侧（本就如此设计，不伤根本）。

**Pilot-B（τ 早停的延迟-准确率可行性）.** 小规模验证"按 τ 早停"能在 `memory_query_avg_seconds` 下降的同时不显著掉准确率，确认杠杆 (b) 的 LAFS 收益方向。

### 5.1 Benchmarks 与指标口径（已按榜规修正）

- **主榜**：LongMemEval-v2（`small`/`medium` 两档，reader 锁定 `qwen3.5-9b`、judge `gpt-5.2`，提交只竞争 Insert+Query 层）。
  - **榜上 headline = LAFS Gain**（`accuracy=overall_full_set×100`，`latency=memory_query_avg_seconds`，相对固定参考前沿的增益）。
  - 分维报 `static` / `dynamic` / `procedure` / `gotchas` / `overall_full_set`。**dynamic 与 gotchas 是我们机制的正角**（版本化→dynamic；信任注入诱导弃答→gotchas/premise）。
  - risk–coverage、三元决策、abstention 质量为**榜外分析**（榜上弃答对"可答题"计错）。
- **迁移（科学贡献主场）**：FindingDory。指标用官方 **HL-SR / HL-SPL**（高层帧选择）、**LL-SR / LL-SPL**（底层导航）、**SC-SR**（语义覆盖）、**DTG-SR**。*注：FindingDory 并无名为 "LLM-Match" 的指标（那是 OpenEQA 口径）；本文据官方指标口径。* 在此处 risk–coverage 与"低 τ→真实重观察"才是正角。
- **副表**：LongMemEval-v1、LoCoMo（标准对比；LoCoMo 近饱和且方法学有争议，不作 headline）。

### 5.2 Baselines（匹配 reader / token 预算 / 检索调用）

| 组 | 方法 | 作用 |
|---|---|---|
| 下限 | No-retrieval（reader only，LongMemEval-v2 报 1.3%） | 地板 |
| RAG | query→slice；query→slice+notes | 朴素检索 |
| SOTA | AgentRunbook-R（RAG）、AgentRunbook-C（coding agent，v2 SOTA ~74.9%/108.3s small） | 关键对比 |
| 结构记忆 | Mem0、Zep（temporal KG）、A-MEM、LangMem | 主流记忆框架 |
| **校准弃答（直接 prior-art）** | **MMA、SABER** | **验证我们三点残差的必要性** |
| 本文 | NB-Recall full | — |

公平性：所有方法固定同一 reader/judge、同 `B_tok`、同检索调用预算；另报 iso-latency 行。MMA/SABER 移植到 LongMemEval-v2 接口下运行。

### 5.3 Ablations（主文）

1. 写入策略 on/off（vs 全写）；
2. **校准头 on/off**（退化为相似度固定阈值）——验证 headline 机制必要性；
3. **版本感知解析 on/off**（version-blind）——直测 `dynamic_accuracy`（对应 Prop 第 1 条 `p_stale→0` 退化）；
4. **τ_fresh 与 τ_evidence 合并 vs 分头**——验证"过期/不足分离校准"；**并直接对标 MMA**：在 MMA-Bench 的冲突/可靠性设置与我们的动态事实集上，比较"分离 stale/evidence 两分校准"与 **MMA 单一 reliability 分数** `C(Mᵢ)` 的 risk–coverage 与 `dynamic_accuracy`。这是 residual #2 的**决定性实验**（见 K5）；
5. **三元决策 vs answer-only；重观察支路 on/off**（具身侧）——验证 Prop 第 2 条；
6. **信任注入 reader 上下文 on/off**（榜上 gotchas/abstention 杠杆）；
7. 结构：图/版本化 vs 扁平；
8. 记忆预算 0.25×/0.5×/1×/2× sweep；
9. 信任头特征消融（去掉时间特征 → 看 dynamic 掉多少）；
10. conformal vs temperature vs isotonic 校准对比。

### 5.4 Metrics

- **榜上**：LAFS Gain；`overall_full_set` 及四分维；`memory_query_avg_seconds`。
- **榜外/具身**：risk–coverage 曲线与 AURC；选择性准确率@覆盖；abstention precision/recall/F1；`dynamic_accuracy` vs fact-change 率斜率；FindingDory HL-SR/HL-SPL/LL-SR/LL-SPL/SC-SR/DTG-SR；重观察预算（导航步数）vs 成功率 Pareto；τ 的 rank-corr 与 conformal 覆盖。
- 所有比例报分子/分母 + 95% CI；统计单位为 question / episode 而非 frame。

### 5.5 Kill criteria（预注册）

- **K1（τ 排序）**：若 `τ` 与实际正确性 rank-corr 的 95% 下界 ≤0，校准 headline 不成立——先修信任头，不往下跑。
- **K2（版本化）**：若 version-aware 相对 version-blind 在 `dynamic_accuracy` 的 paired 改善 95% 下界未过预注册 margin，删除版本化 headline，降级为"结构化记忆 + 校准弃答"。
- **K3（具身重观察）**：FindingDory 上"低 τ→重观察"相对 always-answer 在**匹配导航预算**下 HL-SR 无改善 → 具身重观察主张失败，退为 text-only + risk-coverage。
- **K4（榜单，依赖 Pilot-A）**：仅当 §5.0 Pilot-A 判 Go 才追求 LAFS headline；若 LongMemEval-v2 上 NB-Recall 相对 AgentRunbook-R 在**匹配 latency** 下 LAFS Gain ≤0（或 Pilot-A 判 No-go）→ 榜单主张失败，LongMemEval-v2 退为纯检索质量 sanity check，全文退守具身科学贡献（K3 通过即可独立成文）。
- **K5（residual #2 决定性）**：若"分离 stale/evidence 两分校准"相对 **MMA 单一 reliability 分数**在 `dynamic_accuracy`/risk–coverage 的 paired 改善 95% 下界未过预注册 margin → **把 residual #2 并入 residual #1**（prior-art 表 §8.1 相应合并），不再单独声称"分离校准"为贡献。

### 5.6 统计与预注册

- LongMemEval-v2：对同题做 paired bootstrap；主比较为 NB-Recall vs {AgentRunbook-R, MMA, SABER} 的 LAFS Gain 与 `dynamic`/`gotchas` 分维，Holm 校正；其余为 secondary。
- FindingDory：按 task category 分层报告；multi-goal 单列（已知近 0，作难度上界诊断）。
- checkpoint 仅按 validation 选；test 不调参；失败案例全量发布。

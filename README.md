# 质谱标准谱库候选检索 API

质谱数据工程师从持续补录的标准谱库中为未知样本查找候选物（供人工复核）。
FastAPI + NumPy + SQLite（WAL），索引持久化，支持在线修改坏通道屏蔽方案并无缝重建索引。

## 功能

- **录入**：`POST /spectra` 录入谱号 + 256 通道非负强度；同号替换；库容上限 5000 谱；
  录入与索引写入同一事务，**录入成功即可检索**。屏蔽后全零的库谱照常保存但跳过索引（`indexed: false`）。
- **检索**：`POST /query` 未知谱按当前生效方案（屏蔽坏通道 → L2 归一化）处理后，
  以余弦相似度返回前三名，同分按谱号升序。未知谱屏蔽后全零则拒绝（422）。
- **方案管理**：`POST /scheme/rebuild` 修改屏蔽通道并异步重建索引；
  `GET /schemes` 轮询重建状态；`GET /scheme/active` 查看当前方案。
- **演示入口**：`/docs`（Swagger UI），`/` 重定向到 `/docs`。

## 一致性设计（关键不变量及其实现）

所有共享状态（原始谱、方案、生效方案指针、各方案索引）都在同一个 SQLite 文件（WAL 模式）中，
因此**两个 API 进程共库**与**重启后**行为一致：

| 要求 | 实现 |
| --- | --- |
| 录入成功即可检索 | upsert 与索引向量写入同一 `BEGIN IMMEDIATE` 事务，生效方案在事务内读取 |
| 查询的方案/谱内容/得分同属一个已提交索引 | 单个读事务内读取生效方案 + 该方案全部向量（WAL 快照隔离） |
| 重建期间录入、检索可完成，查询沿用旧方案 | 候选索引在新 scheme_id 下分批小事务构建；生效指针不动，读不阻塞写 |
| 成功切换包含期间新录入的谱 | 切换事务内先做最后一次增量补齐（按单调递增 `seq`），再原子切换 |
| 重建失败保留旧索引 | 生效指针只由切换事务更新；失败仅把候选方案标记为 `failed` |
| 迟到重建不得覆盖后来生效的方案 | 激活是对请求时基准方案的 compare-and-swap，失败则标记 `abandoned` |

谱库每次 upsert 递增逻辑时钟 `seq`；重建按 `seq` 增量构建，切换事务与录入事务通过
SQLite 写锁互斥，因此切换后的索引必然反映切换前所有已提交的谱。

## 快速开始

### Docker Compose（双进程共库演示）

```bash
docker compose up --build
# api1: http://localhost:8000/docs   api2: http://localhost:8001/docs
```

两个服务挂载同一 named volume（`/data/spectra.db`）：在 8000 录入、在 8001 检索即可验证共库；
`docker compose down && docker compose up` 后数据与索引仍在（重启持久性）。

### 本地运行

```bash
pip install -r requirements.txt
MS_DB_PATH=./spectra.db uvicorn app.main:app --port 8000
```

### 演示脚本

```bash
python3 examples/demo.py http://localhost:8000
```

录入 `examples/spectra/` 下 6 条示例谱 → 查询 `examples/unknown.json`（应命中
`STD-0001-Caffeine`）→ 请求屏蔽通道 `[10,11,12]` 重建 → 轮询切换 → 再次查询。
`STD-0006-Artifact-Only` 信号全部位于 10-12 通道，切换后被跳过（`indexed_count` 从 6 变 5）。

### API 摘要

| 方法/路径 | 说明 |
| --- | --- |
| `POST /spectra` | 录入/替换谱。409=库满；422=强度非法（非 256 通道/负值/非有限值） |
| `POST /query` | 返回 `{scheme_id, masked_channels, results:[{spectrum_id, score}]}`（前 3 名） |
| `POST /scheme/rebuild` | 202 启动异步重建；掩码与现方案相同则返回 `already_active` |
| `GET /schemes` | 所有方案及状态：`building/active/superseded/abandoned/failed` |
| `GET /scheme/active` | 当前生效方案、索引谱数、库谱数 |
| `GET /healthz` | 健康检查 |

## 配置（环境变量）

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `MS_DB_PATH` | `./spectra.db`（容器内 `/data/spectra.db`） | SQLite 文件路径 |
| `MS_MAX_SPECTRA` | `5000` | 库容上限 |
| `MS_REBUILD_BATCH_DELAY_MS` | `0` | 重建每批人为延迟（演示/测试并发用） |

## 测试

```bash
pip install -r requirements.txt httpx pytest
python3 -m pytest tests/ -q
```

覆盖：top-3 与同分排序、同号替换、库容上限、屏蔽后零向量（库谱跳过/查询拒绝）、
重建切换、重建期间录入被新索引包含、迟到重建让位、重建失败保留旧索引、
重启持久化、双实例共库、重建期间查询沿用旧方案。

## 已知限制

- 重建由收到请求的进程内的后台线程执行；若该进程在重建中崩溃，候选方案会停留在
  `building`（不影响生效索引与后续重建，重新发起即可）。
- 谱库无删除接口（需求未要求）；同号替换可修正错误谱。
- 单文件 SQLite 适合单机多进程；跨机部署需换外部数据库（当前设计不变量同样适用）。

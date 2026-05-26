# 报告七：CoachMind AI 系统架构设计与实施方案

---

```
文档元信息
──────────────────────────────────────────────────────
项目名称：CoachMind AI — 足球教练智能辅助系统
文档编号：07
文档类型：系统架构设计报告（理论研究）
版本：v1.0
撰写日期：2026-03-16
适用阶段：Demo MVP → 生产环境演进路径
关键词：微服务架构、FastAPI、Celery、TimescaleDB、Qdrant、WebSocket、MinIO
──────────────────────────────────────────────────────
```

---

## 2026 修订摘要：视频主链路 + 可穿戴增强层

本报告原始版本以“多机位视频 + GPS 传感器 + 边缘推理”为完整形态描述系统架构。结合 CoachMind 最新产品定位，系统架构应调整为：

1. **视频仍是默认主链路**：Campus Lite 和大多数校园足球场景优先采用手机/单机位广角/双机位视频，先解决自动剪辑、比赛状态重建、中文复盘、球员成长档案和训练 drill 映射。
2. **可穿戴是增强层，不是默认前置条件**：GPS、心率、IMU、脚部传感器适合 Campus Pro 小规模 pilot 和 Elite Training 场景，用于补齐视频难以稳定量化的训练负荷、心率区间、恢复状态、伤后回归和脚部技术动作。
3. **不建议第一阶段自研硬件**：硬件会引入供应链、续航、防水、认证、售后和 firmware 维护复杂度。前三个月更合理的路线是建立供应商无关的 wearable adapter 和内部统一 schema。
4. **数据壁垒来自融合数据**：真正有壁垒的不是某个 GPS 背心本身，而是视频片段、轨迹、负荷、心率、训练建议、教练反馈和 4/8/12 周效果回流之间的结构化关联。

因此，下文涉及“4路4K摄像机”“全量边缘推理”“GPS传感器网关”等内容，应理解为 Elite Training 或未来全栈部署形态；三个月 MVP 应以云优先、低门槛视频采集、可选 wearable pilot 为准。

## 一、架构决策背景：我们在解决什么规模的问题

### 1.1 数据规模量化

CoachMind AI 系统需要同时处理来自多个维度的高密度数据流，这是做出架构决策的根本出发点。

**视频数据维度：**
一场标准90分钟比赛，部署4路4K摄像机（主摄、辅摄、战术鸟瞰、角落机位）的原始数据体量如下：
- 单路4K@30fps H.264原始码率约 25-40 Mbps
- 4路同时接入：~100-160 Mbps 持续入站带宽
- 90分钟赛事存储需求：约 **67-108 GB** 原始视频
- 推理帧采样（每秒5帧，4路）：每秒20帧需经过目标检测+姿态估计，单帧推理时延要求 < 50ms

**传感器数据维度：**
- GPS定位数据：每名球员每秒1条记录，22名球员 = **22条/秒**
- 若含青训梯队同时分析（3支队伍）：66条/秒
- 90分钟比赛：约 **118,800条** GPS记录（仅一场单队）
- IMU数据（加速度计、陀螺仪）：100Hz采样，远高于GPS

**实时事件流维度：**
- 裁判信号、换人、进球、犯规等手动事件：低频但高优先级
- AI自动检测事件（越位判断、传球成功率、跑位分析）：每分钟约 200-500 条推断事件
- 这些事件需要在 **< 2秒** 内推送到教练端看板

### 1.2 并发场景分析

系统必须支持多场比赛并行分析的核心业务场景：
- 周末联赛日：同城联赛可能有 8-16 支球队同时比赛
- 青训联赛：U8到U18多年龄段同场地进行
- 赛后分析高峰：比赛结束后30分钟内，所有教练同时查看报告

以中等规模俱乐部为目标客户（一个城市10-20个注册俱乐部），系统需设计为支持 **20路并发视频流 + 500并发WebSocket连接** 的基准能力，峰值扩容至2倍。

### 1.3 微服务架构选型理由

面对上述数据规模，单体架构存在三个根本性障碍：

**独立扩展需求：** AI推理是GPU密集型操作，视频存储是I/O密集型，LLM调用是网络等待密集型。将这三者打包在单体中，意味着扩展推理能力时必须同比扩展存储和网络服务，资源浪费极为严重。微服务允许单独对 InferenceService 进行GPU节点水平扩展，而其他服务维持在普通CPU节点。

**故障隔离需求：** Claude API 出现限速或延迟时，不应影响正在进行的视频推理任务。LLM服务降级必须对视频追踪服务透明。微服务的进程隔离天然实现了故障边界。

**技术栈灵活性：** 推理服务需要 Python + CUDA + PyTorch 环境；报告生成服务可能最终用 Node.js（PDF渲染生态更成熟）；存储服务对接 MinIO S3 协议。单体架构强制所有模块使用同一语言和依赖树，而这些技术要求本质上是矛盾的。

---

## 二、整体系统架构

### 2.1 服务拓扑全景图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           外部接入层                                      │
│   浏览器/App客户端          摄像机SDK          GPS传感器网关                │
└──────────┬──────────────────────┬─────────────────┬───────────────────────┘
           │ HTTPS/WSS            │ RTSP/HLS         │ MQTT/HTTP
           ▼                      ▼                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        API Gateway (Nginx + Kong)                        │
│         JWT验证 / Rate Limiting / SSL终止 / 负载均衡                      │
└────┬─────────┬──────────┬────────┬──────────┬────────┬───────────────────┘
     │         │          │        │          │        │
     ▼         ▼          ▼        ▼          ▼        ▼
┌────────┐ ┌──────────┐ ┌──────┐ ┌────────┐ ┌──────┐ ┌────────────┐
│ User   │ │  Video   │ │ Ana- │ │  LLM   │ │Noti- │ │  Storage   │
│Service │ │Ingestion │ │lytics│ │Service │ │fication│ │ Service   │
│:8001   │ │Service   │ │Svc   │ │ :8004  │ │ Svc  │ │  :8006     │
│        │ │ :8002    │ │:8003 │ │        │ │:8005 │ │            │
└────┬───┘ └────┬─────┘ └──┬───┘ └────┬───┘ └──┬───┘ └─────┬──────┘
     │          │           │          │         │            │
     │          ▼           │          │         │            │
     │    ┌─────────────┐   │          │         │            │
     │    │  Inference  │   │          │         │            │
     │    │  Service    │   │          │         │            │
     │    │  :8007      │   │          │         │            │
     │    │  (GPU Node) │   │          │         │            │
     │    └─────┬───────┘   │          │         │            │
     │          │            │          │         │            │
     └──────────┴────────────┴──────────┴─────────┴────────────┘
                                    │
                    ┌───────────────┴──────────────────┐
                    ▼                                  ▼
            ┌──────────────┐                 ┌──────────────────┐
            │ Message Bus  │                 │   Data Layer     │
            │ Redis Streams│                 │                  │
            │ + Celery     │                 │ PostgreSQL :5432 │
            │              │                 │ TimescaleDB ext  │
            └──────────────┘                 │ Qdrant :6333     │
                                             │ Redis :6379      │
                                             │ MinIO :9000      │
                                             └──────────────────┘
```

### 2.2 各服务职责边界

**UserService（:8001）**
认证与授权核心。处理球队注册、教练账号、JWT签发与刷新。维护球队与球员的归属关系，是整个数据隔离的信任锚点。数据库：PostgreSQL（teams, coaches, players表）。

**VideoIngestionService（:8002）**
视频流接入与预处理。接收RTSP推流或上传文件，完成转码（统一到720p/1080p推理分辨率）、关键帧提取、分片存储到MinIO。触发Celery任务通知InferenceService开始推理。是唯一直接与视频原始数据打交道的服务。

**InferenceService（:8007，GPU节点）**
系统算力核心，物理上部署在配备NVIDIA GPU的节点。执行：目标检测（球员/球/裁判识别，YOLOv8/v10）、多目标追踪（ByteTrack）、球员姿态估计（RTMPose）、球场关键点检测（单应性矩阵计算）。输出标准化追踪数据流（JSON），推送至Redis Streams，供AnalyticsService消费。

**AnalyticsService（:8003）**
战术指标计算引擎。消费InferenceService输出的追踪流，计算：传跑位分析、控球率、跑动热图、阵型还原、压迫强度指数、高位逼抢次数。是纯CPU计算服务，可水平扩展。结果写入PostgreSQL和Qdrant（战术片段向量化存储）。

**LLMService（:8004）**
自然语言生成与RAG检索核心。接收AnalyticsService的战术摘要，从Qdrant检索相似历史战术案例，拼装Context后调用Claude API，生成教练可读的中文战术建议文本。维护Prompt模板版本管理，支持教练自定义指令风格。

**NotificationService（:8005）**
实时推送枢纽。管理WebSocket长连接，维护"比赛Room"概念（每场比赛一个独立频道），将AnalyticsService和LLMService的输出实时广播给订阅该场比赛的教练端。

**StorageService（:8006）**
存储抽象层。统一封装MinIO操作（上传/下载/预签名URL），管理视频生命周期（热/温/冷分层），处理报告PDF的生成触发与缓存查询。

---

## 三、FastAPI 选型深度论证

### 3.1 与 Flask 的对比

Flask是Python Web生态的经典之选，但其异步支持是后续通过第三方扩展（flask-async）"打补丁"加入的，并非原生设计。这一点在处理高并发I/O时会暴露根本缺陷。

| 维度 | Flask | FastAPI |
|---|---|---|
| 异步支持 | 非原生，需flask[async]扩展，WSGI本质串行 | 原生ASGI，asyncio贯穿全栈 |
| 每秒请求数（TechEmpower基准） | ~2,000-4,000 RPS | ~8,000-15,000 RPS（约3-5倍差距）|
| WebSocket支持 | flask-socketio（额外依赖） | 原生支持，与路由同一框架 |
| 数据验证 | 手动或WTForms（表单为中心设计） | Pydantic V2，自动验证+序列化 |
| 类型提示集成 | 弱，仅注释层面 | 深度集成，运行时强制 |
| 文档生成 | 需手动维护或用flasgger | 自动OpenAPI 3.0，零维护成本 |

对于CoachMind AI，NotificationService需要同时维持数百个WebSocket长连接，Flask的同步WSGI模型意味着每个连接会占用一个线程，内存开销线性增长且有上限（通常几百个线程后系统不稳定）。FastAPI的asyncio事件循环可以在单线程内复用数千个连接，是处理WebSocket密集场景的正确工具。

### 3.2 与 Django 的对比

Django是"全包式"框架，内置ORM、Admin后台、认证系统等，适合快速构建CRUD应用。但对于CoachMind AI的微服务场景，Django的设计哲学形成了显著的阻力：

**ORM不适合GPU服务的理由：** InferenceService的核心操作是：接收帧数据 → GPU推理 → 输出JSON → 推送到消息队列。整个流程几乎不涉及数据库写入（追踪结果量太大，直接写PostgreSQL是反模式）。强行引入Django ORM只会增加启动时间和内存占用，毫无收益。

**Django Admin的诱惑与陷阱：** 团队可能因为"免费的Admin界面"选择Django，但CoachMind AI的管理界面需要展示实时视频帧、热图可视化、战术板，这远超Django Admin的能力边界，最终仍需要独立前端，前期的"便利"变成了后期的技术债。

**启动时间问题：** 在Kubernetes环境中，Pod启动时间直接影响扩缩容速度。一个空白Django应用的冷启动时间约500-800ms，而一个精简FastAPI服务可以在100-200ms内完成启动。在高峰期需要快速横向扩展10个InferenceService Pod时，这个差距会显著影响响应速度。

### 3.3 Pydantic V2 的架构价值

Pydantic V2相比V1进行了底层重写（Rust实现核心验证逻辑），性能提升约5-50倍。在CoachMind AI中，Pydantic不仅是数据验证工具，更是服务间契约（Contract）的编码实现：

```python
# 追踪数据的服务间契约定义（概念示意）
class PlayerTrackingFrame(BaseModel):
    frame_id: int
    timestamp_ms: int
    match_id: UUID
    players: List[PlayerDetection]
    ball: Optional[BallDetection]

class PlayerDetection(BaseModel):
    track_id: int
    team: Literal["home", "away", "referee"]
    bbox: BoundingBox
    pitch_coords: Optional[PitchCoordinates]  # 球场坐标系（0-105m x 0-68m）
    confidence: float = Field(ge=0.0, le=1.0)
```

当InferenceService的输出格式发生变化时，Pydantic在编译时（配合mypy）和运行时双重验证，确保下游AnalyticsService收到的数据符合预期，避免了微服务间数据格式漂移这一常见故障根源。

### 3.4 自动 OpenAPI 文档对团队协作的价值

CoachMind AI的前端开发团队（React）与后端各微服务并行开发。FastAPI自动生成的OpenAPI 3.0文档（通过 `/docs` 路径实时访问）意味着：前端工程师无需等待后端完成实现，即可基于接口定义使用 `openapi-generator` 生成类型安全的TypeScript客户端代码，实现真正的并行开发。在6个微服务、3名前端工程师的协作规模下，这一特性可节省约20-30%的联调时间。

---

## 四、Celery 异步任务架构设计

### 4.1 视频分析必须异步的根本原因

视频分析任务的执行时间具有内在的不可预测性：
- 网络拥塞导致视频上传速度波动（5分钟视频可能需要30秒到5分钟上传）
- GPU推理速度受当前批量大小和显存占用影响
- 一场比赛的完整战术分析（含LLM报告生成）端到端耗时约 3-8 分钟

如果使用同步HTTP请求处理视频分析，教练端上传视频后的HTTP连接必须保持打开状态等待结果，任何网络抖动都会导致请求超时失败，且服务器的线程/进程资源被长时间占用，系统并发能力极低。

异步设计的正确模式：教练上传视频 → 立即返回 `task_id`（< 100ms）→ 后台Celery Worker异步处理 → 通过WebSocket推送进度和结果。

### 4.2 优先级队列设计

```
Redis Broker
├── queue: realtime          # 优先级 10 — 实时比赛事件（< 5秒延迟SLA）
│   ├── task: detect_event   # 进球/越位AI检测
│   └── task: push_tactical  # 战术警报推送
│
├── queue: postgame          # 优先级 5 — 赛后分析（< 5分钟SLA）
│   ├── task: analyze_match  # 完整比赛AI分析
│   ├── task: generate_heatmap
│   └── task: compute_stats
│
└── queue: report            # 优先级 1 — 报告生成（< 30分钟SLA）
    ├── task: generate_pdf_report
    ├── task: weekly_summary
    └── task: player_report
```

Celery Worker的启动配置将不同Worker进程绑定到不同队列，`realtime` 队列分配专用Worker，确保低优先级的报告生成任务永远无法抢占实时分析资源。

### 4.3 Beat Scheduler 定时任务

```
Celery Beat 调度清单
─────────────────────────────────────────────────────
每日 07:00   generate_daily_training_summary     每支球队生成昨日训练摘要
每周一 06:00  generate_weekly_player_reports      更新全队球员周报（RAG检索历史表现）
每日 02:00   cleanup_cold_video_storage          将7天前视频从热存储迁移至冷归档
每小时       health_check_inference_service      验证GPU Worker存活状态
比赛后+30min  trigger_postgame_analysis           自动触发赛后完整分析（通过比赛日历事件）
─────────────────────────────────────────────────────
```

### 4.4 失败重试策略

```python
# 重试策略设计原则（概念示意）

# 对于AI推理任务：网络抖动可恢复，GPU OOM不可恢复
@app.task(
    bind=True,
    max_retries=3,
    default_retry_delay=60,     # 初始重试延迟60秒
    autoretry_for=(NetworkError, TimeoutError),
    retry_backoff=True,          # 指数退避: 60s → 120s → 240s
    retry_backoff_max=600,       # 最大退避上限10分钟
    retry_jitter=True            # 加入随机抖动，避免惊群效应
)
def analyze_video_segment(self, segment_id: str): ...

# 对于LLM调用任务：Claude API限速时需要退避
@app.task(
    max_retries=5,
    autoretry_for=(RateLimitError,),
    retry_backoff=True,
    retry_backoff_max=300
)
def generate_tactical_report(match_id: str): ...
```

死信队列（Dead Letter Queue）：超过最大重试次数的任务进入 `dlq:failed_tasks`，触发告警通知，并保留完整的任务参数供人工干预或调试分析。

---

## 五、数据库架构设计

### 5.1 PostgreSQL 核心表设计

```sql
-- 球队和人员核心表（概念设计）

teams (id, name, city, tier, created_at, subscription_tier)
coaches (id, team_id, name, role, jwt_sub, created_at)
players (id, team_id, jersey_number, name, position, birth_date, is_active)

matches (
  id UUID PK,
  team_id UUID FK,
  opponent_name VARCHAR,
  match_date TIMESTAMPTZ,
  venue VARCHAR,
  competition VARCHAR,
  status ENUM('scheduled','live','completed','analyzed'),
  video_ingestion_id UUID,        -- 关联VideoIngestionService
  analysis_task_id VARCHAR        -- Celery task ID
)

-- 比赛事件表（稀疏高优先级事件）
match_events (
  id, match_id, event_type ENUM('goal','foul','substitution','offside',...),
  minute INT, second INT, player_id UUID, description TEXT,
  ai_confidence FLOAT,            -- AI自动检测时的置信度，NULL表示人工标注
  created_at TIMESTAMPTZ
)

-- 球员赛次统计聚合表（AnalyticsService写入）
player_match_stats (
  id, match_id, player_id,
  distance_km FLOAT, sprint_count INT, max_speed_kmh FLOAT,
  pass_accuracy FLOAT, key_passes INT, duels_won INT,
  heatmap_data JSONB,             -- 压缩后的位置热图数据
  ai_performance_score FLOAT      -- 综合AI评分 0-100
)
```

### 5.2 时序数据的分区处理策略

球员GPS位置数据每秒30条（考虑GPS+视频追踪融合），一场比赛产生约 **39,600条** 原始位置记录（22名球员 × 30Hz × 60秒 × 90分钟）。

**方案一（推荐生产环境）：TimescaleDB 扩展**
TimescaleDB 是 PostgreSQL 的时序扩展，通过自动分区（chunk）将时序表按时间窗口切分（每块默认7天数据），查询时自动路由到相关分区，避免全表扫描。对于"查询某场比赛某球员后45分钟的位置轨迹"这类时间范围查询，性能可比纯PostgreSQL提升 **10-100倍**。

```sql
-- TimescaleDB超表定义（概念示意）
SELECT create_hypertable('player_positions', 'recorded_at',
  chunk_time_interval => INTERVAL '1 hour');  -- 每小时一个chunk，适合比赛场景

-- 自动压缩策略：3天前数据自动压缩，存储节约60-90%
SELECT add_compression_policy('player_positions',
  compress_after => INTERVAL '3 days');
```

**方案二（Demo/小规模）：PostgreSQL 原生范围分区**
按 `match_id` 和 `recorded_at` 建立复合分区，每场比赛一个分区。查询时显式指定 `WHERE match_id = $1` 触发分区裁剪。实现简单，无需安装扩展，适合Demo阶段。

### 5.3 Qdrant 向量数据库 Collections 设计

Qdrant是CoachMind AI实现RAG（检索增强生成）的核心基础设施，存储战术知识的语义向量表示。

```
Collections 设计：
─────────────────────────────────────────────────────
tactical_patterns          向量维度: 1536 (text-embedding-3-small)
  payload: {
    pattern_type: "pressing|counter|set_piece|...",
    description_zh: "高位逼抢战术描述",
    success_rate: 0.72,
    applicable_situations: ["trailing","late_game"],
    source: "match_id|manual_input",
    team_id: UUID    ← 关键：向量检索时按team_id过滤，实现球队隔离
  }

player_performance_profiles  向量维度: 768 (domain-specific embedding)
  payload: {
    player_id: UUID,
    season: "2025-26",
    role: "central_midfielder",
    strengths_embedding: [...],   ← 综合多场表现的风格向量
    last_updated: timestamp
  }

coaching_knowledge_base      向量维度: 1536
  payload: {
    knowledge_type: "drill|formation|recovery|...",
    content_zh: "训练方法描述",
    applicable_age_group: ["U15","U17","senior"],
    difficulty: "beginner|intermediate|advanced"
    # 此集合不含team_id，为公共知识库
  }
─────────────────────────────────────────────────────
```

---

## 六、实时推送架构

### 6.1 通信协议选型对比

| 技术 | 连接方式 | 双向通信 | 断线重连 | HTTP兼容 | 适用场景 |
|---|---|---|---|---|---|
| Long Polling | 短连接轮询 | 模拟单向 | 自然兼容 | 完全兼容 | 低频更新，兼容性优先 |
| SSE（Server-Sent Events） | 单向持久连接 | 仅服务→客户端 | 浏览器自动重连 | HTTP/1.1 | 单向推送，实现简单 |
| WebSocket | 全双工持久连接 | 完全双向 | 需手动实现 | 需Upgrade握手 | 高频双向交互 |

CoachMind AI 选择 **WebSocket** 的核心理由：教练端不只是被动接收数据，还需要发送实时标注（"标记此刻为关键战术时刻"）、触发即时分析请求（"分析刚才那次传球"）。SSE无法支持客户端发送消息。Long Polling在90分钟比赛期间会产生大量无意义的轮询请求，且延迟（通常1-3秒）超过战术警报的实时性要求。

### 6.2 Room 概念与连接管理

```
WebSocket 连接层次设计：

Namespace: /match
  ├── Room: match_{match_id_1}
  │     ├── conn: coach_A (主教练iPad)
  │     ├── conn: coach_B (助教笔记本)
  │     └── conn: analyst_C (数据分析师)
  │
  └── Room: match_{match_id_2}
        └── conn: coach_D

广播规则：
  - AnalyticsService → 发布到 Redis Channel: match_updates:{match_id}
  - NotificationService 订阅 Redis → 广播到对应 WebSocket Room
  - 教练客户端消息 → NotificationService → 路由到相应处理服务
```

### 6.3 消息格式标准

```json
{
  "msg_type": "tactical_update",
  "match_id": "uuid",
  "timestamp_ms": 1710000000000,
  "sequence": 1842,
  "payload": {
    "event": "high_press_detected",
    "confidence": 0.87,
    "minute": 67,
    "involved_players": [7, 11, 9],
    "ai_suggestion": "对方后腰接球压力不足，建议立即前压",
    "formation_snapshot": "4-3-3"
  }
}
```

---

## 七、存储策略设计

### 7.1 对象存储选型

| 方案 | 优势 | 劣势 | 适用场景 |
|---|---|---|---|
| MinIO（自建） | 完全数据自主、零流量费用、S3协议兼容 | 运维成本高、需自建HA | 私有化部署俱乐部、数据主权敏感 |
| AWS S3 | 无运维、全球CDN、99.999999999%耐久度 | 按流量计费（视频数据量大时成本高）、数据出境合规风险 | SaaS模式、云端部署 |
| 阿里云OSS | 国内访问速度快、合规、价格适中 | 厂商绑定风险 | 中国市场SaaS部署 |

CoachMind AI 的私有化部署版本（面向专业俱乐部）选择 **MinIO**，原因在于：职业俱乐部对视频数据（包含球员未公开战术训练内容）有极强的数据主权要求，不允许上传至第三方云端。

### 7.2 视频分层存储策略

```
存储层级         时间范围        存储介质           访问延迟      成本指数
─────────────────────────────────────────────────────────────────
热数据（Hot）    当天比赛        SSD / NVMe         < 50ms        1.0x
温数据（Warm）   过去7天         HDD RAID            < 500ms       0.3x
冷归档（Cold）   7天以前         MinIO Erasure/S3    < 5s          0.05x
```

Celery Beat 定时任务每日凌晨执行存储分层迁移，通过 MinIO 的生命周期策略（lifecycle policy）自动完成。视频的预签名URL（Presigned URL）有效期设置与存储层匹配：热数据1小时、温数据24小时。

### 7.3 报告 PDF 生成与缓存

报告生成是计算密集的低频任务（一场比赛最多触发3-5次）。生成策略：

1. 教练请求报告 → 查询Redis缓存键 `report:match:{match_id}:v{version}`
2. 缓存命中 → 返回MinIO预签名下载URL（生成耗时 < 10ms）
3. 缓存未命中 → 触发Celery `report`队列任务 → 异步生成（WeasyPrint渲染HTML模板为PDF）→ 上传MinIO → 写入Redis缓存（TTL=24小时）

---

## 八、安全设计

### 8.1 JWT 认证与球队数据隔离

JWT Payload 中强制包含 `team_id` Claim，所有数据查询必须携带此 Claim 作为过滤条件。AnalyticsService 的每一个查询函数在入口处执行 Team Ownership 验证：

```python
# 数据隔离模式（概念示意）
def get_match_analysis(match_id: UUID, current_user: JWTClaims):
    match = db.query(Match).filter(
        Match.id == match_id,
        Match.team_id == current_user.team_id   # 强制 team 隔离
    ).first()
    if not match:
        raise HTTPException(403, "无权访问此比赛数据")
```

Qdrant 向量检索同样在 Filter 中强制附加 `team_id` 条件，确保球队A的战术RAG检索不会返回球队B的数据。

### 8.2 API Rate Limiting

通过 Kong API Gateway 的 Rate Limiting 插件实现：
- LLM报告生成接口：单球队 10次/小时（防止LLM API费用暴涨）
- 视频上传接口：单用户 5个并发上传
- 战术查询接口：200次/分钟（正常使用绰绰有余）

### 8.3 球员数据脱敏规则

面向外部展示或API导出时，自动执行以下脱敏规则：
- 未成年球员（< 18岁）：姓名替换为"球员#号码"，不暴露出生日期
- GPS原始轨迹数据：仅向俱乐部医疗团队开放，教练默认只看聚合统计
- 伤病历史：严格按需开放，默认不随球员Profile返回

---

## 九、监控与可观测性

### 9.1 Prometheus 指标体系

```
核心指标清单（按告警重要性排序）：
────────────────────────────────────────────────────────────
P0（立即告警）：
  inference_latency_p95 > 200ms      GPU推理性能下降
  celery_queue_length{queue=realtime} > 100  实时队列积压
  websocket_active_connections < 期望值*0.5  大量连接断开

P1（5分钟内响应）：
  llm_api_error_rate > 5%            Claude API异常
  storage_write_failure_total > 10/min  MinIO写入故障
  postgame_queue_length > 500        赛后分析积压

P2（监控看板，不告警）：
  player_detection_fps               实时追踪帧率
  qdrant_search_latency_p99          向量检索延迟
  report_generation_duration         报告生成耗时
────────────────────────────────────────────────────────────
```

### 9.2 结构化日志

使用 `structlog` 库，所有日志以JSON格式输出，强制包含 `trace_id`（跨服务请求追踪）、`team_id`（便于按球队过滤日志）、`service_name`、`log_level` 字段。日志统一采集至 Loki，配合 Grafana 展示。

---

## 十、Demo vs. Production 架构差异

Demo阶段（单台开发机，用于演示和早期用户试用）对架构进行以下简化：

| 组件 | Demo简化方案 | 生产方案 |
|---|---|---|
| 数据库 | SQLite（零配置） | PostgreSQL + TimescaleDB |
| 消息队列 | Celery with SQLite broker（或内存broker） | Redis Streams |
| 对象存储 | 本地文件系统 `/data/videos/` | MinIO 集群 |
| 向量数据库 | Qdrant 单节点（无副本） | Qdrant 集群（3副本）|
| Celery Worker | 单进程，所有队列合并 | 独立进程池，按队列隔离 |
| GPU推理 | 降级为CPU推理（检测速度慢5-10倍，可接受用于演示）| A100/H100 GPU节点 |
| 部署方式 | `docker-compose up` 单命令启动所有服务 | Kubernetes + Helm Charts |
| 监控 | 无（或基础日志输出） | Prometheus + Grafana + Loki |
| 负载均衡 | 无，单实例直接暴露端口 | Nginx + Kong Gateway |

Demo架构的核心原则：**保持所有服务接口和数据契约与生产环境完全一致**，仅替换底层基础设施实现。这确保了从Demo到生产的迁移是配置级别的变更，而非代码级别的重写。通过环境变量（`.env` 文件）控制所有基础设施连接字符串，切换环境只需修改配置，不改动一行业务逻辑代码。

---

*报告结束 — 报告七：系统架构设计与实施方案 v1.0*

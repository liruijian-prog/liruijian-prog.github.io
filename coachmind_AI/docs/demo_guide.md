---
title: Demo 开发手册 — Phase 0 一个月冲刺指南
version: v1.0
date: 2025-03
authors: CoachMind AI 技术团队
category: 实施文档
password_required: true
---

# Demo 开发手册
## Phase 0：一个月冲刺，从零到可演示原型

> 目标：第4周在北体大足球场完成第一次真实用户验证
> 前提：GPU服务器（阿里云A10实例）已就绪，Python 3.11+环境已配置

---

## 一、环境准备（Day 1）

### 1.1 阿里云GPU实例配置

推荐：`ecs.gn7i-c16g1.4xlarge`（NVIDIA A10 × 1，16 vCPU，60GB内存）
- 系统：Ubuntu 22.04 LTS
- 价格：约 ¥24/小时（按量），Demo阶段预算¥2,000

```bash
# 基础环境
sudo apt-get update && sudo apt-get install -y git curl wget ffmpeg

# Python 环境（用 miniconda）
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
conda create -n coachminddemo python=3.11
conda activate coachminddemo

# CUDA 验证
nvidia-smi  # 应显示 CUDA 12.x

# PyTorch（CUDA 12.1版本）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 1.2 依赖安装

```bash
pip install ultralytics>=8.3.0    # YOLO11
pip install supervision>=0.24.0   # ByteTrack封装
pip install fastapi uvicorn[standard] python-multipart
pip install anthropic>=0.40.0     # Claude API
pip install openai-whisper         # Whisper v3
pip install qdrant-client langchain langchain-anthropic
pip install redis celery
pip install opencv-python-headless numpy scipy
pip install pillow reportlab       # PDF报告生成
pip install python-dotenv
```

### 1.3 项目目录结构

```
demo/
├── main.py               # FastAPI主服务
├── video_pipeline.py     # 视频分析Pipeline
├── tracker.py            # ByteTrack封装
├── tactical_map.py       # 战术地图生成
├── llm_coach.py          # Claude API + RAG
├── whisper_stt.py        # 语音转文字
├── report_generator.py   # 赛后报告生成
├── models/               # 下载的YOLO权重
├── knowledge/            # 战术知识库文档
├── uploads/              # 上传的视频临时存储
└── frontend/             # React Demo界面
```

---

## 二、YOLO11 + ByteTrack Pipeline（Day 2-3）

### 2.1 YOLO11 初始化

```python
# video_pipeline.py
from ultralytics import YOLO
import supervision as sv
import numpy as np
import cv2

class FootballDetector:
    """
    封装YOLO11目标检测，针对足球场景优化

    为什么用supervision封装：
    - 提供ByteTrack的简洁接口
    - 内置足球场景常用的可视化工具
    - 减少重复代码
    """

    def __init__(self, model_path: str = "yolo11m.pt"):
        # YOLO11m：中等规模，在A10上约80fps，精度足够Demo
        # 如需更高精度用yolo11l，更高速度用yolo11s
        self.model = YOLO(model_path)

        # ByteTrack追踪器配置
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,  # 低置信度追踪激活阈值
            lost_track_buffer=30,              # 丢失30帧（1s）后删除轨迹
            minimum_matching_threshold=0.8,   # 匹配阈值
            frame_rate=30                      # 帧率
        )

        # 球场单应矩阵（初始化时标定）
        self.homography_matrix = None

    def detect_and_track(self, frame: np.ndarray) -> dict:
        """
        单帧检测与追踪

        返回格式：
        {
            "players": [{"track_id": 1, "bbox": [x1,y1,x2,y2], "team": "home/away", "position": [x, y]}, ...]
            "ball": {"position": [x, y]} or None
        }
        """
        # 运行YOLO检测
        results = self.model(frame, verbose=False)[0]

        # 转换为supervision格式
        detections = sv.Detections.from_ultralytics(results)

        # 过滤：只保留person(0)和sports_ball(32)
        # COCO数据集类别ID
        person_mask = detections.class_id == 0
        ball_mask = detections.class_id == 32

        # 球员追踪
        player_detections = detections[person_mask]
        tracked_players = self.tracker.update_with_detections(player_detections)

        # 获取球的位置
        ball_detections = detections[ball_mask]
        ball_pos = None
        if len(ball_detections) > 0:
            # 取置信度最高的球检测结果
            best_ball_idx = np.argmax(ball_detections.confidence)
            ball_bbox = ball_detections.xyxy[best_ball_idx]
            ball_pos = [
                (ball_bbox[0] + ball_bbox[2]) / 2,
                (ball_bbox[1] + ball_bbox[3]) / 2
            ]

        # 转换到球场坐标系
        players = []
        for i, track_id in enumerate(tracked_players.tracker_id):
            bbox = tracked_players.xyxy[i]
            pixel_pos = [
                (bbox[0] + bbox[2]) / 2,  # 中心x
                bbox[3]                    # 脚底y（更准确反映位置）
            ]

            field_pos = self._pixel_to_field(pixel_pos) if self.homography_matrix is not None else pixel_pos

            players.append({
                "track_id": int(track_id),
                "bbox": bbox.tolist(),
                "position": field_pos,   # 球场坐标（0-105, 0-68）
                "team": "unknown"        # Phase 1再做球队分类
            })

        return {"players": players, "ball": {"position": ball_pos} if ball_pos else None}

    def _pixel_to_field(self, pixel_pos: list) -> list:
        """
        使用单应矩阵将像素坐标转换为球场标准坐标

        球场标准坐标系：左下角(0,0)，右上角(105,68)，单位：米
        与StatsBomb、Opta等数据标准一致
        """
        pt = np.array([[pixel_pos[0], pixel_pos[1]]], dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pt, self.homography_matrix)
        return transformed[0][0].tolist()

    def calibrate_homography(self, src_points: list, dst_points: list):
        """
        标定单应矩阵

        src_points：图像中的像素坐标（4个角点，如球场角点）
        dst_points：对应的球场实际坐标（已知）

        使用方法：
        - 在第一次使用前，手动点击摄像机画面中的4个球场角点
        - 输入这4个点的实际球场坐标（来自球场图纸）
        """
        src = np.array(src_points, dtype=np.float32)
        dst = np.array(dst_points, dtype=np.float32)
        self.homography_matrix, _ = cv2.findHomography(src, dst)
```

### 2.2 视频处理主Loop

```python
# video_pipeline.py (续)

import asyncio
from pathlib import Path
import json

async def analyze_video(video_path: str, job_id: str) -> dict:
    """
    异步分析视频文件，返回结构化分析结果

    使用asyncio而非threading，因为I/O密集的操作（文件读写、API调用）
    可以通过asyncio高效并发，GPU推理部分虽然是CPU密集但放在executor中
    """
    detector = FootballDetector("models/yolo11m.pt")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 分析结果存储
    timeline = []       # 时间线数据：每秒一个快照
    all_positions = []  # 所有帧的球员位置

    frame_count = 0
    sample_interval = max(1, int(fps / 5))  # 每秒采样5帧（降低计算量）

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % sample_interval != 0:
            continue

        # 检测与追踪（放入executor避免阻塞事件循环）
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,  # 使用默认ThreadPoolExecutor
            detector.detect_and_track,
            frame
        )

        timestamp = frame_count / fps
        snapshot = {
            "time": timestamp,
            "players": result["players"],
            "ball": result["ball"]
        }
        all_positions.append(snapshot)

        # 每秒生成一个战术快照
        if int(timestamp) > (len(timeline)):
            formation = detect_formation([p["position"] for p in result["players"][:10]])
            timeline.append({
                "time": int(timestamp),
                "formation": formation,
                "possession_zone": estimate_possession_zone(result)
            })

        # 报告进度（通过Redis）
        if frame_count % (sample_interval * 30) == 0:
            progress = frame_count / total_frames * 100
            # redis_client.set(f"job:{job_id}:progress", progress)
            print(f"Job {job_id}: {progress:.1f}%")

    cap.release()

    # 聚合分析结果
    return {
        "job_id": job_id,
        "duration": total_frames / fps,
        "timeline": timeline,
        "heatmap_data": compute_heatmap(all_positions),
        "player_stats": compute_player_stats(all_positions),
        "formation_summary": summarize_formations(timeline)
    }


def detect_formation(positions: list) -> str:
    """
    基于K-Means聚类识别阵型

    算法：
    1. 按y坐标排序（从守门员到前锋）
    2. 去除守门员（y最小的1人）
    3. 对剩余10人做K-Means（K=3或K=4）
    4. 统计每行的球员数，生成如"4-3-3"的字符串

    局限性：
    - 只对进攻阵型有效（防守时球员分布不规律）
    - K值需要预先假设行数
    """
    from sklearn.cluster import KMeans

    if len(positions) < 10:
        return "unknown"

    positions_arr = np.array(positions[:10])  # 只取10个外场球员

    # 假设3行（后卫-中场-前锋）
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(positions_arr)

    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # 按y坐标（从小到大=从后到前）排序
    sorted_idx = np.argsort(centers[:, 1])

    # 统计每行人数
    row_counts = []
    for idx in sorted_idx:
        count = np.sum(labels == idx)
        row_counts.append(count)

    # 标准化为常见阵型
    formation_str = "-".join(map(str, row_counts))
    return formation_str


def compute_heatmap(all_positions: list) -> dict:
    """
    计算球员位置热图数据
    返回格式：{"home": [[x,y,intensity],...], "away": [[x,y,intensity],...]}
    """
    home_positions = []
    away_positions = []

    for snapshot in all_positions:
        for player in snapshot["players"]:
            pos = player.get("position")
            if pos and len(pos) == 2:
                if player.get("team") == "home":
                    home_positions.append(pos)
                else:
                    away_positions.append(pos)

    return {
        "home": home_positions,
        "away": away_positions,
        "all": home_positions + away_positions
    }
```

---

## 三、Claude API 战术问答（Day 3-4）

### 3.1 LLM Coach 服务

```python
# llm_coach.py
import anthropic
from typing import Optional
import json

client = anthropic.Anthropic()  # 从环境变量读取ANTHROPIC_API_KEY

# 系统提示词设计原则：
# 1. 明确角色（专业足球战术顾问）
# 2. 约束输出格式（简洁、可操作）
# 3. 指定知识边界（基于数据，不要臆测）
# 4. 中文回答

SYSTEM_PROMPT = """你是一位精通现代足球战术的AI教练助手，拥有丰富的足球分析经验。
你协助场边教练做战术决策，你的建议需要：

1. **简洁可操作**：教练只有30秒时间听你说，给出1-3条核心建议
2. **基于数据**：当有比赛数据时，引用具体数字支撑判断
3. **专业准确**：使用专业足球术语，但避免过度技术化
4. **及时性**：优先考虑比赛当前状态，而非泛泛而谈

当前比赛数据将在用户消息中提供。如果没有数据，请基于问题给出通用战术建议，并注明是一般性建议。

回答格式：
- 核心判断（1-2句）
- 具体建议（1-3条，用"→"标注）
- 风险提示（可选，用"⚠️"标注）

禁止：推测球员受伤状态、批评球员个人、做超出战术范围的评论。"""


class TacticalCoach:

    def __init__(self, knowledge_base_path: Optional[str] = None):
        self.conversation_history = []
        # Phase 0：简单的文件知识库，Phase 1再换Qdrant
        self.knowledge_base = self._load_knowledge_base(knowledge_base_path)

    def _load_knowledge_base(self, path: Optional[str]) -> str:
        """加载战术知识库（Phase 0用简单文本文件）"""
        if not path:
            # 内置基础知识
            return """
足球基础战术知识：

阵型特点：
- 4-3-3：进攻性强，适合控球打法；三前锋创造宽度；中场需要高跑动量
- 4-4-2：平衡，双前锋配合；中场需要传控和跑动结合；适合边路进攻
- 3-5-2：中场人数优势；翼卫需要覆盖全边路；防守需要三中卫默契配合
- 4-2-3-1：双后腰保护；三中场技术要求高；单前锋需要强支点能力

换人原则：
- 体能下降标志：跑动距离明显减少、反应速度变慢、防守位置错误增加
- 战术换人时机：比分领先时加强防守（换边锋换边后卫）；落后时提高进攻（换双前锋/全面进攻）
- 伤停球时间换人：节省体力，准备最后15分钟

高位压迫要点：
- PPDA < 8 表示压迫有效（每8次对方传球才有1次我方防守行动）
- 压迫触发信号：对方守门员持球、对方回传、对方边后卫拿球时
- 压迫失效表现：被简单长球越过、体能不足跑不到位
"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except:
            return ""

    def ask(self, question: str, match_data: Optional[dict] = None) -> str:
        """
        向AI教练提问

        match_data格式：
        {
            "score": "1:0",
            "time": 67,
            "formation": "4-3-3",
            "possession": 58,
            "ppda": 7.2,
            "players": [{"name": "X", "distance_km": 8.5, "sprints": 12}, ...]
        }
        """
        # 构建上下文消息
        user_message = question

        if match_data:
            context = f"""
当前比赛数据（第{match_data.get('time', '?')}分钟）：
- 比分：{match_data.get('score', '未知')}
- 我方阵型：{match_data.get('formation', '未知')}
- 控球率：{match_data.get('possession', '?')}%
- 压迫强度（PPDA）：{match_data.get('ppda', '?')}

球员状态：
{self._format_player_stats(match_data.get('players', []))}

---
教练问题：{question}
"""
            user_message = context

        # 加入知识库上下文（简单拼接，Phase 1改为RAG）
        if self.knowledge_base:
            system_with_kb = SYSTEM_PROMPT + f"\n\n参考战术知识库：\n{self.knowledge_base[:2000]}"
        else:
            system_with_kb = SYSTEM_PROMPT

        # 调用Claude API
        # 使用多轮对话保持上下文（比赛中教练可能追问）
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,   # 限制长度，确保简洁
            system=system_with_kb,
            messages=self.conversation_history
        )

        assistant_message = response.content[0].text
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })

        # 保留最近10轮对话（避免token超限）
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

        return assistant_message

    def _format_player_stats(self, players: list) -> str:
        if not players:
            return "暂无球员数据"
        lines = []
        for p in players[:6]:  # 只显示前6人，节省token
            lines.append(f"  - {p.get('name', '未知')}: 跑动{p.get('distance_km', '?')}km, 冲刺{p.get('sprints', '?')}次")
        return "\n".join(lines)

    def reset_conversation(self):
        """新比赛时重置对话历史"""
        self.conversation_history = []


# 示例测试
if __name__ == "__main__":
    coach = TacticalCoach()

    match_data = {
        "score": "0:1",
        "time": 67,
        "formation": "4-3-3",
        "possession": 42,
        "ppda": 12.5,
        "players": [
            {"name": "球员A", "distance_km": 9.2, "sprints": 8},
            {"name": "球员B", "distance_km": 7.1, "sprints": 5},
        ]
    }

    answer = coach.ask("我们落后1球，左路老是被反击，怎么调整？", match_data)
    print(answer)
```

---

## 四、FastAPI 主服务（Day 4-5）

```python
# main.py
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
import os
import asyncio
from pathlib import Path

from video_pipeline import analyze_video
from llm_coach import TacticalCoach
from whisper_stt import transcribe_audio

app = FastAPI(
    title="CoachMind AI Demo",
    description="足球教练智能辅助系统 Demo API",
    version="0.1.0"
)

# CORS：允许React前端访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Demo阶段放开，Production需限制
    allow_methods=["*"],
    allow_headers=["*"],
)

# 存储分析任务状态（Demo用内存，Production用Redis）
job_store = {}

# 全局TacticalCoach实例（Demo用单实例）
coach = TacticalCoach()


class QuestionRequest(BaseModel):
    question: str
    match_data: dict | None = None

class CalibrationRequest(BaseModel):
    src_points: list  # 4个像素坐标点
    dst_points: list  # 4个球场实际坐标点


@app.post("/api/video/upload")
async def upload_video(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    上传视频并开始异步分析

    返回job_id，前端通过/api/video/{job_id}/status轮询进度
    """
    # 保存文件
    job_id = str(uuid.uuid4())[:8]
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)

    file_path = upload_dir / f"{job_id}_{file.filename}"
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # 记录任务
    job_store[job_id] = {
        "status": "processing",
        "progress": 0,
        "file_path": str(file_path),
        "result": None
    }

    # 异步后台分析
    background_tasks.add_task(run_analysis, job_id, str(file_path))

    return {"job_id": job_id, "message": "分析已开始"}


async def run_analysis(job_id: str, video_path: str):
    """后台任务：运行视频分析"""
    try:
        result = await analyze_video(video_path, job_id)
        job_store[job_id]["status"] = "completed"
        job_store[job_id]["result"] = result
    except Exception as e:
        job_store[job_id]["status"] = "failed"
        job_store[job_id]["error"] = str(e)


@app.get("/api/video/{job_id}/status")
async def get_job_status(job_id: str):
    """轮询分析任务状态"""
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="任务不存在")

    job = job_store[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job.get("progress", 0),
        "result": job["result"] if job["status"] == "completed" else None
    }


@app.post("/api/coach/ask")
async def ask_coach(request: QuestionRequest):
    """
    向AI教练提问

    前端可以每隔几分钟主动请求"当前比赛状态建议"
    或在特定事件（失球、换人时机）触发问答
    """
    try:
        answer = coach.ask(request.question, request.match_data)
        return {"answer": answer, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI服务暂时不可用: {str(e)}")


@app.post("/api/voice/transcribe")
async def transcribe_voice(file: UploadFile = File(...)):
    """语音输入转文字（Whisper）"""
    audio_path = f"/tmp/{uuid.uuid4()}.wav"
    with open(audio_path, "wb") as f:
        f.write(await file.read())

    text = transcribe_audio(audio_path)
    os.remove(audio_path)

    return {"text": text}


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "demo-0.1.0"}
```

---

## 五、Whisper 语音识别（Day 5）

```python
# whisper_stt.py
import whisper
import numpy as np

# 加载模型（第一次运行会自动下载）
# large-v3是最新最准确版本，中文WER约2.7%
# 如果GPU内存不足，用medium（效果略差但更快）
_model = None

def get_model():
    global _model
    if _model is None:
        print("加载Whisper large-v3模型...")
        _model = whisper.load_model("large-v3")
        print("模型加载完成")
    return _model


def transcribe_audio(audio_path: str) -> str:
    """
    音频转文字

    为什么选Whisper v3 large：
    - 中文WER（词错误率）约2.7%，是开源最好水平
    - 对口音和嘈杂环境（球场噪音！）有较强鲁棒性
    - 支持中英文混合（教练常用中英混搭术语）
    - 完全本地运行，无隐私问题

    注：在A10上，large-v3处理1分钟音频约需5秒
    如需更快速度，可用medium（约1秒）但中文精度稍低
    """
    model = get_model()

    result = model.transcribe(
        audio_path,
        language="zh",              # 指定中文（不指定则自动检测，稍慢）
        task="transcribe",          # transcribe=原语言转文字；translate=翻译成英文
        initial_prompt="这是一段足球训练场上的教练指令，",  # 给模型提示上下文，提高足球术语识别率
        temperature=0.0,            # 确定性输出（不随机采样）
        best_of=1
    )

    return result["text"].strip()
```

---

## 六、Demo前端（Day 6-9）

### 6.1 React前端核心组件

```bash
# 创建React项目
npx create-react-app demo-frontend --template typescript
cd demo-frontend
npm install recharts axios react-hot-toast
```

前端关键页面：
1. **视频上传页**：拖拽上传 → 进度条 → 跳转分析页
2. **分析报告页**：战术热图 + 阵型图 + 球员数据 + AI问答框
3. **AI问答框**：文字输入 + 语音录制按钮 → 实时展示AI回答

---

## 七、北体大验证准备（Week 4）

### 7.1 验证清单

```
技术验证：
□ 视频分析Pipeline完整可用
□ AI问答回答质量符合专业标准（内部评审）
□ 界面教练可独立操作（无培训）
□ 系统稳定运行2小时无崩溃

用户测试方案：
□ 邀请3位教练（国家级/高级/基层各1位）
□ 提供2段真实比赛录像（各约10分钟）
□ 每人独立试用30分钟
□ 收集结构化反馈（问卷+访谈）

反馈问卷（10题）：
1. AI战术建议的专业性如何？（1-10分）
2. 视觉热图是否帮助你理解比赛？（1-10分）
3. 你会在正式比赛中使用这个工具吗？（是/否/可能）
4. 哪个功能最有价值？（多选）
5. 哪个功能最需要改进？（开放填写）
...
```

### 7.2 预期验证结果与决策树

```
如果教练满意度 >= 7/10：
  → 按计划推进Phase 1（硬件采购）

如果 5-7/10：
  → 重点收集反馈，2周快速迭代后再次验证

如果 < 5/10：
  → 暂停Phase 1，与北体大教练深度访谈重新定义需求
```

---

## 附录：Demo演示脚本（给教练看）

1. "教练，这是我们AI系统的试用版。让我展示一场10分钟的比赛分析。"
2. 上传视频，等待分析（约3分钟）
3. "这里是球员位置热图，红色区域是球员活动最密集的地方。您可以看到中路有明显空当。"
4. "您可以直接问我任何战术问题。比如：'我们右路为什么老是被突破？'"
5. 展示AI回答，让教练评价专业性
6. "下周我们会把这套系统升级，支持比赛中实时分析。"

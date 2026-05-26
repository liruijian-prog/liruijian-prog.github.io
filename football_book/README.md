# 人才漏斗阅读器原型

这是一个独立于书稿目录的手机阅读器原型，目标是：

- 用 `Markdown` 作为内容真源
- 自动构建成阅读器可消费的 `JSON`
- 提供接近微信读书的移动端长文阅读体验
- 支持你后续修订书稿后重新构建并更新内容

## 1. 构建内容

默认会优先读取最新的 `outputs/book_终稿前收口版_*/manuscript/`。

```bash
cd /Users/work-0110/projects/youth-football-research
python3 reader/build_content.py
```

如果你以后想切换到别的书稿目录：

```bash
python3 reader/build_content.py --source outputs/book_终稿前收口版_2026-05-22/manuscript
```

## 2. 本地打开

阅读器是静态/PWA 站点，不能直接双击 `file://` 打开，需通过 HTTP 服务访问。

```bash
python3 -m http.server 8000 -d reader
```

浏览器访问：

```text
http://localhost:8000
```

如果要在手机上同一局域网内访问，可以用你电脑的局域网 IP：

```text
http://<你的局域网IP>:8000
```

## 3. 后续修订怎么更新

流程很简单：

1. 继续修订 `md` 书稿
2. 一键发布：

```bash
python3 scripts/deploy_reader.py
```

它会自动完成：

- 重建 `reader/content/`
- 同步到 GitHub Pages 仓库 `liruijian-prog/talent-funnel-reader`
- 等待线上构建完成并打印可访问地址

当前线上地址说明：

- 根域入口：`https://liruijian-prog.github.io/`
- 实际部署页：`https://liruijian-prog.github.io/talent-funnel-reader/`
- 根域仓库 `liruijian-prog.github.io` 目前用于跳转到阅读器项目页

常用参数：

```bash
python3 scripts/deploy_reader.py --skip-wait
python3 scripts/deploy_reader.py --source outputs/book_终稿前收口版_2026-05-22/manuscript
```

阅读器启动后会比对 `content/book.json` 的版本号并提示刷新。

## 4. 当前能力

- 手机优先的阅读排版
- 目录导航
- 搜索
- 章节进度保存
- 收藏本章
- 图表点击放大
- 章节内段落批注（本地存储）
- 术语点按弹窗与全书术语索引
- 脚注/参考文献卡片抽屉
- 简单离线缓存（PWA）

## 5. 新增内容 schema

重新运行构建后，阅读器现在会生成：

- `content/glossary.json`
- 每章 `referenceCards`
- 每章 `blockIndex`

这些字段分别用于：

- 术语弹窗
- 脚注卡片
- 章节段落批注锚点

## 6. 部署提醒

如果把整个 `reader/` 部署到 GitHub Pages 或其他公网静态托管：

- 手机可以随时打开
- 当前推荐对外分享根域入口 `https://liruijian-prog.github.io/`，实际内容仍部署在项目页 `https://liruijian-prog.github.io/talent-funnel-reader/`
- 你后续只要重跑 `python3 reader/build_content.py` 并重新发布，就能更新内容
- 但书稿内容会变成公网可访问资源，适合“愿意公开阅读”的版本，不适合保密稿

## 7. 当前边界

- 这是阅读器原型，不是出版社排版系统
- `Word` 仍然更适合精确排版和打印稿审校
- `HTML` 更适合手机阅读和持续更新

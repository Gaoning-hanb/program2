# coursebook-digest · 教材方法库

把大学教材**蒸馏成结构化“方法卡片”**（概念 / 定理 / 解题方法 / 典型例题 / 易错点），
学生在提问时**优先注入课本的逻辑与运算技巧**再作答。面向大学应试场景：**每学期一本书、
课程之间不连续**，因此所有数据按 `course`（课程）命名空间隔离，互不串味。

```
教材(PDF/MD) ──parse──► 章节文本 ──distill(LLM)──► 方法卡片 ──store──► JSONL + Chroma 向量
B站视频(链接) ──video──► 字幕/转写(带时间戳) ─┘                                  │
                                                                        │
学生题目 ──find_methods──► Top-K 方法卡片 ──inject(优先于通用知识)──► LLM 作答
                 └─ notes ──► markmap 思维导图(HTML) + 章节复习笔记(MD，可跳回视频原时刻)
```

## 目录结构

```
program2/coursebook-digest/
├─ coursebook_digest/
│  ├─ schema.py      # MethodCard / MethodKind / RetrievedMethod（统一数据模型）
│  ├─ config.py      # Settings（.env / 环境变量；路径以项目根为基准）
│  ├─ llm.py         # OpenAI 兼容客户端（默认 DeepSeek）+ 结构化 JSON 自纠
│  ├─ parser.py      # mineru 生产级解析（PDF，本地 CLI）+ md/txt 直读，无 pypdf 兜底
│  ├─ mineru_http.py # 云端解析后端 mineru-http（学校网关，鉴权复用模型 key）
│  ├─ distill.py     # 分块 → LLM 抽方法卡片 → 去重赋稳定 id（教材/视频双 prompt）
│  ├─ video.py       # B站视频 → 字幕抓取/whisper转写 → 时间戳锚点章节（视频教材源）
│  ├─ notes.py       # 方法卡片 → markmap 思维导图(HTML) + 章节复习笔记(MD)
│  ├─ store.py       # 按课程 JSONL + Chroma 集合 + 离线词面嵌入/检索
│  ├─ retrieve.py    # find_methods：词面分 + 向量分融合排序
│  ├─ answer.py      # 把命中方法渲染成“优先采用”注入块，再让 LLM 作答
│  └─ cli.py         # coursebook 命令
├─ examples/sample_course.md   # 示例教材片段（离线演示/测试用）
├─ test_smoke.py              # 离线冒烟测试（不联网）
├─ .env.example → 复制为 .env
└─ data/                      # 运行后生成（课程 jsonl、chroma、parsed）
```

## 环境与安装

运行环境为 conda 环境 **firpro**。所需依赖**firpro 已全部具备**（openai / pydantic /
pydantic-settings / chromadb / onnxruntime / typer / tenacity / pyyaml）。
无需新装包即可跑通骨架。

**注意**：PDF 解析走 mineru 家族后端——`mineru`（本机 CLI）或 `mineru-http`
（学校网关云端，见下节）。已移除 pypdf 纯文本兜底（公式/排版质量不打折）。

## 加速蒸馏（800 页 PPT 也适用）

蒸馏是耗时大头（模型“长 JSON 输出”生成慢，实测单块 ~107s）。三个可叠加的提速手段，
全部默认开启，均可按需关闭：

| 手段 | 默认 | 说明 |
|---|---|---|
| **蒸馏前去噪** `denoise` | 开 | 蒸馏前用纯规则（零 LLM）剔除**背景/人物生平/PPT 模板杂讯**（谢谢、占位符、导航、目录、页码、网址、纯装饰行等），技术/公式/标题/列表/页码锚点一律保留、宁可少删不误删。`--no-strip` 关闭 |
| **加大分块** `DISTILL_CHUNK_CHARS` | 8000 | 每块更大 → 调用次数更少（原 6000） |
| **并行蒸馏** `DISTILL_PARALLEL` | 3 | 各分块独立可安全并发，墙钟约 /N（学校网关若有并发额度，可调 `--parallel`/env） |

```bat
:: 典型提速用法（默认已开）：
coursebook ingest "xxx.pdf" --course 某课程 --parser mineru-http
:: 显式调大并发：
coursebook ingest "xxx.pdf" --course 某课程 --parser mineru-http --parallel 4
:: 关闭去噪（想保留背景内容时）：
coursebook ingest "xxx.pdf" --course 某课程 --no-strip
```

去噪只影响**蒸馏的输入**（删除非事实性背景），`parse` 命令输出的仍是原始解析文本，
可随时抽查去噪是否有误删。

## 云端解析（mineru-http）· 不用本机 mineru/GPU

学校网关提供了云端 MinerU 服务（`POST https://api.llm.ustc.edu.cn/mineru/file_parse`），
**鉴权复用同一个 `DEEPSEEK_API_KEY`**，无需本机安装 mineru、无需 GPU：

```bat
:: 整本入库：--parser mineru-http 走云端；自动拆片，完成后入库
coursebook ingest "books\liangzi.pdf" --course 量子物理 --parser mineru-http

:: 或用 --slice-pages 控制每片页数（云端按文件上传，默认 ≤30 页/片、≤8 片/次请求）
coursebook ingest "books\liangzi.pdf" --course 量子物理 --parser mineru-http --slice-pages 40
```

- `--parser auto`（不显式指定）：**开始运行时弹出 本地/云端 选择菜单**（只列当前
  可用的后端，回车默认云端）；非交互（脚本/PIPE）下自动按“有 key 且无本地 → 云端”选择。
- **PPT / DOCX / 图片**：走云端整文件直传（`--parser mineru-http`，MinerU 原生支持，
  无需拆片）；PDF 才需要拆片。超大体量文件若网关拒收，可先导出为 PDF 再走拆片路径。
- 云端产物同样落盘到 `<pdf>同目录/mineru_out/partNN/*.md`，**续跑/复用**全部沿用
  `--mineru-out` / `--reuse-mineru` 语义：已落盘的片自动跳过。
- 云端返回的 md **不带页码锚点**：章节仍按“第X章”切分（`Chapter.pages` 元数据为空，
  不影响蒸馏/检索/作答）。配额度可在网关控制台查看，或 `coursebook env` 看云端配置。
- 云端鉴权可单独覆盖：`.env` 里设 `MINERU_API_KEY`/`MINERU_BASE_URL`（默认同模型网关）。

按需（可选，自行安装）：

```bat
:: 生产级教材解析：公式/扫描件/复杂排版（强烈建议，体积较大）
conda activate firpro && pip install "mineru>=2.0"

:: 更准确的中文分词检索（不装也有内置回退分词）
conda activate firpro && pip install jieba>=0.42

:: 语义向量：想用 MiniLM 而非离线 hash 嵌入时（首次使用需联网下载模型）
:: 在 .env 里设 EMBEDDING_MODE=mini，同时依赖 onnxruntime（已在 firpro）
```

> PowerShell 里激活等价方式：`cmd /c "call D:\Aitest\Scripts\activate.bat firpro && ..."`

## 快速开始

```bat
:: 1) 配置密钥
copy .env.example .env        :: 填入 DEEPSEEK_API_KEY

;; 2) 导入一本教材（pdf: --parser mineru-http 云端 / 本地 mineru；md / txt 直读）
coursebook ingest "《高等数学》.pdf" --course 高数

:: 3) 只先解析看质量（不蒸馏）
coursebook parse "《高等数学》.pdf" --course 高数

:: 4) 检索命中的方法卡片
coursebook find "求极限，分子分母都趋于0" --course 高数 --top-k 5

:: 5) 提问：检索 → 优先注入课本方法 → 作答
coursebook ask "证明当 x→0 时 sin x ~ x" --course 高数

:: 6) 其它
coursebook courses    :: 列出已建课程
coursebook env        :: 环境自检
```

无 API Key 也能测检索：`test_smoke.py`（离线）；`coursebook find` 只检索不调模型。

## B站视频 → 方法卡片（看课自学场景）

很多同学习惯在B站看课程视频自学。`coursebook video` 把**视频当作另一种教材源**：
贴个链接，自动取字幕（或本地转写）→ 蒸馏成方法卡片 → 入库，之后 `find`/`ask`
照常检索；`coursebook notes` 还能生成思维导图和复习笔记，**每张卡带时间戳，
可一键跳回视频原时刻**——纸质教材给不了的复习体验。

```bat
:: 一条命令：字幕/转写 → 蒸馏 → 入库（视频系列独立成课，不污染教材库）
coursebook video "https://www.bilibili.com/video/BVxxxxxxxxxx" --course 王道408强化

:: 只处理指定分P（长系列先试一两P看质量）
coursebook video "https://www.bilibili.com/video/BVxxxxxxxxxx" --course 王道408强化 --parts 2,5-7

:: 生成思维导图 + 复习笔记（纯代码秒出，不调模型）
coursebook notes --course 王道408强化

:: 照常提问（检索/作答与教材课程完全同构）
coursebook ask "什么是数据的逻辑结构和存储结构" --course 王道408强化
```

**字幕获取策略**（自动，无需操心）：

1. **B站字幕优先**：UP主上传的 CC 字幕 / 官方 AI 字幕，秒级拿到。AI 字幕
   需要登录态——在 `.env` 填 `BILIBILI_SESSDATA`（浏览器登录B站 → F12 →
   Cookie → 复制 SESSDATA 值）后即可直取；
2. **whisper 本地转写兜底**：无字幕的分P自动下载音频，用 faster-whisper
   转写（**GPU 自动优先**，RTX 4050 实测 82 分钟课程约 11 分钟；CPU 也可跑，
   约与视频等长）。转写文本带 `[mm:ss]` 时间戳锚点，蒸馏出的卡片溯源到
   `P2 12:35` 这种粒度。

**视频蒸馏的特别处理**：口语转写有同音错字（“真体”→“真题”）、有闲聊噪音
（求三连/下节预告）——视频专用 prompt 会按上下文纠正术语、剔除编排性内容，
只留知识点。转写结果缓存在 `data/transcripts/<课程>/`，重跑自动复用，
断点续跑不重复转写。

**两个实测坑与对策**（王道强化班 44P 系列实测）：

- **录播课含课间音乐**：直播录制的课程视频常带几十分钟课间休息 BGM。
  转写启用 VAD（语音活动检测）自动跳过无语音段——实测 82 分钟视频只有
  22 分钟讲授，VAD 只转写讲授部分（墙钟 59s、零音乐幻觉），另有音乐
  署名类幻觉兜底过滤。若转写覆盖远小于视频时长会打印提示
  （“转写覆盖 22/83 分钟：其后无语音”），**不是漏转**。
- **B站风控（412）**：分P列表枚举有三层兜底——yt-dlp flat-playlist
  （自动带 buvid3/buvid4 指纹 cookie）→ view API → 逐P单视频探测
  （对风控最宽容，实测严打期仍全通）。指定 `--parts` 时逐P探测只探到
  所需最大P，大系列不浪费。仍然失败时配 `BILIBILI_SESSDATA` 登录态
  基本可解。

依赖（仅 video 功能需要）：`pip install yt-dlp faster-whisper nvidia-cublas-cu12`
（末者为 Windows NVIDIA GPU 加速所需，纯 CPU 可不装；都不装则 video 命令给出明确指引）。

## 测试

```bat
python test_smoke.py      :: 6 项冒烟（存取/幂等、检索优先、蒸馏去重、注入块、视频时间戳/分P/溯源、导图笔记）
python test_pipeline.py   :: 6 项端到端集成（章节切分、md/PDF 解析、持久化+向量+融合检索、
                               ask 全链路优先注入、蒸馏链路）——全部离线、无需 key
```

## 设计要点

- **方法卡片是“可执行卡”**：含 `applicability`（何时用）、`steps`（按序步骤）、
  `core_formula_latex`（公式）、`error_notes`（坑），不是原文搬运。
- **检索双通道**：关键词命中（离线必通）+ 向量相似度（Chroma；离线 hash 嵌入兜底，
  联网可切 MiniLM 求语义）。方法/例题类加权，体现“作答优先用方法”的设计意图。
- **优先注入**：`ask` 把命中方法按相关度渲染在通用知识**之前**，并约束模型
  “先判适用条件→按步骤→引用公式→提醒易错点→最后交叉验证”。
- **课程隔离**：`data/courses/<course>.jsonl` + 每课程一个 Chroma collection，
  一门学期一本书互不干扰；换学期 = 换 `--course`。

## R2 Web 服务（FastAPI）· 私人教师 API/演示页

`TeacherAgent`（受控 agent 循环：意图分类 → 画像注入 → 课本+个人记忆检索 → 作答 → 写回
记忆/画像）已封装为 HTTP 服务，自带单页对话演示界面：

```bat
:: 启动（浏览器打开 http://127.0.0.1:8000/）
coursebook serve --host 127.0.0.1 --port 8000
:: 或
python -m coursebook_digest.api_server
```

| 接口 | 说明 |
|---|---|
| `GET /` | 内嵌演示页（输入学生id/课程/问题，含“今日打卡/我的画像”按钮） |
| `GET /api/health` | 服务与课程/模型状态 |
| `GET /api/courses` | 课程列表（含卡数） |
| `POST /api/teach` | `{question, course, user_id, writeback}` → 作答 + 命中方法与个人记忆 + 写回 + 画像 |
| `GET /api/profile?user_id=..&course=..` | 该生画像与四版块记忆计数 |
| `POST /api/daily_cards` | 每日抽取打卡卡（课本库 + 个人记忆库） |

- 个性化隔离：不同 `user_id` 的画像与记忆存于 `data/users/<uid>/`，互不串味。
- 离线降级：无 API Key/断网时接口照常返回（`offline:true` + 方法卡预览，不写回），适合比赛现场兜底。
- 网页是演示级入口；后端 `TeacherAgent` 也可被任何前端/DSH 插件直接 import 调用。

## 后续：包装成 DSH 插件

本仓库是**纯 Python 核心**（便于你在 firpro 里快速迭代）。把它接入 DSH 的方式：
一个 Cordis 宿主插件以子进程调用 `coursebook find/ask`（与 `dsh-bash-local`
包 shell 的方式同构）对外暴露工具服务，例如：

| DSH 工具 | 对应 CLI | 作用 |
|---|---|---|
| `find_methods(question, course)` | `coursebook find ... --output -` | 作答前检索课本方法 |
| `ask_coursebook(question, course)` | `coursebook ask ... --methods-json -` | 优先注入课本方法作答 |
| `ingest_coursebook(pdf, course)` | `coursebook ingest ...` | 学期初导入教材 |
| `list_courses()` | `coursebook courses` | 切换“本学期教材” |

工具 schema 进会话后，模型在回答课程问题时自然先调 `find_methods`，再把这些
方法以内联指导的形式排在通用知识前面——即你最初设想的“**优先考虑课本蒸馏出的方法**”。
（本仓库当前阶段只提供 CLI/核心，DSH 插件壳在后续迭代。）

## 分片 / 续跑（MinerU 大 PDF 推荐）

mineru 因"整批算完才写盘"且**无官方断点续传**，大 PDF 中断会白烧算力。缓解办法：**按页切片跑**，每片独立落盘固化，中断只损失当前片：

```bat
:: mineru 按页范围解析到不同子目录（-s/-e 从 0 起，234 页示例 -> 3 片）
mineru -p "books\liangzi.pdf" -o "books\mu\p1" -s 0   -e 77
mineru -p "books\liangzi.pdf" -o "books\mu\p2" -s 78  -e 155
mineru -p "books\liangzi.pdf" -o "books\mu\p3" -s 156 -e 233

:: 全部切片完成后，复用这些产物直接蒸馏入库（不再跑 mineru）
python -m coursebook_digest.cli ingest "books\liangzi.pdf" --course 量子物理 --mineru-out "books\mu" --reuse-mineru
```

> `--reuse-mineru` 会递归合并 `--mineru-out` 目录树下所有 `.md`（含各分片子目录），
> 按页锚点拼回顺序再统一切章；目录下没有 md 时明确报错。适合"解析已完成/部分完成"的复用与续跑。

## 中断清理（防"幽灵进程"烧 GPU）

mineru 的解析服务是独立子进程，Ctrl+C/关窗只杀主进程时它会残留继续烧 GPU。
本工具已内置自动清理：`ingest` 启动前会清掉上次残留的 mineru 进程；Ctrl+C 中断时
也会连带清理。工具内也可调用 `kill_mineru_processes()`（按命令行匹配，只杀 mineru 相关）。

## License / 注意

- 个人学习可处理自己拥有/学校配发的教材；勿做公开再分发。
- 蒸馏质量依赖 LLM：关键章节建议用 `coursebook find` 抽查命中率，必要时人工校对方法卡。
# AI Voice Memo Agent

一个使用 SwiftUI 和 AVFoundation 构建的 iPhone 录音应用。基础交互参考
Voice Memos，并加入录音转写、结构化摘要、语义搜索和可操作录音数据的 AI
Agent。

## 已实现功能

- 本地录音、暂停、播放、波形、快进/快退、裁剪和删除片段
- 录音重命名、收藏、复制、分享和删除
- 保存录音后自动触发 AI 转写和分析
- Summary 状态与缓存：`not_started`、`processing`、`ready`、`failed`、`expired`
- Transcript、Summary、Key Points、Action Items、建议标题和建议标签
- 基于 OpenAI Embedding 的跨录音语义搜索
- Responses API 工具调用 Agent
- 多轮会话中的“第二条”“它”“这条录音”等上下文
- Agent 重命名、加标签和删除前的人机确认
- 编辑音频后自动将旧分析标记为 `expired`

## 架构

```text
iPhone SwiftUI App
  ├─ AVFoundation: 录音、播放、编辑
  ├─ 本地 JSON + m4a: 数据源
  └─ URLSession
          ↓
Local FastAPI Service
  ├─ Audio Transcriptions: 语音转文字
  ├─ Structured Outputs: 摘要、要点、行动项
  ├─ Embeddings: 语义检索
  └─ Responses API + tools: Agent 工作流
          ↓
OpenAI API
```

OpenAI API Key 只保存在 Mac 的 `backend/.env`，不会写入或打包进 iPhone
App。录音仍以 iPhone 本地数据为准。

## 1. 配置 AI 服务

在项目根目录执行：

```bash
cp backend/.env.example backend/.env
open -e backend/.env
```

在文件中填写：

```text
OPENAI_API_KEY=你的API密钥
```

然后启动服务：

```bash
bash backend/start.sh
```

看到 `Uvicorn running on http://0.0.0.0:8000` 即启动成功。浏览器打开
`http://127.0.0.1:8000/docs` 可以查看 API。

## 2. 运行 iPhone App

1. 用 Xcode 16.4 或更新版本打开 `VoiceMemoAgent.xcodeproj`。
2. 在 Signing & Capabilities 选择你的 Team。
3. 选择 iPhone 模拟器或真机并运行。
4. 模拟器中的 AI Server URL 使用 `http://127.0.0.1:8000`。
5. 真机需要与 Mac 在同一网络，并在 Agent 设置中使用 Mac 的 `.local`
   地址，例如 `http://Amy-Mac.local:8000`。

Mac 主机名可以在终端查看：

```bash
scutil --get LocalHostName
```

## 测试

```bash
backend/.venv/bin/python -m unittest discover backend/tests

xcodebuild \
  -project VoiceMemoAgent.xcodeproj \
  -scheme VoiceMemoAgent \
  -destination 'generic/platform=iOS' \
  -derivedDataPath /tmp/VoiceMemoAgentDerivedData \
  CODE_SIGNING_ALLOWED=NO \
  build
```

## 数据与成本

分析结果保存在录音 metadata 中，因此重复打开 Summary 不会重复调用模型。
只有首次生成、失败重试或音频编辑后重新生成时才会产生新的 AI 请求。

# 1.0_项目初期 - Design Spec

## I. Project Information

| Item | Value |
| ---- | ----- |
| **Project Name** | 1.0_项目初期 |
| **Canvas Format** | PPT 16:9 (1280x720) |
| **Page Count** | 14 页 |
| **Design Style** | General Consulting + academic_defense |
| **Target Audience** | 导师、答辩评审、实验室同学 |
| **Use Case** | 项目答辩 / 研究汇报 |
| **Created Date** | 2026-03-25 |

---

## II. Canvas Specification

| Property | Value |
| -------- | ----- |
| **Format** | PPT 16:9 |
| **Dimensions** | 1280x720 |
| **viewBox** | `0 0 1280 720` |
| **Margins** | left/right 40px, top 0px, bottom 35px |
| **Content Area** | x=40-1240, y=135-650 |

---

## III. Visual Theme

### Theme Style

- **Style**: General Consulting
- **Theme**: Light theme
- **Tone**: 学术、克制、结构清晰、技术导向

### Color Scheme

| Role | HEX | Purpose |
| ---- | --- | ------- |
| **Background** | `#F7F9FC` | 页面背景 |
| **Secondary bg** | `#E8EEF5` | 卡片背景、观点条 |
| **Primary** | `#1F4E79` | 页眉、一级标题、核心强调 |
| **Accent** | `#2F6B8A` | 连线、次级装饰、图标 |
| **Secondary accent** | `#6FA3A8` | 局部强调、流程链路 |
| **Body text** | `#1A1A1A` | 正文文字 |
| **Secondary text** | `#5B6573` | 解释文字 |
| **Tertiary text** | `#7A8795` | 页脚、辅助说明 |
| **Border/divider** | `#C9D3DF` | 卡片边框、分隔线 |
| **Success** | `#2E8B57` | 正向提示 |
| **Warning** | `#C43D3D` | 风险、问题提示 |

---

## IV. Typography System

### Font Plan

**Recommended preset**: P1

| Role | Chinese | English | Fallback |
| ---- | ------- | ------- | -------- |
| **Title** | Microsoft YaHei | Arial | sans-serif |
| **Body** | Microsoft YaHei | Arial | sans-serif |
| **Code** | - | Consolas | Monaco |
| **Emphasis** | SimHei | Arial Black | Arial |

**Font stack**: `"Microsoft YaHei", "微软雅黑", Arial, sans-serif`

### Font Size Hierarchy

**Baseline**: Body font size = 18px

| Purpose | Ratio | 24px baseline (relaxed) | 18px baseline (dense) | Weight |
| ------- | ----- | ---------------------- | -------------------- | ------ |
| Cover title | 2.5-3x | 60-72px | 52-56px | Bold |
| Chapter title | 2-2.5x | 48-60px | 48-56px | Bold |
| Content title | 1.5-2x | 36-48px | 24-28px | Bold |
| Subtitle | 1.2-1.5x | 29-36px | 20-24px | SemiBold |
| **Body content** | **1x** | **24px** | **18px** | Regular |
| Annotation | 0.75-0.85x | 18-20px | 14-16px | Regular |
| Page number/date | 0.55-0.65x | 13-16px | 12-14px | Regular |

---

## V. Layout Principles

### Page Structure

- **Header area**: 70px 深蓝页眉，左红条装饰，左侧页标题，右侧项目名
- **Content area**: 135-650px，自由布局，以卡片、流程链路、结构框图为主
- **Footer area**: 55px 页脚，标注来源、章节与页码

### Common Layout Modes

| Mode | Suitable Scenarios |
| ---- | ----------------- |
| **Single column centered** | 封面、过渡页、结束页 |
| **Left-right split (5:5)** | 问题-定位、短板-归因 |
| **Left-right split (4:6)** | 图解 + 解释、机制 + 结论 |
| **Top-bottom split** | 总体架构、升级路线 |
| **Three/four column cards** | 能力概览、系统分层 |
| **Matrix grid** | 机制拆解、角色归类 |

### Spacing Specification

| Element | Recommended Range | Current Project |
| ------- | ---------------- | --------------- |
| Card gap | 20-32px | 20-28px |
| Content block gap | 24-40px | 24-32px |
| Card padding | 20-32px | 20-24px |
| Card border radius | 8-16px | 8px |
| Icon-text gap | 8-16px | 12px |

---

## VI. Icon Usage Specification

### Source

- **Built-in icon library**: `templates/icons/`
- **Usage method**: `<use data-icon="icon-name" ... />`

### Recommended Icon List

| Purpose | Icon Path | Page |
| ------- | --------- | ---- |
| 问题与目标 | `target` | 04 |
| 模型与结构 | `microchip` | 04/06/09 |
| 系统层次 | `layers` | 05/10 |
| 关键能力 | `circle-checkmark` | 05 |
| 指标/加速 | `gauge-high` | 06 |
| 缓存/记忆 | `database` | 09/10 |
| 参数控制 | `sliders` | 09 |
| 条件与地形 | `mountains` | 09/10 |
| 文本条件 | `comment-dots` | 09 |
| 写回更新 | `arrow-rotate-right` | 09 |
| 升级启发 | `lightbulb` | 13 |

---

## VII. Chart Reference List

| Chart Type | Reference Template | Used In |
| ---------- | ------------------ | ------- |
| `process_flow` | `templates/charts/process_flow.svg` | Slide 06 / Slide 13 |

---

## VIII. Image Resource List

| Filename | Dimensions | Ratio | Purpose | Type | Status | Generation Description |
| -------- | --------- | ----- | ------- | ---- | ------ | --------------------- |
| - | - | - | 本项目不使用外部图片，以结构图、流程图、卡片布局为主 | - | - | - |

---

## IX. Content Outline

### Part 1: 研究定位与系统理解

#### Slide 01 - Cover
- **Layout**: Full-screen title + centered author info
- **Title**: 基于 Live2Diff 的实时水墨风格视频渲染研究
- **Subtitle**: Ink-Diffusion 项目答辩汇报

#### Slide 02 - 目录
- **Layout**: Two-column TOC cards
- **Title**: 目录
- **Content**:
  - 研究定位与系统理解
  - 关键机制拆解
  - 瓶颈与改造方向
  - 升级路线与总结

#### Slide 03 - 章节页
- **Layout**: Chapter template
- **Title**: 研究定位与系统理解
- **Content**:
  - 交代研究对象、选型逻辑和系统边界

#### Slide 04 - 项目定位与研究问题
- **Layout**: Left-right split
- **Title**: 本项目聚焦实时扩散系统拆解与水墨导向改造，而不是重训新模型
- **Content**:
  - 研究问题
  - 项目定位
  - 答辩主线

#### Slide 05 - 为什么选择 Live2Diff
- **Layout**: Left-right split
- **Title**: Live2Diff 值得作为底座，因为它已解决实时视频扩散的共性难题
- **Content**:
  - 已具备的关键能力
  - 项目技术边界

#### Slide 06 - 系统架构与单帧主链
- **Layout**: Top-bottom split
- **Title**: 系统可拆为四层，单帧推理则由 latent、depth、UNet、scheduler 串成闭环
- **Chart**: process_flow
- **Content**:
  - 四层工程结构
  - 单帧在线推理主链

### Part 2: 关键机制拆解

#### Slide 07 - 章节页
- **Layout**: Chapter template
- **Title**: 关键机制拆解
- **Content**:
  - 解释实时性、UNet 中枢与缓存机制

#### Slide 08 - 实时性的真正来源
- **Layout**: Left chart right text
- **Title**: 实时性首先来自流式建模与少步采样，其次才是执行层优化
- **Content**:
  - 六步因果链
  - 进入实时区间与部署区间的区别

#### Slide 09 - UNet 为何是系统中枢
- **Layout**: Center core + surrounding nodes
- **Title**: 所有关键状态都在 Streaming UNet 汇合，因此它不只是普通去噪器
- **Content**:
  - 当前 noisy latent
  - depth latent
  - prompt embeds
  - KV-cache / attn_bias / pe_idx / update_idx

#### Slide 10 - 缓存与结构条件机制
- **Layout**: 2x2 cards
- **Title**: noisy batch、双 buffer、KV-cache 与 depth 条件一起，构成系统运行骨架
- **Content**:
  - noisy batch 的严格含义
  - PrevXT / PrevDepth Buffer
  - KV-cache 的作用与边界
  - Depth 在主路径里的真实角色

### Part 3: 瓶颈与改造方向

#### Slide 11 - 章节页
- **Layout**: Chapter template
- **Title**: 瓶颈与改造方向
- **Content**:
  - 从短板、归因到升级路线

#### Slide 12 - 当前系统的核心短板
- **Layout**: Left-right split
- **Title**: 现有系统对普通风格化已经很强，但离高质量水墨视频仍有明显距离
- **Content**:
  - 三个直接短板
  - 四个更本质的技术归因

#### Slide 13 - 升级路线与总结
- **Layout**: Top-bottom split
- **Title**: 升级方向可以压缩为两条主线：多条件融合，以及分层的面向风格采样调度
- **Chart**: process_flow
- **Content**:
  - 升级链路
  - 对应改造动作
  - 三条答辩结论

#### Slide 14 - Ending
- **Layout**: Ending template
- **Title**: 谢谢
- **Content**:
  - 欢迎老师批评指正

---

## X. Notes Strategy

- **Total duration**: 12-15 分钟
- **Notes style**: formal + conversational
- **Presentation purpose**: report / persuade

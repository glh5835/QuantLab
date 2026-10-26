# UTSC 双人项目笔记（会话迁移）

> 本文件由 2026-10-26 的会话 `session_653fce52480f7fb2285b27d164562c1c` 迁移而来，
> 内容与本项目（QuantLab）无关，仅作个人项目档案存放在此处。
> **分工以本人更正为准**：原会话分析曾把两人角色弄反，已在本文中纠正。

## 一、项目与分工

**项目**：UTSC 职业与实践导航网站（与 Seraphina 的双人项目）。
计划书 `UTSC_Two_Person_Code_Plan.docx` v1 已于 2026-10-26 冻结（口头确认），之后不宜再大改范围。

| 角色 | 负责人 | 范围 | 目录 |
|---|---|---|---|
| 开发负责人 | **我** | React+Supabase 网站、登录、存储、权限、部署 | `web/` `supabase/` |
| 数据负责人 | **Seraphina** | Python pipeline、合同、规则、数据包 | `src/` `config/` `tests/` `resources/` `contracts/` |

远端基线 commit `31635613`（此前已核对最新代码与 AGENTS.md）。

## 二、Seraphina 的数据侧任务（S0–D7，按依赖顺序）

1. **S0 共同合同**：角色/活动枚举、JSON Schema、fixtures；v1 冻结后单独 PR 先合。阻塞双方——开发侧要用 fixtures 搭网站，不等采集。
2. **D0 基线**：跑已有测试、记录环境+固定离线输入 → `docs/data_baseline.md`；不要先重构。
3. **D1 八类角色**：新增 3 类查询+正负关键词，调整 adjacent/outside_scope；每类 ≥5 正例、3 混淆负例（`config/settings.py` 等）。
4. **D2/D3 采集与累计**：24 组合每周各第一页、月度预算记账、可配置超时+退避；按 source_job_id 合并、快照不可覆盖（`history.py`、`market_export.py`）。
5. **D6 社团接入**（Instagram 专项，见第四节）→ `src/opportunities/`、`organizers.json`。
6. **D4/D5/D7**：市场数据包+参考统计；15–25 门课程+15–20 条资源（每条带来源和核实日期）；发布校验+损坏包拒收。

## 三、我的开发侧任务

A0 起步：等 S0 fixtures 就绪后用它们搭网站（React+Supabase），做登录/存储/权限/部署；不依赖采集数据，可并行推进。

## 四、D6 Instagram 专项方案

**背景**（2026-10-26）：Seraphina 发来 21 个社团 INS 账号表 `UTSC_Club_Instagram.xlsx`（9 数据/技术、11 商业/管理、1 金融/投资）+ 7 张帖子截图，说"先用这些"，并问"Instagram reel 内容要怎么让 AI 搞"。截图全是社团活动 posts（CAPTIVATE 叙事比赛、UTSFC 金融案例赛、CREATE 开发者招募、DS3 ML Bootcamp）——要的是**活动/招募类帖子**（岗位招募、竞赛组队、技能工作坊、行业交流），排除普通入会/加群宣传。

**角色归属**：D6 是 Seraphina（数据负责人）的管线任务；我有 INS 账号，负责提供账号访问并手动浏览、保存帖子素材（HTML/caption/截图）。

**采集范围（2026-10-26 已确认）**：账号以 21 个清单为准，`utscthebridge` **不纳入**；采 **posts+caption，reels 也要**。

**方案（自动爬虫优先，2026-10-26 拍板）**——先做合规性评估（Instagram 服务条款、合法访问方式、频率限制）：**合规则直接自动爬取；不合规才回退**到人工截图+半自动导入。红线不变：不绕过访问控制、图片 OCR 仅作候选文本、人工复核留痕。

- **路线 A（合规通过）**：自动爬取 → 数据侧批量解析成结构化 CSV → 关键词规则产生分类草稿（形式/主题/能力/门槛枚举）→ 人工复核日期、资格和不确定标签后统一导入。
- **路线 B（不合规回退）**：用自己 INS 账号人工浏览 21 个账号、保存帖子截图/caption → 其余同路线 A。

**试点 5 账号**：`ds3.utsc`、`create.uoft`、`investment_society`、`amacss_utsc`、`hackthevalley`（覆盖截图里的类型）；验证 caption 可得性、图片依赖度、更新稳定性后再扩量。

**字段模板**（xlsx 第 28 行，9 字段）：岗位名称、具体职责、技能要求、参与资格、时间投入、截止日期、申请链接、有薪/无薪、CCR 有无——原帖没写的标"未说明"，不许推断。

**验收口径**：整理 20–30 条可核实机会（不足就展示实际数量）；同源帖子去重；采集失败不删历史；数据带 source_post_id、verified_at、classification_reasons。

**采集范围**（原两项待确认已于 2026-10-26 拍板，见上）。

## 五、纪律（照旧执行）

- 离线可复现；不把 schema 通过当分类准确；不把本地逻辑说成端到端跑通；人工复核记录留痕。
- 每个任务开 issue+分支（如 `data/D1-roles`）；contracts 变更先单独 PR。
- 别让 AI 覆盖对方未提交的改动。

## 六、源材料位置

`E:\AI Analyzer3.0DSH\聊天记录\20261026\`：聊天文本.txt、11 张截图、`UTSC_Two_Person_Code_Plan.docx`（计划书全文）、`UTSC_Club_Instagram.xlsx`（21 社团账号表）。

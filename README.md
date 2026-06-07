# 珠海智能垃圾分类管家系统 - 项目总结

## 项目概述

### 项目名称
**珠海智能垃圾分类管家系统**

### 项目定位
面向珠海家庭用户的家用AI垃圾分类管理系统，适配南方湿热气候特点。

### 核心功能
- AI图像识别 - 基于MobileNetV2的垃圾图像分类
- 语音问答交互 - 支持文字/语音双模式输入
- 大模型集成 - 本地AI大模型（qwen3-0.6b）智能回答
- 投放指引 - 详细的分类和投放指导
- 气候适配 - 针对珠海湿热气候的专属提示
- 积分系统 - 环保积分、等级成长、每日打卡、成就徽章
- 数据统计 - 环保数据可视化（分类占比/趋势/热力图/对比）
- 设备管理 - 智能垃圾桶设备绑定与状态监控
- 排行榜 - 全平台用户环保排名

---

## 技术架构

### 后端技术栈
```
Flask Web框架
  ├── RESTful API 设计 (52个路由端点)
  ├── SQLite 数据库 (14张表)
  └── PyTorch 深度学习 (可选)

AI 大模型集成
  ├── OpenAI API 兼容接口
  ├── LM Studio / Ollama 支持
  └── 模型: qwen3-0.6b

前端技术
  ├── HTML5 + CSS3
  ├── JavaScript (原生ES5兼容)
  ├── Web Speech API (语音)
  └── 响应式设计 + 玻璃态UI
```

### 文件结构
```
c:\Users\31037\Desktop\python project\垃圾分类\
│
├── app_minimal.py              # 主应用文件（Flask服务）
├── trash_classifier.pth        # 训练好的PyTorch模型
├── main.py                     # 入口文件
├── train.py                    # 模型训练脚本
├── test.py                     # 测试脚本
├── data_loader.py              # 数据加载器
│
├── templates/                  # HTML模板目录 (16个页面)
│   ├── base.html               # 基础布局模板
│   ├── index.html              # 系统首页
│   ├── login.html              # 登录页面
│   ├── register.html           # 注册页面
│   ├── qa.html                 # 智能问答页面
│   ├── scan.html               # 图像识别页面
│   ├── guide.html              # 分类指南页面
│   ├── south_guide.html        # 南方气候指南
│   ├── profile.html            # 个人中心
│   ├── history.html            # 投放记录
│   ├── devices.html            # 设备管理
│   ├── notifications.html      # 消息通知
│   ├── points.html             # 积分中心
│   ├── leaderboard.html        # 排行榜
│   └── stats.html              # 数据统计
│
└── static/
    └── css/
        └── style.css           # 全局样式表
```

---

## 核心功能详解

### 1. 智能问答系统 (QA)

#### 功能特点
- 双模式切换: 规则匹配 / AI大模型
- 智能回退机制: 多级容错确保总能得到答案
- 终端详细日志: 完整记录AI处理过程
- 语音播报优化: 自动清理格式符号

#### 处理流程
```
用户提问
    ↓
    AI模式                    规则模式
    ↓                          ↓
 ① 调用AI                 ① 查规则库(33个关键词)
    ↓                          ↓
 成功? → 返回              匹配? → 返回答案
 ↓ 失败                      ↓ 未匹配
 ② 回退规则               ② 尝试AI补充
    ↓                          ↓
 匹配? → 返回              成功? → 返回AI回答
 ↓ 未匹配                    ↓ 失败
 ③ 重试AI                ③ 返回默认提示
    ↓
 成功? → 返回
 ↓ 失败
 ④ 友好提示
```

#### 规则库关键词 (33个)
| 分类 | 关键词 |
|------|--------|
| 厨余垃圾 | 苹果核、香蕉皮、鱼骨、肉骨、排骨、骨头、菜叶、剩饭、剩菜、果皮、蛋壳、茶叶渣、咖啡渣、鸡骨 |
| 可回收物 | 玻璃瓶、易拉罐、报纸、书本、衣服 |
| 有害垃圾 | 电池、灯管、药品、化妆品 |
| 其他垃圾 | 奶茶杯、纸巾、塑料袋、烟蒂、尿不湿、口罩、外卖盒、猪骨、牛骨 |

#### 语音播报清理
自动删除的符号:
- Markdown: `**` `*` `【` `】`
- 列表: `1.` `2.` `-` `•`
- Emoji: ✓ ✅ ❌ ⚠️ 💡 等
- 特殊字符: `→` `←` `—` `──`

---

### 2. 图像识别系统

#### 模型信息
- 架构: MobileNetV2
- 类别数: 4类 (可回收物/厨余垃圾/其他垃圾/有害垃圾)
- 输入尺寸: 224x224 RGB
- 模型文件: trash_classifier.pth

#### 兼容性处理
PyTorch不可用时自动使用模拟识别，返回随机分类结果+高置信度。

---

### 3. 南方气候适配

#### 珠海气候特点
- 年平均气温: 22-23°C
- 相对湿度: 70%-90%
- 台风季节: 6-10月
- 特殊问题: 厨余易发霉变质

#### 专属建议
| 垃圾类型 | 处理建议 |
|----------|----------|
| 厨余垃圾 | 沥干水分→报纸包裹→每日清理→小苏打除味 |
| 可回收物 | 清空内容物→简单清洗→压扁整理 |
| 有害垃圾 | 单独存放→定期送至回收点 |
| 其他垃圾 | 密封包装→避免液体渗漏 |

---

### 4. 积分系统

#### 功能模块
| 模块 | 说明 |
|------|------|
| 等级体系 | 7个等级：环保新手→环保入门→环保达人→环保专家→环保大师→环保宗师→环保传奇 |
| 经验值 | 根据总积分自动计算等级，显示进度条 |
| 每日打卡 | 每日签到获得积分奖励，记录连续打卡天数 |
| 成就徽章 | 18枚徽章，按稀有度分为common/rare/epic/legendary四档 |
| 积分记录 | 详细记录每次积分变动来源和数量 |

#### 积分获取方式
| 行为 | 积分 |
|------|------|
| 图像识别 | +10 |
| 问答互动 | +5 |
| 每日打卡 | +20 |
| 徽章解锁 | +50 |

#### 徽章稀有度配色
| 稀有度 | 颜色 | 示例 |
|--------|------|------|
| common (普通) | #9E9E9E 灰色 | 初次识别、初次问答 |
| rare (稀有) | #2196F3 蓝色 | 连续打卡7天、识别50次 |
| epic (史诗) | #9C27B0 紫色 | 连续打卡30天、识别200次 |
| legendary (传说) | #FFD700 金色 | 连续打卡100天、识别500次 |

---

### 5. 数据统计面板

#### 统计维度
| 维度 | 说明 |
|------|------|
| 总览统计 | 总投放次数、总积分、连续活跃天数、环保等级 |
| 分类占比 | 四类垃圾投放比例环形图 |
| 趋势分析 | 近30天每日投放量折线图 |
| 贡献分析 | 各行为(识别/问答/打卡)贡献占比 |
| 热力图 | 每周各时段投放频率矩阵 |
| 对比分析 | 本周 vs 上周数据对比 |

---

### 6. 设备管理系统

#### 功能
- 设备绑定/解绑
- 设备信息编辑
- 实时状态监控（在线/离线/满载/异常）
- 垃圾桶满载预警
- 滤芯更换提醒
- 一键更换滤芯操作

#### 设备状态类型
- online (在线)
- offline (离线)
- full (已满)
- error (异常)

---

## UI设计规范

### 设计风格
- 主色调: 生态绿 (#2E7D32) + 科技蓝 (#0288D1)
- AI主题色: 紫色渐变 (#9C27B0 -> #673AB7)
- 积分主题色: 紫粉渐变 (#9C27B0 / #E91E63)
- 风格: 智能家居简约科技风 + 玻璃态效果
- 目标用户: 家庭用户（含长辈）

### 核心视觉元素
- 渐变背景 (linear-gradient)
- 玻璃态效果 (backdrop-filter: blur)
- 发光阴影 (box-shadow glow)
- 动态动画 (transition + keyframes)
- 圆角卡片 (border-radius: 12-24px)

### AI状态指示器
| 状态 | 视觉效果 |
|------|----------|
| 可用 | 紫色发光边框 + 呼吸动画 + 大脑图标 + AI徽章 |
| 不可用 | 灰色普通样式 + 橙色警告文字 |
| 检测中 | 加载动画 + "正在连接..." |

---

## API接口文档

### 认证相关
| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/register` | 用户注册 |
| POST | `/api/login` | 用户登录 |
| POST | `/api/logout` | 用户登出 |
| GET | `/api/user` | 获取当前用户信息 |
| POST | `/api/password/change` | 修改密码 |

### 核心功能
| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/recognize` | AI图像识别 |
| POST | `/api/qa` | 智能问答 |
| GET | `/api/ai_status` | AI服务状态检测 |
| GET | `/api/south_tips` | 南方气候小贴士 |
| GET | `/api/south_guide` | 南方气候指南详情 |
| GET | `/api/category_detail/<category>` | 垃圾分类详情 |

### 积分系统
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/points` | 获取用户积分概览 |
| POST | `/api/points/checkin` | 每日签到打卡 |
| GET | `/api/points/records` | 积分变动记录 |
| GET | `/api/badges` | 成就徽章列表及解锁状态 |

### 数据统计
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/stats/overview` | 总览统计数据 |
| GET | `/api/stats/category` | 分类占比数据 |
| GET | `/api/stats/trend` | 趋势分析数据 |
| GET | `/api/stats/contribution` | 贡献分析数据 |
| GET | `/api/stats/heatmap` | 热力图数据 |
| GET | `/api/stats/compare` | 对比分析数据 |

### 设备管理
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/devices` | 设备列表 |
| POST | `/api/devices/add` | 添加设备 |
| POST | `/api/devices/update/<id>` | 编辑设备 |
| POST | `/api/devices/delete/<id>` | 删除设备 |
| GET | `/api/bin_status/<id>` | 垃圾桶状态 |
| POST | `/api/bin_status/update` | 更新状态 |
| GET | `/api/bin_alerts` | 满载预警列表 |
| GET | `/api/filter_alerts` | 滤芯提醒列表 |
| POST | `/api/filter/change` | 更换滤芯 |

### 其他
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/history` | 投放历史记录 |
| GET | `/api/notifications` | 通知消息列表 |
| POST | `/api/notifications/read/<id>` | 标记已读 |
| GET | `/api/notifications/unread_count` | 未读数量 |
| POST | `/api/notifications/add` | 添加通知 |
| GET | `/api/leaderboard` | 排行榜数据 |
| GET | `/api/eco_ranking` | 环保排名 |

---

## 数据库设计

### 表结构 (14张表)

```sql
-- 用户表
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 投放记录表
CREATE TABLE records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    item_name TEXT NOT NULL,
    category TEXT NOT NULL,
    image_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- 设备表
CREATE TABLE devices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    device_name TEXT NOT NULL,
    device_type TEXT,
    status TEXT DEFAULT 'online',
    last_active TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- 通知消息表
CREATE TABLE notifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    title TEXT NOT NULL,
    content TEXT,
    type TEXT,
    is_read INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- 垃圾桶状态表
CREATE TABLE bin_status (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id INTEGER,
    fill_level REAL DEFAULT 0,
    status TEXT DEFAULT 'normal',
    last_emptied TIMESTAMP,
    filter_status TEXT DEFAULT 'good',
    last_filter_change TIMESTAMP,
    FOREIGN KEY (device_id) REFERENCES devices(id)
);

-- 用户积分表
CREATE TABLE user_points (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER UNIQUE,
    total_points INTEGER DEFAULT 0,
    level INTEGER DEFAULT 1,
    exp_progress INTEGER DEFAULT 0,
    recognition_count INTEGER DEFAULT 0,
    question_count INTEGER DEFAULT 0,
    checkin_days INTEGER DEFAULT 0,
    continuous_checkin INTEGER DEFAULT 0,
    last_checkin DATE,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- 积分记录表
CREATE TABLE point_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    points INTEGER NOT NULL,
    source TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- 徽章定义表
CREATE TABLE badges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    icon TEXT,
    rarity TEXT DEFAULT 'common',
    condition_type TEXT,
    condition_value INTEGER DEFAULT 0
);

-- 用户徽章表
CREATE TABLE user_badges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    badge_id INTEGER,
    unlocked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, badge_id),
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (badge_id) REFERENCES badges(id)
);

-- 排行榜表
CREATE TABLE leaderboard (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER UNIQUE,
    total_points INTEGER DEFAULT 0,
    eco_level INTEGER DEFAULT 1,
    rank_position INTEGER,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- 每日统计表
CREATE TABLE daily_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    date DATE NOT NULL,
    recognition_count INTEGER DEFAULT 0,
    question_count INTEGER DEFAULT 0,
    checkin_done INTEGER DEFAULT 0,
    points_earned INTEGER DEFAULT 0,
    PRIMARY KEY (user_id, date),
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- 社区统计表
CREATE TABLE community_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE NOT NULL,
    total_users INTEGER DEFAULT 0,
    total_recognitions INTEGER DEFAULT 0,
    total_questions INTEGER DEFAULT 0,
    total_checkins INTEGER DEFAULT 0,
    total_points INTEGER DEFAULT 0,
    avg_per_user REAL DEFAULT 0,
    PRIMARY KEY (date)
);

-- 社区表
CREATE TABLE communities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    city TEXT DEFAULT '珠海',
    member_count INTEGER DEFAULT 0,
    total_points INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 社区成员表
CREATE TABLE community_members (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    community_id INTEGER NOT NULL,
    joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_admin INTEGER DEFAULT 0,
    FOREIGN KEY (user_id) REFERENCES users (id),
    FOREIGN KEY (community_id) REFERENCES communities (id),
    UNIQUE(user_id, community_id)
);
```

---

## 部署指南

### 环境要求
- Python 3.8+
- Flask 2.x
- SQLite3 (Python内置)
- PyTorch (可选，用于图像识别)
- 浏览器支持Web Speech API

### 启动步骤
```bash
# 1. 进入项目目录
cd c:\Users\31037\Desktop\python project\垃圾分类

# 2. 启动本地大模型 (可选但推荐)
# 在LM Studio或Ollama中启动 qwen3-0.6b
# 默认地址: http://127.0.0.1:1234

# 3. 启动Flask服务
python app_minimal.py

# 4. 打开浏览器访问
# http://127.0.0.1:5000
```

### 配置说明
```python
# app_minimal.py 中的关键配置

# AI大模型配置
AI_CONFIG = {
    'base_url': 'http://127.0.0.1:1234/v1/',
    'model': 'qwen3-0.6b',
    'timeout': 10  # 秒
}

# 模型路径
MODEL_PATH = 'trash_classifier.pth'

# 上传目录
UPLOAD_FOLDER = 'uploads'
```

---

## 项目亮点

### 1. 智能回退机制
- AI失败->规则->重试AI->默认提示
- 确保用户总能得到有用答案

### 2. 完整积分生态
- 7级等级体系 + 经验值进度条
- 18枚成就徽章 (4档稀有度)
- 每日打卡 + 连续签到激励
- 多渠道积分获取

### 3. 数据可视化统计
- 6大统计维度全覆盖
- 环形图/折线图/热力图/柱状图
- 周同比数据分析

### 4. 设备智能管理
- 实时状态监控
- 满载/滤芯双重预警
- 设备CRUD完整操作

### 5. 终端日志系统
- 详细记录每次AI调用的完整过程
- 包含问题、配置、耗时、回答等所有细节

### 6. 语音体验优化
- 自动清理Markdown格式符号
- 音量控制并保存设置
- speechSynthesis.cancel()重新播放

### 7. 南方气候深度适配
- 珠海专属垃圾分类建议
- 湿热气候特殊处理方法
- 季节性提示和预警

---

## 已解决的问题

| # | 问题 | 解决方案 | 状态 |
|---|------|----------|------|
| 1 | 置信度一直为0 | PyTorch兼容性检查+模拟识别 | 已解决 |
| 2 | 语音播报无法调节音量 | 音量控制滑块+localStorage | 已解决 |
| 3 | 调节音量后声音不变 | speechSynthesis.cancel()重新播放 | 已解决 |
| 4 | 调节音量后从头播放 | currentSpeechText保存当前文本 | 已解决 |
| 5 | 大模型未检测到 | 修正模型名+增加超时时间 | 已解决 |
| 6 | 终端编码错误(GBK) | 移除emoji字符 | 已解决 |
| 7 | 终端看不到AI回答 | 详细日志输出系统 | 已解决 |
| 8 | 多分类答案提取困难 | 直接返回完整回答 | 已解决 |
| 9 | 规则库找不到答案 | 扩展关键词库+AI补充机制 | 已解决 |
| 10 | AI状态不明显 | 紫色发光UI+动态效果 | 已解决 |
| 11 | 语音读出格式符号 | cleanTextForSpeech()清理函数 | 已解决 |
| 12 | ES6语法兼容性问题 | 全面改写为ES5语法(var/function/字符串拼接) | 已解决 |

---

## 后续优化方向

### 功能增强
- [ ] 流式输出AI回答（实时显示生成过程）
- [ ] 对话历史记录和上下文理解
- [ ] 多轮对话支持
- [ ] 用户反馈学习机制
- [ ] 社区互动功能增强

### 性能优化
- [ ] Redis缓存热门问答
- [ ] 数据库查询优化
- [ ] 静态资源CDN加速
- [ ] 图片压缩和处理优化

### 用户体验
- [ ] PWA离线支持
- [ ] 暗黑模式切换
- [ ] 多语言支持（粤语）
- [ ] 无障碍访问优化

### 安全加固
- [ ] JWT Token认证
- [ ] 接口限流防刷
- [ ] XSS/CSRF防护
- [ ] 敏感数据加密

---

## 参考资料

### 分类标准依据
- 《生活垃圾分类标志》(GB/T 19095-2019)
- 广东省城市生活垃圾分类指引
- 珠海市生活垃圾管理条例

### 相关技术文档
- [Flask官方文档](https://flask.palletsprojects.com/)
- [PyTorch文档](https://pytorch.org/docs/)
- [Web Speech API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Speech_API)
- [OpenAI API参考](https://platform.openai.com/docs/api-reference)

---

## 许可证

本项目仅供学习和演示使用。

---

**最后更新**: 2026年06月04日
**版本**: v2.0.0
**状态**: 生产就绪

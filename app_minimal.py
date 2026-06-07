import os
import sqlite3
import random

try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    from PIL import Image
    TORCH_AVAILABLE = True
    print("[OK] PyTorch已安装，将使用AI模型进行识别")
except ImportError as e:
    TORCH_AVAILABLE = False
    print("[WARN] PyTorch未安装 ({0})，将使用模拟识别".format(e))
    torch = None
    nn = None
    models = None
    transforms = None
    Image = None

try:
    import openai
    import requests
    AI_AVAILABLE = True
except ImportError as e:
    AI_AVAILABLE = False
    print("[WARN] AI库未安装 ({0})，智能问答将使用规则匹配模式".format(e))

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import json

app = Flask(__name__)
app.secret_key = 'zhuhai_smart_trash_2026'

AI_CONFIG = {
    'base_url': 'http://192.168.165.241:3000/v1',
    'api_key': 'Empty',
    'model': 'qwen3-0.6b'
}

ai_client = None
ai_available_status = None

def check_ai_available(force_check=False):
    global ai_client, ai_available_status
    
    if not force_check and ai_available_status is not None:
        return ai_available_status
    
    if not AI_AVAILABLE:
        ai_available_status = False
        print("[WARN] AI库未安装，无法使用大模型")
        return False
    
    try:
        import openai
        test_client = openai.OpenAI(
            base_url=AI_CONFIG['base_url'],
            api_key=AI_CONFIG['api_key']
        )
        
        print("[INFO] 正在检测本地大模型: {0}".format(AI_CONFIG['base_url']))
        
        response = requests.get("{0}models".format(AI_CONFIG['base_url']), timeout=5)
        print("[INFO] 大模型API响应状态: {0}".format(response.status_code))
        
        if response.status_code == 200:
            models_data = response.json()
            available_models = [m.get('id', '') for m in models_data.get('data', [])]
            print("[INFO] 可用模型列表: {0}...".format(available_models[:5]))
            
            ai_client = test_client
            ai_available_status = True
            print("[OK] 本地大模型已连接，智能问答将使用AI模式")
            return True
        else:
            ai_available_status = False
            print("[WARN] 本地大模型服务响应异常 (HTTP {0})".format(response.status_code))
            return False
            
    except requests.exceptions.ConnectTimeout:
        ai_available_status = False
        print("[WARN] 连接本地大模型超时（5秒），请确认服务已启动")
        return False
    except requests.exceptions.ConnectionError as e:
        ai_available_status = False
        print("[WARN] 无法连接到本地大模型 ({0})".format(e))
        print("[TIP] 请确保 LM Studio 或 Ollama 等服务正在运行")
        return False
    except Exception as e:
        ai_available_status = False
        print("[ERROR] 检测大模型时发生错误: {0}".format(e))
        import traceback
        traceback.print_exc()
        return False

def ask_ai(question):
    if not check_ai_available():
        print("\n{0}".format('█'*70))
        print("█  [AI问答] ⚠️  大模型不可用，无法回答问题")
        print("█  [AI问答] 问题: {0}".format(question))
        print("{0}\n".format('█'*70))
        return None
    
    try:
        print("\n{0}".format('█'*70))
        print("█  🤖  AI智能问答 - 开始处理")
        print("█{0}█".format('─'*68))
        print("█  👤 用户问题:")
        print("█     \"{0}\"".format(question))
        print("█{0}█".format('─'*68))
        print("█  🔧 模型配置:")
        print("█     • 模型名称: {0}".format(AI_CONFIG['model']))
        print("█     • API地址: {0}".format(AI_CONFIG['base_url']))
        print("█     • 最大token: 800")
        print("█     • 温度参数: 0.8")
        print("█{0}█".format('─'*68))
        print("█  📝 发送提示词 (System Prompt):")
        print("█  ┌──────────────────────────────────────────────────────┐")
        
        system_prompt = """你是"珠海智能垃圾分类管家"，一个专业的垃圾分类AI助手。你的服务对象主要是珠海家庭用户，需要考虑南方的湿热气候特点。

## 你的身份设定
- 名称：珠海智能垃圾分类管家
- 特点：专业、亲切、实用、懂珠海气候
- 风格：简洁明了，适合长辈理解

## 回答格式要求（必须严格遵循）

请按以下结构回答：

**【分类结果】**
明确说明属于哪一类垃圾（可回收物/厨余垃圾/其他垃圾/有害垃圾）

**【详细解释】**
1-2句话简单解释为什么这样分类

**【正确投放方法】**
具体的投放步骤和建议（2-3条）

**【珠海气候专属提醒】**
根据珠海湿热气候特点给出特别提醒（防潮、除湿、除味等）

**【常见误区】**
指出人们容易犯的错误（1-2条）

## 知识库参考

### 四大分类标准
- **可回收物（蓝色桶）**：纸张、塑料、玻璃、金属、织物等可循环利用的废弃物
- **厨余垃圾（绿色桶）**：剩菜剩饭、果皮果核、菜叶茶渣等易腐烂的有机废弃物
- **其他垃圾（灰色桶）**：除上述三类外的其他生活废弃物
- **有害垃圾（红色桶）**：对人体健康或自然环境造成危害的废弃物

### 珠海气候特点
- 年平均气温22-23°C，夏季高温多湿
- 相对湿度常年较高（70%-90%）
- 台风季节（6-10月）雨水较多
- 厨余垃圾容易发霉变质滋生细菌蟑螂

### 特殊处理建议
- **厨余垃圾**：沥干水分→用报纸包裹→每日清理→小苏打除味
- **可回收物**：清空内容物→简单清洗→压扁整理→投入蓝桶
- **有害垃圾**：单独存放→不要混入其他垃圾→定期送至指定回收点
- **其他垃圾**：密封包装→避免液体渗漏→及时投放"""
        
        for line in system_prompt.split('\n')[:15]:
            if len(line) > 54:
                print("█  │ {0}...".format(line[:54]))
            else:
                print("█  │ {0}".format(line))
        if len(system_prompt.split('\n')) > 15:
            print("█  │ ... (共{0}行)".format(len(system_prompt.split('\n'))))
        print("█  └──────────────────────────────────────────────────────┘")
        print("█{0}█".format('─'*68))
        print("█  ⏳ 正在调用大模型... 请稍候...")
        print("█{0}█".format('─'*68))

        import time
        start_time = time.time()
        
        completion = ai_client.chat.completions.create(
            model=AI_CONFIG['model'],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            max_tokens=800,
            temperature=0.8
        )
        
        end_time = time.time()
        response_time = round((end_time - start_time) * 1000, 1)
        
        answer = completion.choices[0].message.content
        
        print("█  ✅ 大模型响应完成! 耗时: {0}ms".format(response_time))
        print("█{0}█".format('─'*68))
        print("█  💬 大模型原始回答:")
        print("█  ┌──────────────────────────────────────────────────────┐")
        for i, line in enumerate(answer.split('\n'), 1):
            if line.strip():
                display_line = line[:56] + '...' if len(line) > 58 else line
                print("█  │ {0}".format(display_line))
        print("█  └──────────────────────────────────────────────────────┘")
        print("█{0}█".format('─'*68))
        
        print("█  ✅ AI模式: 直接使用大模型完整回答")
        print("█  📦 准备返回给前端的数据:")
        print("█     ├─ answer: {0}字符 (AI完整生成)".format(len(answer)))
        print("█     ├─ category: None (由前端解析显示)")
        print("█     ├─ tip: (无，答案中已包含)")
        print("█     ├─ south_tip: (无，答案中已包含)")
        print("█     └─ source: ai (大模型生成)")
        print("█{0}█".format('─'*68))
        print("█  ✨ AI问答处理完成!")
        print("{0}\n".format('█'*70))
        
        result = {
            'answer': answer,
            'category': None,
            'tip': '',
            'south_tip': '',
            'source': 'ai'
        }
        
        return result
        
    except Exception as e:
        print("\n{0}".format('█'*70))
        print("█  ❌ [ERROR] 大模型调用失败!")
        print("█{0}█".format('─'*68))
        print("█  问题: {0}".format(question))
        print("█  错误类型: {0}".format(type(e).__name__))
        print("█  错误信息: {0}".format(e))
        import traceback
        traceback.print_exc()
        print("{0}\n".format('█'*70))
        return None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'instance', 'trash.db')
MODEL_PATH = os.path.join(BASE_DIR, 'trash_classifier.pth')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

CATEGORY_MAPPING = {
    0: ('其他垃圾', 'dry', '#9E9E9E', '请投入灰色其他垃圾桶'),
    1: ('可回收物', 'recyclable', '#2196F3', '请投入蓝色可回收物桶'),
    2: ('厨余垃圾', 'wet', '#4CAF50', '请投入绿色厨余垃圾桶'),
    3: ('有害垃圾', 'hazardous', '#F44336', '请投入红色有害垃圾桶'),
}

CLASS_NAMES = [
    '其他垃圾', '可回收物', '厨余垃圾', '有害垃圾'
]

ITEM_TIPS = {
    '其他垃圾': '难以回收利用，请投入灰色其他垃圾桶',
    '可回收物': '可循环利用，请清空内容物后投入蓝色可回收物桶',
    '厨余垃圾': '易腐烂，请沥干水分后投入绿色厨余垃圾桶',
    '有害垃圾': '对人体或环境有害，请小心投入红色有害垃圾桶',
}

SOUTH_TIPS = {
    '厨余垃圾': '【珠海湿热气候专属】厨余垃圾请务必沥干水分！建议：①用报纸/厨房纸包裹后再丢弃 ②每日清理，避免过夜 ③垃圾桶内可放小苏打除味 ④夏季建议早晚各清理一次。防止异味和滋生细菌蟑螂！',
    '其他垃圾': '【珠海湿热气候专属】潮湿天气请保持干燥！建议：①尿不湿等卫生用品密封后再丢 ②烟蒂请完全熄灭 ③避免混入厨余垃圾污染。潮湿环境容易发霉发臭，请及时清理！',
    '可回收物': '【珠海湿热气候专属】南方湿度大，请保持干燥！建议：①塑料瓶倒空晾干 ②纸类避免受潮 ③金属避免生锈 ④雨天暂存室内。发霉的纸张和塑料无法回收，造成资源浪费！',
    '有害垃圾': '【珠海湿热气候专属】潮湿环境危险！建议：①电池密封后投放 ②药品原包装丢弃 ③灯管小心防碎 ④尽快投放勿久存。潮湿可能导致电池漏液、药品变质，危害家人健康！',
}

SOUTH_WEATHER_TIPS = [
    {
        'title': '🌡️ 湿热气候提醒',
        'content': '珠海属南亚热带季风气候，年均湿度75%以上。厨余垃圾极易腐烂发臭，建议每日清理2次（早晚各一次），沥干水分后投放。',
        'icon': 'fa-temperature-high',
        'color': '#FF9800',
        'actions': ['每日清理厨余垃圾', '沥干水分再投放', '使用密封垃圾袋']
    },
    {
        'title': '💧 防潮除湿要点',
        'content': '潮湿天气（湿度>80%）时，可回收物请暂存室内干燥处，避免发霉。纸张、纸箱受潮后无法回收，造成资源浪费。',
        'icon': 'fa-tint',
        'color': '#2196F3',
        'actions': ['可回收物保持干燥', '雨天暂存室内', '定期检查存放区']
    },
    {
        'title': '🦠 防臭除味技巧',
        'content': '厨余垃圾桶建议：①使用带盖垃圾桶 ②底部铺报纸吸水 ③撒小苏打除味 ④每周清洗一次桶内。防止异味和滋生蟑螂蚊虫！',
        'icon': 'fa-biohazard',
        'color': '#4CAF50',
        'actions': ['使用带盖垃圾桶', '撒小苏打除味', '每周清洗消毒']
    },
    {
        'title': '📦 密封投放建议',
        'content': '有害垃圾（电池、药品等）请密封后投放，防止潮湿环境导致漏液、变质。建议使用原包装或密封袋包装。',
        'icon': 'fa-box',
        'color': '#F44336',
        'actions': ['电池密封包装', '药品原包装丢弃', '尽快投放勿久存']
    },
    {
        'title': '🪳 防虫防鼠贴士',
        'content': '南方湿热易滋生蟑螂、蚊虫、老鼠。建议：①厨余垃圾不过夜 ②垃圾桶加盖 ③投放后清洗桶边 ④定期消毒垃圾桶周边。',
        'icon': 'fa-bug',
        'color': '#9C27B0',
        'actions': ['厨余垃圾不过夜', '垃圾桶加盖', '定期消毒清洁']
    },
    {
        'title': '☔ 雨天特别提醒',
        'content': '雨天投放垃圾注意：①穿防滑鞋 ②垃圾袋密封防漏 ③可回收物暂存室内 ④避免垃圾被雨水淋湿污染环境。',
        'icon': 'fa-cloud-rain',
        'color': '#00BCD4',
        'actions': ['穿防滑鞋出行', '垃圾袋密封', '可回收物暂存']
    }
]

SOUTH_SEASONAL_TIPS = {
    'spring': {
        'title': '🌸 春季（3-5月）',
        'content': '回南天高湿季节，墙壁地面易出水珠。建议：①厨余垃圾每日清理2次 ②室内使用除湿机 ③垃圾桶远离墙壁放置 ④可回收物密封保存。',
        'tips': ['每日清理厨余垃圾', '使用除湿机', '垃圾桶远离墙壁']
    },
    'summer': {
        'title': '☀️ 夏季（6-8月）',
        'content': '高温高湿，垃圾极易腐烂发臭。建议：①厨余垃圾早晚各清理一次 ②使用冰块保鲜厨余垃圾 ③垃圾桶放阴凉处 ④增加清洗频率。',
        'tips': ['早晚各清理一次', '垃圾桶放阴凉处', '增加清洗频率']
    },
    'autumn': {
        'title': '🍂 秋季（9-11月）',
        'content': '相对干燥舒适，但仍需注意。建议：①厨余垃圾每日清理 ②可回收物可适当囤积 ③利用好天气清洗垃圾桶 ④检查滤芯更换。',
        'tips': ['每日清理厨余垃圾', '清洗垃圾桶', '检查滤芯更换']
    },
    'winter': {
        'title': '❄️ 冬季（12-2月）',
        'content': '相对干燥凉爽，垃圾不易腐烂。建议：①厨余垃圾可隔日清理 ②可回收物囤积待回收 ③年底大扫除清理积存垃圾 ④注意防鼠。',
        'tips': ['可隔日清理厨余', '年底大扫除', '注意防鼠']
    }
}

QA_DATABASE = {
    '苹果核': {'category': '厨余垃圾', 'answer': '苹果核属于厨余垃圾（湿垃圾），请投入绿色厨余垃圾桶。'},
    '香蕉皮': {'category': '厨余垃圾', 'answer': '香蕉皮属于厨余垃圾（湿垃圾），请投入绿色厨余垃圾桶。'},
    '奶茶杯': {'category': '其他垃圾', 'answer': '奶茶杯属于其他垃圾。注意：喝剩的奶茶要倒进下水道，珍珠等固体残渣属于厨余垃圾，杯子本身属于其他垃圾。'},
    '纸巾': {'category': '其他垃圾', 'answer': '纸巾属于其他垃圾（干垃圾），即使湿了也是其他垃圾，不可回收。'},
    '电池': {'category': '有害垃圾', 'answer': '电池属于有害垃圾，含有重金属，请投入红色有害垃圾桶。'},
    '塑料袋': {'category': '其他垃圾', 'answer': '普通塑料袋属于其他垃圾，难以回收利用。但干净的厚塑料袋可以回收。'},
    '鱼骨': {'category': '厨余垃圾', 'answer': '鱼骨属于厨余垃圾（湿垃圾），请投入绿色厨余垃圾桶。'},
    '肉骨': {'category': '厨余垃圾', 'answer': '肉骨属于厨余垃圾（湿垃圾），请投入绿色厨余垃圾桶。'},
    '排骨': {'category': '厨余垃圾', 'answer': '排骨属于厨余垃圾（湿垃圾），请投入绿色厨余垃圾桶。'},
    '骨头': {'category': '厨余垃圾', 'answer': '骨头属于厨余垃圾（湿垃圾）。注意：大骨头（如猪腿骨、牛骨）因难以粉碎处理，应作为其他垃圾投放；小骨头（如鱼骨、鸡骨、排骨）属于厨余垃圾。'},
    '菜叶': {'category': '厨余垃圾', 'answer': '菜叶属于厨余垃圾（湿垃圾），请投入绿色厨余垃圾桶。'},
    '剩饭': {'category': '厨余垃圾', 'answer': '剩饭属于厨余垃圾（湿垃圾），请投入绿色厨余垃圾桶。'},
    '剩菜': {'category': '厨余垃圾', 'answer': '剩菜属于厨余垃圾（湿垃圾），请沥干水分后投入绿色厨余垃圾桶。'},
    '果皮': {'category': '厨余垃圾', 'answer': '果皮属于厨余垃圾（湿垃圾），请投入绿色厨余垃圾桶。'},
    '蛋壳': {'category': '厨余垃圾', 'answer': '蛋壳属于厨余垃圾（湿垃圾），请投入绿色厨余垃圾桶。'},
    '茶叶渣': {'category': '厨余垃圾', 'answer': '茶叶渣属于厨余垃圾（湿垃圾），请沥干后投入绿色厨余垃圾桶。'},
    '咖啡渣': {'category': '厨余垃圾', 'answer': '咖啡渣属于厨余垃圾（湿垃圾），可堆肥处理。'},
    '玻璃瓶': {'category': '可回收物', 'answer': '玻璃瓶属于可回收物，请清空内容物后投入蓝色可回收物桶。'},
    '易拉罐': {'category': '可回收物', 'answer': '易拉罐属于可回收物，请压扁后投入蓝色可回收物桶。'},
    '报纸': {'category': '可回收物', 'answer': '报纸属于可回收物，请投入蓝色可回收物桶。'},
    '书本': {'category': '可回收物', 'answer': '书本属于可回收物，请投入蓝色可回收物桶。'},
    '衣服': {'category': '可回收物', 'answer': '旧衣物属于可回收物，请清洗干净后投入蓝色可回收物桶或捐赠。'},
    '灯管': {'category': '有害垃圾', 'answer': '灯管属于有害垃圾，含有汞，请小心投入红色有害垃圾桶。'},
    '药品': {'category': '有害垃圾', 'answer': '过期药品属于有害垃圾，请投入红色有害垃圾桶。'},
    '烟蒂': {'category': '其他垃圾', 'answer': '烟蒂属于其他垃圾（干垃圾），请投入灰色其他垃圾桶。'},
    '尿不湿': {'category': '其他垃圾', 'answer': '尿不湿属于其他垃圾（干垃圾），请投入灰色其他垃圾桶。'},
    '化妆品': {'category': '有害垃圾', 'answer': '化妆品属于有害垃圾，请投入红色有害垃圾桶。'},
    '口罩': {'category': '其他垃圾', 'answer': '使用过的口罩属于其他垃圾，请投入灰色其他垃圾桶。'},
    '外卖盒': {'category': '其他垃圾', 'answer': '外卖餐盒属于其他垃圾。注意：剩菜剩饭属于厨余垃圾，餐盒要清洗干净后投放。'},
    '鸡骨': {'category': '厨余垃圾', 'answer': '鸡骨属于厨余垃圾（湿垃圾），体积小易腐烂，请投入绿色厨余垃圾桶。'},
    '猪骨': {'category': '其他垃圾', 'answer': '大猪骨属于其他垃圾（干垃圾），因体积大难粉碎，请投入灰色其他垃圾桶。'},
    '牛骨': {'category': '其他垃圾', 'answer': '牛骨属于其他垃圾（干垃圾），因质地坚硬难处理，请投入灰色其他垃圾桶。'},
}

model = None
device = None

def load_model():
    global model, device
    
    if not TORCH_AVAILABLE:
        print("[WARN] PyTorch不可用，使用模拟识别模式")
        return
    
    if model is not None:
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if os.path.exists(MODEL_PATH):
        try:
            loaded_data = torch.load(MODEL_PATH, map_location=device)
            
            if isinstance(loaded_data, dict):
                model = models.mobilenet_v2(pretrained=False)
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)
                model.load_state_dict(loaded_data)
                print("[OK] 模型加载成功 (state_dict格式): {0}".format(MODEL_PATH))
            else:
                model = loaded_data
                print("[OK] 模型加载成功 (完整模型格式): {0}".format(MODEL_PATH))
            
            model.to(device)
            model.eval()
            
            test_input = torch.randn(1, 3, 224, 224).to(device)
            with torch.no_grad():
                test_output = model(test_input)
            print("[OK] 模型测试通过，输出维度: {0}".format(test_output.shape))
            
        except Exception as e:
            print("[ERROR] 模型加载失败: {0}".format(e))
            import traceback
            traceback.print_exc()
            model = None
    else:
        print("[WARN] 模型文件不存在: {0}".format(MODEL_PATH))
        model = None

def predict_image(image_path):
    if model is None or not TORCH_AVAILABLE:
        random_idx = random.randint(0, 3)
        category_name, _, color, tip = CATEGORY_MAPPING[random_idx]
        simulated_confidence = random.uniform(0.75, 0.95)
        return {
            'item_name': '模拟识别-{0}'.format(category_name),
            'category': category_name,
            'color': color,
            'tip': tip,
            'south_tip': SOUTH_TIPS.get(category_name, ''),
            'confidence': simulated_confidence
        }
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
            
        idx = predicted.item()
        category_name, _, color, tip = CATEGORY_MAPPING[idx]
        
        return {
            'item_name': CLASS_NAMES[idx],
            'category': category_name,
            'color': color,
            'tip': tip,
            'south_tip': SOUTH_TIPS.get(category_name, ''),
            'confidence': float(confidence.item())
        }
    except Exception as e:
        print("预测错误: {0}".format(e))
        random_idx = random.randint(0, 3)
        category_name, _, color, tip = CATEGORY_MAPPING[random_idx]
        return {
            'item_name': '未知物品',
            'category': category_name,
            'color': color,
            'tip': tip,
            'south_tip': SOUTH_TIPS.get(category_name, ''),
            'confidence': 0.0
        }

class PointsService:
    LEVEL_EXP_BASE = 100
    LEVEL_EXP_GROWTH = 1.5
    
    POINT_RULES = {
        'recognition': 10,
        'question': 5,
        'checkin': 20,
        'checkin_continuous': 30,
        'badge_unlock': 0,
        'course_complete': 50,
        'invite_friend': 100,
    }
    
    @staticmethod
    def get_or_create_user_points(user_id):
        conn = get_db()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM user_points WHERE user_id = ?', (user_id,))
        user_points = cursor.fetchone()
        
        if not user_points:
            cursor.execute('''
                INSERT INTO user_points (user_id, total_points, level, exp, continuous_days, 
                                        total_checkins, total_recognitions, total_questions)
                VALUES (?, 0, 1, 0, 0, 0, 0, 0)
            ''', (user_id,))
            conn.commit()
            cursor.execute('SELECT * FROM user_points WHERE user_id = ?', (user_id,))
            user_points = cursor.fetchone()
        
        conn.close()
        return dict(user_points) if user_points else None
    
    @staticmethod
    def add_points(user_id, action_type, points=None, desc=None, related_id=None):
        if points is None:
            points = PointsService.POINT_RULES.get(action_type, 0)
        
        if points <= 0:
            return None
        
        conn = get_db()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO point_records (user_id, points, action_type, action_desc, related_id)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, points, action_type, desc, related_id))
            
            user_points = PointsService.get_or_create_user_points(user_id)
            new_total = user_points['total_points'] + points
            new_exp = user_points['exp'] + points
            
            new_level = PointsService.calculate_level(new_exp)
            
            cursor.execute('''
                UPDATE user_points 
                SET total_points = ?, exp = ?, level = ?, updated_at = CURRENT_TIMESTAMP
                WHERE user_id = ?
            ''', (new_total, new_exp, new_level, user_id))
            
            if action_type == 'recognition':
                cursor.execute('''
                    UPDATE user_points SET total_recognitions = total_recognitions + 1 
                    WHERE user_id = ?
                ''', (user_id,))
            elif action_type == 'question':
                cursor.execute('''
                    UPDATE user_points SET total_questions = total_questions + 1 
                    WHERE user_id = ?
                ''', (user_id,))
            
            conn.commit()
            
            unlocked_badges = PointsService.check_and_unlock_badges(user_id)
            
            return {
                'points_added': points,
                'new_total': new_total,
                'new_level': new_level,
                'level_up': new_level > user_points['level'],
                'unlocked_badges': unlocked_badges
            }
            
        except Exception as e:
            print("[ERROR] 添加积分失败: {0}".format(e))
            conn.rollback()
            return None
        finally:
            conn.close()
    
    @staticmethod
    def calculate_level(exp):
        if exp < 100:
            return 1
        level = 1
        total_exp_needed = 0
        while True:
            exp_needed = int(PointsService.LEVEL_EXP_BASE * (PointsService.LEVEL_EXP_GROWTH ** (level - 1)))
            total_exp_needed += exp_needed
            if exp < total_exp_needed:
                break
            level += 1
        return level
    
    @staticmethod
    def get_exp_for_next_level(current_level):
        return int(PointsService.LEVEL_EXP_BASE * (PointsService.LEVEL_EXP_GROWTH ** (current_level - 1)))
    
    @staticmethod
    def checkin(user_id):
        conn = get_db()
        cursor = conn.cursor()
        
        try:
            user_points = PointsService.get_or_create_user_points(user_id)
            today = datetime.now().date()
            last_checkin = user_points.get('last_checkin_date')
            
            if last_checkin:
                last_checkin_date = datetime.strptime(last_checkin, '%Y-%m-%d').date()
                if last_checkin_date == today:
                    conn.close()
                    return {'success': False, 'message': '今日已打卡'}
                
                days_diff = (today - last_checkin_date).days
                if days_diff == 1:
                    new_continuous = user_points['continuous_days'] + 1
                    points = PointsService.POINT_RULES['checkin_continuous']
                    desc = '连续打卡{0}天'.format(new_continuous)
                else:
                    new_continuous = 1
                    points = PointsService.POINT_RULES['checkin']
                    desc = '打卡成功'
            else:
                new_continuous = 1
                points = PointsService.POINT_RULES['checkin']
                desc = '首次打卡'
            
            cursor.execute('''
                UPDATE user_points 
                SET continuous_days = ?, 
                    last_checkin_date = ?,
                    total_checkins = total_checkins + 1,
                    updated_at = CURRENT_TIMESTAMP
                WHERE user_id = ?
            ''', (new_continuous, today, user_id))
            
            conn.commit()
            conn.close()
            
            result = PointsService.add_points(user_id, 'checkin', points, desc)
            
            return {
                'success': True,
                'continuous_days': new_continuous,
                'points_added': points,
                'message': desc,
                'level_up': result.get('level_up', False) if result else False,
                'unlocked_badges': result.get('unlocked_badges', []) if result else []
            }
            
        except Exception as e:
            print("[ERROR] 打卡失败: {0}".format(e))
            conn.rollback()
            conn.close()
            return {'success': False, 'message': '打卡失败'}
    
    @staticmethod
    def check_and_unlock_badges(user_id):
        conn = get_db()
        cursor = conn.cursor()
        
        try:
            user_points = PointsService.get_or_create_user_points(user_id)
            
            cursor.execute('SELECT badge_id FROM user_badges WHERE user_id = ?', (user_id,))
            unlocked = [row['badge_id'] for row in cursor.fetchall()]
            
            cursor.execute('SELECT * FROM badges')
            all_badges = cursor.fetchall()
            
            newly_unlocked = []
            
            for badge in all_badges:
                badge_dict = dict(badge)
                badge_id = badge_dict['badge_id']
                
                if badge_id in unlocked:
                    continue
                
                req_type = badge_dict['requirement_type']
                req_value = badge_dict['requirement_value']
                
                user_value = 0
                if req_type == 'recognitions':
                    user_value = user_points['total_recognitions']
                elif req_type == 'questions':
                    user_value = user_points['total_questions']
                elif req_type == 'continuous_days':
                    user_value = user_points['continuous_days']
                elif req_type == 'level':
                    user_value = user_points['level']
                elif req_type == 'total_points':
                    user_value = user_points['total_points']
                elif req_type == 'checkins':
                    user_value = user_points['total_checkins']
                
                if user_value >= req_value:
                    cursor.execute('''
                        INSERT INTO user_badges (user_id, badge_id)
                        VALUES (?, ?)
                    ''', (user_id, badge_id))
                    
                    if badge_dict['points_reward'] > 0:
                        conn.commit()
                        conn.close()
                        PointsService.add_points(
                            user_id, 'badge_unlock', 
                            badge_dict['points_reward'],
                            '解锁徽章: {0}'.format(badge_dict["name"])
                        )
                        conn = get_db()
                        cursor = conn.cursor()
                    
                    newly_unlocked.append(badge_dict)
            
            conn.commit()
            conn.close()
            return newly_unlocked
            
        except Exception as e:
            print("[ERROR] 检查徽章失败: {0}".format(e))
            conn.close()
            return []
    
    @staticmethod
    def get_user_badges(user_id):
        conn = get_db()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT b.*, ub.unlocked_at
            FROM badges b
            LEFT JOIN user_badges ub ON b.badge_id = ub.badge_id AND ub.user_id = ?
            ORDER BY b.rarity, b.requirement_value
        ''', (user_id,))
        
        badges = []
        for row in cursor.fetchall():
            badge = dict(row)
            badge['unlocked'] = badge['unlocked_at'] is not None
            badges.append(badge)
        
        conn.close()
        return badges
    
    @staticmethod
    def get_point_records(user_id, limit=20):
        conn = get_db()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM point_records 
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        ''', (user_id, limit))
        
        records = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return records
    
    @staticmethod
    def get_leaderboard(period='weekly', limit=10):
        conn = get_db()
        cursor = conn.cursor()
        
        if period == 'weekly':
            date_filter = "datetime('now', '-7 days')"
        elif period == 'monthly':
            date_filter = "datetime('now', '-30 days')"
        else:
            date_filter = "datetime('now', '-100 years')"
        
        cursor.execute('''
            SELECT u.id, u.username, 
                   COALESCE(SUM(pr.points), 0) as period_points,
                   up.total_points, up.level, up.continuous_days
            FROM users u
            LEFT JOIN point_records pr ON u.id = pr.user_id 
                AND pr.created_at >= {date_filter}
            LEFT JOIN user_points up ON u.id = up.user_id
            GROUP BY u.id
            HAVING period_points > 0
            ORDER BY period_points DESC
            LIMIT ?
        ''', (limit,))
        
        leaderboard = []
        for rank, row in enumerate(cursor.fetchall(), 1):
            item = dict(row)
            item['rank'] = rank
            leaderboard.append(item)
        
        conn.close()
        return leaderboard

class StatsService:
    WEIGHT_ESTIMATES = {
        '可回收物': 0.3,
        '厨余垃圾': 0.5,
        '其他垃圾': 0.2,
        '有害垃圾': 0.1
    }
    
    CARBON_FACTORS = {
        '可回收物': 2.5,
        '厨余垃圾': 0.5,
        '其他垃圾': 0.1,
        '有害垃圾': 3.0
    }
    
    @staticmethod
    def get_category_stats(user_id, days=7):
        conn = get_db()
        cursor = conn.cursor()
        
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        cursor.execute('''
            SELECT category, COUNT(*) as count
            FROM records
            WHERE user_id = ? AND date(timestamp) >= ?
            GROUP BY category
        ''', (user_id, start_date))
        
        stats = {row['category']: row['count'] for row in cursor.fetchall()}
        conn.close()
        
        return {
            '可回收物': stats.get('可回收物', 0),
            '厨余垃圾': stats.get('厨余垃圾', 0),
            '其他垃圾': stats.get('其他垃圾', 0),
            '有害垃圾': stats.get('有害垃圾', 0)
        }
    
    @staticmethod
    def get_daily_trend(user_id, days=7):
        conn = get_db()
        cursor = conn.cursor()
        
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        cursor.execute('''
            SELECT date(timestamp) as date, COUNT(*) as count,
                   SUM(CASE WHEN category = '可回收物' THEN 1 ELSE 0 END) as recyclable
            FROM records
            WHERE user_id = ? AND date(timestamp) >= ?
            GROUP BY date(timestamp)
            ORDER BY date(timestamp)
        ''', (user_id, start_date))
        
        trend = []
        for row in cursor.fetchall():
            trend.append({
                'date': row['date'],
                'count': row['count'],
                'recyclable': row['recyclable'],
                'recyclable_rate': round(row['recyclable'] / row['count'] * 100, 1) if row['count'] > 0 else 0
            })
        
        conn.close()
        return trend
    
    @staticmethod
    def get_contribution(user_id, days=30):
        conn = get_db()
        cursor = conn.cursor()
        
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        cursor.execute('''
            SELECT category, COUNT(*) as count
            FROM records
            WHERE user_id = ? AND date(timestamp) >= ?
            GROUP BY category
        ''', (user_id, start_date))
        
        stats = {row['category']: row['count'] for row in cursor.fetchall()}
        conn.close()
        
        total_weight = 0
        carbon_saved = 0
        resource_recovered = 0
        
        for category, count in stats.items():
            weight = count * StatsService.WEIGHT_ESTIMATES.get(category, 0.2)
            total_weight += weight
            carbon_saved += weight * StatsService.CARBON_FACTORS.get(category, 0.1)
            
            if category == '可回收物':
                resource_recovered = weight * 0.8
        
        energy_saved = resource_recovered * 3.2
        landfill_reduced = total_weight * 0.9
        
        return {
            'total_weight': round(total_weight, 2),
            'carbon_saved': round(carbon_saved, 2),
            'resource_recovered': round(resource_recovered, 2),
            'energy_saved': round(energy_saved, 2),
            'landfill_reduced': round(landfill_reduced, 2),
            'trees_equivalent': round(carbon_saved / 20, 2)
        }
    
    @staticmethod
    def get_heatmap(user_id, days=30):
        conn = get_db()
        cursor = conn.cursor()
        
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        cursor.execute('''
            SELECT 
                strftime('%w', timestamp) as weekday,
                strftime('%H', timestamp) as hour,
                COUNT(*) as count
            FROM records
            WHERE user_id = ? AND date(timestamp) >= ?
            GROUP BY weekday, hour
        ''', (user_id, start_date))
        
        heatmap = {}
        for row in cursor.fetchall():
            weekday = int(row['weekday'])
            hour = int(row['hour'])
            key = "{0}_{1}".format(weekday, hour)
            heatmap[key] = row['count']
        
        conn.close()
        
        result = {
            'data': heatmap,
            'peak_hours': StatsService._find_peak_hours(heatmap),
            'peak_days': StatsService._find_peak_days(heatmap)
        }
        
        return result
    
    @staticmethod
    def _find_peak_hours(heatmap):
        hour_counts = {}
        for key, count in heatmap.items():
            hour = key.split('_')[1]
            hour_counts[hour] = hour_counts.get(hour, 0) + count
        
        sorted_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_hours[:3]
    
    @staticmethod
    def _find_peak_days(heatmap):
        day_counts = {}
        day_names = ['周日', '周一', '周二', '周三', '周四', '周五', '周六']
        for key, count in heatmap.items():
            day = key.split('_')[0]
            day_counts[day] = day_counts.get(day, 0) + count
        
        sorted_days = sorted(day_counts.items(), key=lambda x: x[1], reverse=True)
        return [(day_names[int(d)], c) for d, c in sorted_days[:3]]
    
    @staticmethod
    def get_community_compare(user_id):
        conn = get_db()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT COUNT(*) as total, 
                   AVG(daily_count) as avg_count
            FROM (
                SELECT user_id, COUNT(*) as daily_count
                FROM records
                WHERE date(timestamp) = date('now')
                GROUP BY user_id
            )
        ''')
        
        community = cursor.fetchone()
        
        cursor.execute('''
            SELECT COUNT(*) as count
            FROM records
            WHERE user_id = ? AND date(timestamp) = date('now')
        ''', (user_id,))
        
        user_today = cursor.fetchone()['count']
        
        cursor.execute('''
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN category = '可回收物' THEN 1 ELSE 0 END) as recyclable
            FROM records
            WHERE user_id = ?
        ''', (user_id,))
        
        user_total = cursor.fetchone()
        user_recyclable_rate = (user_total['recyclable'] / user_total['total'] * 100) if user_total['total'] > 0 else 0
        
        cursor.execute('''
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN category = '可回收物' THEN 1 ELSE 0 END) as recyclable
            FROM records
        ''')
        
        community_total = cursor.fetchone()
        community_recyclable_rate = (community_total['recyclable'] / community_total['total'] * 100) if community_total['total'] > 0 else 0
        
        conn.close()
        
        return {
            'user_daily_count': user_today,
            'community_avg_count': round(community['avg_count'] or 0, 1),
            'user_recyclable_rate': round(user_recyclable_rate, 1),
            'community_recyclable_rate': round(community_recyclable_rate, 1),
            'daily_diff': round((user_today - (community['avg_count'] or 0)) / (community['avg_count'] or 1) * 100, 1) if community['avg_count'] else 0,
            'recyclable_diff': round((user_recyclable_rate - community_recyclable_rate) / (community_recyclable_rate or 1) * 100, 1) if community_recyclable_rate else 0
        }
    
    @staticmethod
    def get_overview(user_id):
        conn = get_db()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT COUNT(*) as total,
                   MAX(timestamp) as last_record
            FROM records
            WHERE user_id = ?
        ''', (user_id,))
        
        basic = cursor.fetchone()
        
        cursor.execute('''
            SELECT category, COUNT(*) as count
            FROM records
            WHERE user_id = ?
            GROUP BY category
            ORDER BY count DESC
            LIMIT 1
        ''', (user_id,))
        
        top_category = cursor.fetchone()
        
        cursor.execute('''
            SELECT COUNT(*) as count
            FROM records
            WHERE user_id = ? AND date(timestamp) = date('now')
        ''', (user_id,))
        
        today_count = cursor.fetchone()['count']
        
        conn.close()
        
        return {
            'total_records': basic['total'],
            'last_record': basic['last_record'],
            'top_category': top_category['category'] if top_category else None,
            'today_count': today_count
        }

def init_badges(cursor):
    badges_data = [
        ('first_step', '环保新手', '完成第一次垃圾分类识别', '🌱', 'beginner', 'recognitions', 1, 10, 'common'),
        ('recognition_10', '识别达人', '累计完成10次垃圾识别', '🔍', 'recognition', 'recognitions', 10, 30, 'common'),
        ('recognition_50', '识别专家', '累计完成50次垃圾识别', '🔬', 'recognition', 'recognitions', 50, 100, 'rare'),
        ('recognition_100', '识别大师', '累计完成100次垃圾识别', '🏆', 'recognition', 'recognitions', 100, 200, 'epic'),
        ('qa_10', '问答新手', '累计提问10次', '💬', 'qa', 'questions', 10, 20, 'common'),
        ('qa_50', '问答达人', '累计提问50次', '🗣️', 'qa', 'questions', 50, 80, 'rare'),
        ('checkin_7', '坚持一周', '连续打卡7天', '📅', 'checkin', 'continuous_days', 7, 50, 'common'),
        ('checkin_30', '坚持一月', '连续打卡30天', '📆', 'checkin', 'continuous_days', 30, 150, 'rare'),
        ('checkin_100', '坚持百日', '连续打卡100天', '💯', 'checkin', 'continuous_days', 100, 500, 'epic'),
        ('level_5', '环保达人', '达到5级', '⭐', 'level', 'level', 5, 100, 'rare'),
        ('level_10', '环保专家', '达到10级', '🌟', 'level', 'level', 10, 300, 'epic'),
        ('level_20', '环保大师', '达到20级', '👑', 'level', 'level', 20, 800, 'legendary'),
        ('points_1000', '千分成就', '累计获得1000积分', '💎', 'points', 'total_points', 1000, 100, 'rare'),
        ('points_5000', '五千分成就', '累计获得5000积分', '💠', 'points', 'total_points', 5000, 300, 'epic'),
        ('points_10000', '万分成就', '累计获得10000积分', '🔮', 'points', 'total_points', 10000, 800, 'legendary'),
        ('recyclable_pro', '可回收专家', '识别可回收物20次', '♻️', 'category', 'recyclable_count', 20, 50, 'rare'),
        ('wet_pro', '厨余专家', '识别厨余垃圾20次', '🍂', 'category', 'wet_count', 20, 50, 'rare'),
        ('hazardous_pro', '有害专家', '识别有害垃圾10次', '⚠️', 'category', 'hazardous_count', 10, 80, 'epic'),
    ]
    
    for badge in badges_data:
        try:
            cursor.execute('''
                INSERT OR IGNORE INTO badges 
                (badge_id, name, description, icon, category, requirement_type, requirement_value, points_reward, rarity)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', badge)
        except Exception as e:
            print("[WARN] 初始化徽章失败 {0}: {1}".format(badge[0], e))

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            eco_score INTEGER DEFAULT 0
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            item_name TEXT,
            category TEXT,
            image_path TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS devices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            device_name TEXT,
            device_type TEXT,
            status TEXT DEFAULT 'online',
            last_check TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            title TEXT,
            content TEXT,
            type TEXT DEFAULT 'info',
            is_read INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS bin_status (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            device_id INTEGER,
            bin_type TEXT,
            fill_level INTEGER DEFAULT 0,
            filter_days INTEGER DEFAULT 30,
            last_filter_change TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (device_id) REFERENCES devices (id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_points (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER UNIQUE NOT NULL,
            total_points INTEGER DEFAULT 0,
            level INTEGER DEFAULT 1,
            exp INTEGER DEFAULT 0,
            continuous_days INTEGER DEFAULT 0,
            last_checkin_date DATE,
            total_checkins INTEGER DEFAULT 0,
            total_recognitions INTEGER DEFAULT 0,
            total_questions INTEGER DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS point_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            points INTEGER NOT NULL,
            action_type TEXT NOT NULL,
            action_desc TEXT,
            related_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS badges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            badge_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            description TEXT,
            icon TEXT,
            category TEXT,
            requirement_type TEXT,
            requirement_value INTEGER,
            points_reward INTEGER DEFAULT 0,
            rarity TEXT DEFAULT 'common',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_badges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            badge_id TEXT NOT NULL,
            unlocked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id),
            FOREIGN KEY (badge_id) REFERENCES badges (badge_id),
            UNIQUE(user_id, badge_id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS leaderboard (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            period TEXT NOT NULL,
            rank INTEGER,
            points INTEGER DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id),
            UNIQUE(user_id, period)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            stat_date DATE,
            total_count INTEGER DEFAULT 0,
            recyclable_count INTEGER DEFAULT 0,
            wet_count INTEGER DEFAULT 0,
            dry_count INTEGER DEFAULT 0,
            hazardous_count INTEGER DEFAULT 0,
            correct_count INTEGER DEFAULT 0,
            total_weight REAL DEFAULT 0,
            carbon_saved REAL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id),
            UNIQUE(user_id, stat_date)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS community_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stat_date DATE UNIQUE,
            total_users INTEGER DEFAULT 0,
            total_records INTEGER DEFAULT 0,
            avg_daily_count REAL DEFAULT 0,
            avg_accuracy REAL DEFAULT 0,
            total_carbon_saved REAL DEFAULT 0,
            recyclable_rate REAL DEFAULT 0,
            wet_rate REAL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS communities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            city TEXT DEFAULT '珠海',
            member_count INTEGER DEFAULT 0,
            total_points INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS community_members (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            community_id INTEGER NOT NULL,
            joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_admin INTEGER DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users (id),
            FOREIGN KEY (community_id) REFERENCES communities (id),
            UNIQUE(user_id, community_id)
        )
    ''')

    # 创建默认社区：珠海模拟社区
    cursor.execute(
        "INSERT OR IGNORE INTO communities (id, name, description, city) VALUES (1, '珠海模拟社区', '珠海市智能垃圾分类模拟社区，欢迎所有环保爱好者加入！', '珠海')"
    )

    init_badges(cursor)

    conn.commit()
    conn.close()

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def index():
    return render_template('index.html', active_page='home')

@app.route('/scan')
def scan():
    return render_template('scan.html', active_page='scan')

@app.route('/qa')
def qa():
    return render_template('qa.html', active_page='qa')

@app.route('/guide')
def guide():
    return render_template('guide.html', active_page='guide')

@app.route('/profile')
def profile():
    return render_template('profile.html', active_page='profile')

@app.route('/history')
def history():
    return render_template('history.html', active_page='history')

@app.route('/devices')
def devices():
    return render_template('devices.html', active_page='devices')

@app.route('/notifications')
def notifications():
    return render_template('notifications.html', active_page='notifications')

@app.route('/south_guide')
def south_guide():
    return render_template('south_guide.html', active_page='guide')

@app.route('/login')
def login_page():
    return render_template('login.html', active_page='login')

@app.route('/register')
def register_page():
    return render_template('register.html', active_page='register')

@app.route('/api/recognize', methods=['POST'])
def api_recognize():
    if 'image' not in request.files:
        return jsonify({'error': '没有上传图片'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': '没有选择图片'}), 400
    
    filename = "{0}_{1}.jpg".format(datetime.now().strftime('%Y%m%d%H%M%S'), random.randint(1000, 9999))
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    result = predict_image(filepath)
    
    if 'user_id' in session:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO records (user_id, item_name, category, image_path) VALUES (?, ?, ?, ?)',
            (session['user_id'], result['item_name'], result['category'], filepath)
        )
        cursor.execute(
            'UPDATE users SET eco_score = eco_score + 1 WHERE id = ?',
            (session['user_id'],)
        )
        conn.commit()
        conn.close()
        
        points_result = PointsService.add_points(
            session['user_id'], 
            'recognition', 
            desc='识别垃圾: {0} -> {1}'.format(result["item_name"], result["category"])
        )
        
        if points_result:
            result['points_added'] = points_result['points_added']
            result['level_up'] = points_result.get('level_up', False)
            if points_result.get('unlocked_badges'):
                result['new_badges'] = points_result['unlocked_badges']
    
    return jsonify(result)

@app.route('/api/qa', methods=['POST'])
def api_qa():
    data = request.get_json()
    question = data.get('question', '')
    use_ai = data.get('use_ai', False)
    
    print("\n{0}".format('█'*70))
    print("█  📥 [API请求] /api/qa")
    print("█{0}█".format('─'*68))
    print("█  问题: \"{0}\"".format(question))
    print("█  模式: {0}".format('🤖 AI大模式' if use_ai else '📖 规则匹配'))
    print("{0}".format('█'*70))
    
    result = None
    
    if use_ai:
        print("█  ➡️  转发至 AI 处理模块...")
        ai_result = ask_ai(question)
        if ai_result:
            print("{0}".format('█'*70))
            print("█  📤 [API响应] 返回AI完整回答给前端")
            print("     状态: ✅ 成功")
            print("     回答长度: {0}字符".format(len(ai_result['answer'])))
            print("     来源: {0}".format(ai_result['source']))
            print("{0}\n".format('█'*70))
            result = ai_result
        
        if not result:
            print("█  ⚠️  AI首次调用失败，回退到规则匹配...")
            rule_answer = _get_rule_answer(question)
            
            if rule_answer.get('category'):
                print("{0}".format('█'*70))
                print("█  📤 [API响应] 规则匹配成功 (AI回退)")
                print("     分类: {0}".format(rule_answer['category']))
                print("     来源: rule_fallback")
                print("{0}\n".format('█'*70))
                rule_answer['source'] = 'rule_fallback'
                rule_answer['answer'] = '⚠️ 大模型暂时不可用，已使用规则库回答。\n\n' + rule_answer['answer']
                result = rule_answer
            
            if not result:
                print("█  ⚠️  规则也未找到答案，再次尝试AI...")
                ai_result_retry = ask_ai(question)
                if ai_result_retry:
                    print("{0}".format('█'*70))
                    print("█  📤 [API响应] AI重试成功!")
                    print("     状态: ✅ 成功 (重试)")
                    print("     回答长度: {0}字符".format(len(ai_result_retry['answer'])))
                    print("     来源: ai_retry")
                    print("{0}\n".format('█'*70))
                    ai_result_retry['source'] = 'ai_retry'
                    result = ai_result_retry
                
                if not result:
                    print("{0}".format('█'*70))
                    print("█  📤 [API响应] 返回默认提示")
                    print("     状态: ⚠️ AI和规则均不可用")
                    print("{0}\n".format('█'*70))
                    result = {
                        'answer': '🤔 抱歉，我暂时无法回答这个问题。\n\n可能的原因：\n• 大模型服务未启动\n• 问题超出知识范围\n\n建议：\n1. 检查本地大模型是否运行在 http://127.0.0.1:1234\n2. 尝试更具体的问题描述\n3. 使用拍照识别功能\n4. 查看"分类指南"获取更多信息',
                        'category': None,
                        'tip': '',
                        'south_tip': '',
                        'source': 'unavailable'
                    }
    else:
        print("█  ➡️  使用规则库匹配...")
        rule_result = _get_rule_answer(question)
        
        if rule_result.get('category'):
            print("{0}".format('█'*70))
            print("█  📤 [API响应] 规则匹配成功")
            print("     状态: ✅ 成功")
            print("     分类: {0}".format(rule_result['category']))
            print("     来源: rule")
            print("{0}\n".format('█'*70))
            result = rule_result
        
        if not result:
            if not check_ai_available():
                print("{0}".format('█'*70))
                print("█  📤 [API响应] 规则未找到，AI不可用，返回默认答案")
                print("     状态: ⚠️ 仅规则可用")
                print("     分类: (未识别)")
                print("     来源: rule_fallback")
                print("{0}\n".format('█'*70))
                result = rule_result
            else:
                print("█  ℹ️  规则库未匹配到，尝试调用AI补充回答...")
                ai_result = ask_ai(question)
                if ai_result:
                    print("{0}".format('█'*70))
                    print("█  📤 [API响应] AI补充成功!")
                    print("     状态: ✅ 成功 (AI补充)")
                    print("     回答长度: {0}字符".format(len(ai_result['answer'])))
                    print("     来源: {0}".format(ai_result['source']))
                    print("{0}\n".format('█'*70))
                    result = ai_result
                else:
                    print("{0}".format('█'*70))
                    print("█  📤 [API响应] 返回规则默认答案")
                    print("     状态: ⚠️ AI也失败")
                    print("     分类: (未识别)")
                    print("     来源: rule")
                    print("{0}\n".format('█'*70))
                    result = rule_result
    
    if 'user_id' in session and result.get('source') != 'unavailable':
        points_result = PointsService.add_points(
            session['user_id'],
            'question',
            desc='智能问答: {0}...'.format(question[:30])
        )
        if points_result:
            result['points_added'] = points_result['points_added']
            result['level_up'] = points_result.get('level_up', False)
            if points_result.get('unlocked_badges'):
                result['new_badges'] = points_result['unlocked_badges']
    
    return jsonify(result)

def _get_rule_answer(question):
    for keyword, info in QA_DATABASE.items():
        if keyword in question:
            return {
                'answer': info['answer'],
                'category': info['category'],
                'tip': ITEM_TIPS.get(info['category'], ''),
                'south_tip': SOUTH_TIPS.get(info['category'], ''),
                'source': 'rule'
            }
    
    if '什么垃圾' in question or '怎么分' in question or '扔哪里' in question:
        return {
            'answer': '抱歉，这个问题我暂时无法回答。您可以尝试更具体的问题，比如"苹果核是什么垃圾？"',
            'category': None,
            'tip': '建议使用拍照识别功能，更准确地识别垃圾类别。',
            'source': 'rule'
        }
    
    return {
        'answer': '我理解您的问题了。建议您使用拍照识别功能，或者查看分类指南了解更多信息。',
        'category': None,
        'tip': '如有疑问，可以点击"分类指南"查看详细说明。',
        'source': 'rule'
    }

@app.route('/api/ai_status')
def api_ai_status():
    force_check = request.args.get('refresh', 'false').lower() == 'true'
    available = check_ai_available(force_check=force_check)
    
    if available:
        try:
            models_resp = requests.get("{0}models".format(AI_CONFIG['base_url']), timeout=3)
            if models_resp.status_code == 200:
                models_data = models_resp.json()
                model_list = [m.get('id', '') for m in models_data.get('data', [])]
                current_model = AI_CONFIG['model']
                model_found = any(current_model in m for m in model_list)
        except:
            model_list = []
            model_found = False
    else:
        model_list = []
        model_found = False
    
    return jsonify({
        'available': available,
        'mode': 'ai' if available else 'rule',
        'message': '本地大模型已连接（{0}）'.format(AI_CONFIG["model"]) if available else '本地大模型未连接，使用规则匹配模式',
        'model': AI_CONFIG['model'],
        'model_found': model_found,
        'base_url': AI_CONFIG['base_url'],
        'available_models': model_list[:10] if available else []
    })

@app.route('/api/register', methods=['POST'])
def api_register():
    data = request.get_json()
    username = data.get('username', '').strip()
    password = data.get('password', '')
    
    if not username or not password:
        return jsonify({'success': False, 'message': '用户名和密码不能为空'})
    
    conn = get_db()
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            'INSERT INTO users (username, password) VALUES (?, ?)',
            (username, generate_password_hash(password))
        )
        conn.commit()
        return jsonify({'success': True, 'message': '注册成功'})
    except sqlite3.IntegrityError:
        return jsonify({'success': False, 'message': '用户名已存在'})
    finally:
        conn.close()

@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json()
    username = data.get('username', '').strip()
    password = data.get('password', '')
    
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
    user = cursor.fetchone()
    conn.close()
    
    if user and check_password_hash(user['password'], password):
        session['user_id'] = user['id']
        session['username'] = user['username']
        
        cursor = get_db().cursor()
        cursor.execute('SELECT COUNT(*) as count FROM records WHERE user_id = ?', (user['id'],))
        total = cursor.fetchone()['count']
        
        cursor.execute('SELECT COUNT(*) as count FROM records WHERE user_id = ? AND timestamp >= ?', 
                      (user['id'], datetime.now() - timedelta(days=30)))
        month = cursor.fetchone()['count']
        
        return jsonify({
            'success': True,
            'stats': {
                'total': total,
                'month': month,
                'score': user['eco_score'],
                'accuracy': 95
            }
        })
    
    return jsonify({'success': False, 'message': '用户名或密码错误'})

@app.route('/api/logout', methods=['POST'])
def api_logout():
    session.clear()
    return jsonify({'success': True})

@app.route('/api/user')
def api_user():
    if 'user_id' not in session:
        return jsonify({'logged_in': False})
    
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE id = ?', (session['user_id'],))
    user = cursor.fetchone()
    
    if not user:
        return jsonify({'logged_in': False})
    
    cursor.execute('SELECT COUNT(*) as count FROM records WHERE user_id = ?', (user['id'],))
    total = cursor.fetchone()['count']
    
    cursor.execute('SELECT COUNT(*) as count FROM records WHERE user_id = ? AND timestamp >= ?',
                  (user['id'], datetime.now() - timedelta(days=30)))
    month = cursor.fetchone()['count']
    
    conn.close()
    
    return jsonify({
        'logged_in': True,
        'username': user['username'],
        'stats': {
            'total': total,
            'month': month,
            'score': user['eco_score'],
            'accuracy': 95
        }
    })

@app.route('/api/history')
def api_history():
    if 'user_id' not in session:
        return jsonify({'records': []})
    
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        'SELECT * FROM records WHERE user_id = ? ORDER BY timestamp DESC LIMIT 50',
        (session['user_id'],)
    )
    records = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return jsonify({'records': records})

@app.route('/api/devices')
def api_devices():
    if 'user_id' not in session:
        return jsonify({'devices': []})
    
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM devices WHERE user_id = ?', (session['user_id'],))
    devices = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return jsonify({'devices': devices})

@app.route('/api/devices/add', methods=['POST'])
def api_add_device():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': '请先登录'})
    
    data = request.get_json()
    device_name = data.get('device_name', '智能垃圾桶')
    device_type = data.get('device_type', 'standard')
    
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO devices (user_id, device_name, device_type) VALUES (?, ?, ?)',
        (session['user_id'], device_name, device_type)
    )
    device_id = cursor.lastrowid
    
    for bin_type in ['recyclable', 'wet', 'dry', 'hazardous']:
        cursor.execute(
            'INSERT INTO bin_status (device_id, bin_type) VALUES (?, ?)',
            (device_id, bin_type)
        )
    
    conn.commit()
    conn.close()
    
    return jsonify({'success': True, 'device_id': device_id})

@app.route('/api/bin_status/<int:device_id>')
def api_bin_status(device_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM bin_status WHERE device_id = ?', (device_id,))
    bins = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return jsonify({'bins': bins})

@app.route('/api/bin_status/update', methods=['POST'])
def api_update_bin_status():
    data = request.get_json()
    bin_id = data.get('bin_id')
    fill_level = data.get('fill_level')
    
    conn = get_db()
    cursor = conn.cursor()
    
    if fill_level is not None:
        cursor.execute('UPDATE bin_status SET fill_level = ? WHERE id = ?', (fill_level, bin_id))
    
    conn.commit()
    conn.close()
    
    return jsonify({'success': True})

@app.route('/api/notifications')
def api_notifications():
    if 'user_id' not in session:
        return jsonify({'notifications': []})
    
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        'SELECT * FROM notifications WHERE user_id = ? ORDER BY created_at DESC LIMIT 20',
        (session['user_id'],)
    )
    notifications = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return jsonify({'notifications': notifications})

@app.route('/api/notifications/read/<int:notification_id>', methods=['POST'])
def api_read_notification(notification_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('UPDATE notifications SET is_read = 1 WHERE id = ?', (notification_id,))
    conn.commit()
    conn.close()
    
    return jsonify({'success': True})

@app.route('/api/stats')
def api_stats():
    if 'user_id' not in session:
        return jsonify({'error': '请先登录'}), 401
    
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) as count FROM records WHERE user_id = ?', (session['user_id'],))
    total = cursor.fetchone()['count']
    
    cursor.execute(
        'SELECT category, COUNT(*) as count FROM records WHERE user_id = ? GROUP BY category',
        (session['user_id'],)
    )
    category_stats = {row['category']: row['count'] for row in cursor.fetchall()}
    
    cursor.execute(
        'SELECT COUNT(*) as count FROM records WHERE user_id = ? AND timestamp >= ?',
        (session['user_id'], datetime.now() - timedelta(days=30))
    )
    month = cursor.fetchone()['count']
    
    conn.close()
    
    return jsonify({
        'total': total,
        'month': month,
        'category_stats': category_stats
    })

@app.route('/api/password/change', methods=['POST'])
def api_change_password():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': '请先登录'})
    
    data = request.get_json()
    old_password = data.get('old_password', '')
    new_password = data.get('new_password', '')
    
    if not old_password or not new_password:
        return jsonify({'success': False, 'message': '密码不能为空'})
    
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE id = ?', (session['user_id'],))
    user = cursor.fetchone()
    
    if not user or not check_password_hash(user['password'], old_password):
        conn.close()
        return jsonify({'success': False, 'message': '原密码错误'})
    
    cursor.execute(
        'UPDATE users SET password = ? WHERE id = ?',
        (generate_password_hash(new_password), session['user_id'])
    )
    conn.commit()
    conn.close()
    
    return jsonify({'success': True, 'message': '密码修改成功'})

@app.route('/api/bin_alerts')
def api_bin_alerts():
    if 'user_id' not in session:
        return jsonify({'alerts': []})
    
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM devices WHERE user_id = ?', (session['user_id'],))
    devices = cursor.fetchall()
    
    alerts = []
    for device in devices:
        cursor.execute('''
            SELECT bs.*, d.device_name 
            FROM bin_status bs 
            JOIN devices d ON bs.device_id = d.id 
            WHERE bs.device_id = ? AND bs.fill_level >= 80
        ''', (device['id'],))
        
        for row in cursor.fetchall():
            bin_names = {
                'recyclable': '可回收物桶',
                'wet': '厨余垃圾桶',
                'dry': '其他垃圾桶',
                'hazardous': '有害垃圾桶'
            }
            alerts.append({
                'type': 'bin_full',
                'device_name': row['device_name'],
                'bin_type': row['bin_type'],
                'bin_name': bin_names.get(row['bin_type'], '垃圾桶'),
                'fill_level': row['fill_level'],
                'message': "{0}的{1}已满{2}%，请及时清理".format(row['device_name'], bin_names.get(row['bin_type'], '垃圾桶'), row['fill_level']),
                'level': 'warning' if row['fill_level'] < 90 else 'danger'
            })
    
    conn.close()
    return jsonify({'alerts': alerts})

@app.route('/api/filter_alerts')
def api_filter_alerts():
    if 'user_id' not in session:
        return jsonify({'alerts': []})
    
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM devices WHERE user_id = ?', (session['user_id'],))
    devices = cursor.fetchall()
    
    alerts = []
    for device in devices:
        cursor.execute('''
            SELECT bs.*, d.device_name 
            FROM bin_status bs 
            JOIN devices d ON bs.device_id = d.id 
            WHERE bs.device_id = ? AND bs.bin_type = 'wet'
        ''', (device['id'],))
        
        for row in cursor.fetchall():
            last_change = datetime.strptime(row['last_filter_change'], '%Y-%m-%d %H:%M:%S') if row['last_filter_change'] else datetime.now()
            days_passed = (datetime.now() - last_change).days
            
            if days_passed >= row['filter_days'] - 3:
                alerts.append({
                    'type': 'filter_change',
                    'device_name': row['device_name'],
                    'days_passed': days_passed,
                    'filter_days': row['filter_days'],
                    'message': "{0}的滤芯已使用{1}天，建议更换（周期{2}天）".format(row['device_name'], days_passed, row['filter_days']),
                    'level': 'warning' if days_passed < row['filter_days'] else 'danger'
                })
    
    conn.close()
    return jsonify({'alerts': alerts})

@app.route('/api/filter/change', methods=['POST'])
def api_change_filter():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': '请先登录'})
    
    data = request.get_json()
    bin_id = data.get('bin_id')
    
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        'UPDATE bin_status SET last_filter_change = ? WHERE id = ?',
        (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), bin_id)
    )
    conn.commit()
    conn.close()
    
    return jsonify({'success': True, 'message': '滤芯更换记录已更新'})

@app.route('/api/notifications/add', methods=['POST'])
def api_add_notification():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': '请先登录'})
    
    data = request.get_json()
    title = data.get('title', '')
    content = data.get('content', '')
    notif_type = data.get('type', 'info')
    
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO notifications (user_id, title, content, type) VALUES (?, ?, ?, ?)',
        (session['user_id'], title, content, notif_type)
    )
    conn.commit()
    conn.close()
    
    return jsonify({'success': True})

@app.route('/api/notifications/unread_count')
def api_unread_count():
    if 'user_id' not in session:
        return jsonify({'count': 0})
    
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        'SELECT COUNT(*) as count FROM notifications WHERE user_id = ? AND is_read = 0',
        (session['user_id'],)
    )
    count = cursor.fetchone()['count']
    conn.close()
    
    return jsonify({'count': count})

@app.route('/api/devices/delete/<int:device_id>', methods=['POST'])
def api_delete_device(device_id):
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': '请先登录'})
    
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM devices WHERE id = ? AND user_id = ?', (device_id, session['user_id']))
    if not cursor.fetchone():
        conn.close()
        return jsonify({'success': False, 'message': '设备不存在或无权限'})
    
    cursor.execute('DELETE FROM bin_status WHERE device_id = ?', (device_id,))
    cursor.execute('DELETE FROM devices WHERE id = ?', (device_id,))
    conn.commit()
    conn.close()
    
    return jsonify({'success': True, 'message': '设备已删除'})

@app.route('/api/devices/update/<int:device_id>', methods=['POST'])
def api_update_device(device_id):
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': '请先登录'})
    
    data = request.get_json()
    device_name = data.get('device_name')
    status = data.get('status')
    
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM devices WHERE id = ? AND user_id = ?', (device_id, session['user_id']))
    if not cursor.fetchone():
        conn.close()
        return jsonify({'success': False, 'message': '设备不存在或无权限'})
    
    if device_name:
        cursor.execute('UPDATE devices SET device_name = ? WHERE id = ?', (device_name, device_id))
    if status:
        cursor.execute('UPDATE devices SET status = ?, last_check = ? WHERE id = ?', 
                      (status, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), device_id))
    
    conn.commit()
    conn.close()
    
    return jsonify({'success': True})

@app.route('/api/south_tips')
def api_south_tips():
    month = datetime.now().month
    if month in [3, 4, 5]:
        season = 'spring'
        season_name = '春季'
    elif month in [6, 7, 8]:
        season = 'summer'
        season_name = '夏季'
    elif month in [9, 10, 11]:
        season = 'autumn'
        season_name = '秋季'
    else:
        season = 'winter'
        season_name = '冬季'
    
    weather_conditions = [
        ('晴天', 30, 45),
        ('多云', 50, 65),
        ('雷阵雨', 80, 95),
        ('高温', 35, 50),
        ('回南天', 85, 95),
        ('台风', 90, 98)
    ]
    weather, min_hum, max_hum = random.choice(weather_conditions)
    humidity = random.randint(min_hum, max_hum)
    
    return jsonify({
        'tips': SOUTH_WEATHER_TIPS,
        'seasonal': SOUTH_SEASONAL_TIPS[season],
        'current_season': season,
        'season_name': season_name,
        'weather': weather,
        'humidity': humidity,
        'city': '珠海',
        'month': month,
        'is_high_humidity': humidity >= 80,
        'is_rainy': weather in ['雷阵雨', '台风', '回南天'],
        'alert_level': 'danger' if humidity >= 85 else 'warning' if humidity >= 75 else 'normal'
    })

@app.route('/api/south_guide')
def api_south_guide():
    return jsonify({
        'title': '珠海家庭垃圾分类专属指南',
        'subtitle': '南方湿热气候适配 · 防潮除湿除味',
        'sections': [
            {
                'title': '🌡️ 湿热气候特点',
                'content': '珠海属南亚热带季风海洋性气候，年均气温22.4°C，年均湿度75%以上，年降雨量约2000mm。高温高湿环境加速垃圾腐烂，滋生细菌蚊虫。',
                'highlight': True
            },
            {
                'title': '� 厨余垃圾处理要点',
                'content': '①沥干水分：投放前用漏网沥干，或用报纸/厨房纸包裹\n②及时清理：夏季每日2次，其他季节每日1次\n③防臭除味：桶底铺报纸，撒小苏打，使用带盖桶\n④防虫防鼠：投放后清洗桶边，定期消毒周边区域',
                'items': ['沥干水分', '及时清理', '防臭除味', '防虫防鼠']
            },
            {
                'title': '♻️ 可回收物保存建议',
                'content': '①保持干燥：塑料瓶倒空晾干，纸类防潮\n②暂存室内：雨天不投放，避免淋湿\n③分类堆放：纸类、塑料、金属、玻璃分开\n④定期回收：积攒一定数量后统一回收',
                'items': ['保持干燥', '暂存室内', '分类堆放', '定期回收']
            },
            {
                'title': '⚠️ 有害垃圾注意事项',
                'content': '①密封包装：电池、药品用原包装或密封袋\n②防止破损：灯管、玻璃小心轻放\n③尽快投放：避免久存导致漏液变质\n④远离儿童：存放在儿童接触不到的地方',
                'items': ['密封包装', '防止破损', '尽快投放', '远离儿童']
            },
            {
                'title': '🧹 其他垃圾处理技巧',
                'content': '①密封投放：尿不湿、卫生巾等密封后再丢\n②完全熄灭：烟蒂确保熄灭，防止火灾\n③避免污染：不混入厨余垃圾，保持干燥\n④压缩体积：纸箱拆开压扁，节省空间',
                'items': ['密封投放', '完全熄灭', '避免污染', '压缩体积']
            }
        ],
        'seasonal_tips': SOUTH_SEASONAL_TIPS,
        'weather_tips': SOUTH_WEATHER_TIPS
    })

@app.route('/api/eco_ranking')
def api_eco_ranking():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT username, eco_score FROM users ORDER BY eco_score DESC LIMIT 10')
    ranking = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return jsonify({'ranking': ranking})

@app.route('/api/category_detail/<category>')
def api_category_detail(category):
    details = {
        'recyclable': {
            'name': '可回收物',
            'color': '#2196F3',
            'icon': 'fa-recycle',
            'description': '适宜回收利用和资源化利用的生活废弃物',
            'items': [
                {'name': '废纸类', 'examples': '报纸、杂志、书籍、纸板箱、办公用纸'},
                {'name': '塑料类', 'examples': '塑料瓶、塑料盆、塑料玩具、塑料桶'},
                {'name': '金属类', 'examples': '易拉罐、金属罐、金属餐具、铁钉'},
                {'name': '玻璃类', 'examples': '玻璃瓶、玻璃杯、玻璃镜子、玻璃窗'},
                {'name': '织物类', 'examples': '旧衣服、床单、毛巾、鞋子'}
            ],
            'tips': ['投放前请清空内容物', '保持干燥清洁', '压扁节省空间'],
            'south_tip': '南方湿度大，塑料和纸张请保持干燥，避免发霉影响回收。'
        },
        'wet': {
            'name': '厨余垃圾',
            'color': '#4CAF50',
            'icon': 'fa-leaf',
            'description': '易腐烂的生物质生活废弃物',
            'items': [
                {'name': '果皮果核', 'examples': '苹果核、香蕉皮、西瓜皮、橘子皮'},
                {'name': '蔬菜残叶', 'examples': '菜叶、菜根、土豆皮、葱姜蒜皮'},
                {'name': '剩菜剩饭', 'examples': '米饭、面条、肉类、鱼骨、蛋壳'},
                {'name': '茶叶渣', 'examples': '茶叶渣、中药渣、咖啡渣'},
                {'name': '过期食品', 'examples': '过期面包、糕点、水果'}
            ],
            'tips': ['沥干水分后投放', '去除包装袋', '不要混入其他垃圾'],
            'south_tip': '南方气候潮湿，厨余垃圾请务必沥干水分，建议用报纸包裹后再丢弃，防止异味和滋生细菌。'
        },
        'dry': {
            'name': '其他垃圾',
            'color': '#9E9E9E',
            'icon': 'fa-trash',
            'description': '除可回收物、有害垃圾、厨余垃圾以外的其他生活废弃物',
            'items': [
                {'name': '卫生用品', 'examples': '卫生纸、纸巾、尿不湿、卫生巾'},
                {'name': '烟蒂灰土', 'examples': '烟蒂、灰土、宠物粪便、毛发'},
                {'name': '陶瓷制品', 'examples': '碎陶瓷、碎玻璃（非回收）、花盆'},
                {'name': '一次性用品', 'examples': '一次性餐具、塑料袋、保鲜膜'},
                {'name': '其他', 'examples': '大骨头、硬贝壳、复合包装袋'}
            ],
            'tips': ['尽量保持干燥', '不要混入其他类别垃圾', '难以辨识的垃圾投入此类'],
            'south_tip': '潮湿天气请保持干燥，避免污染其他垃圾。'
        },
        'hazardous': {
            'name': '有害垃圾',
            'color': '#F44336',
            'icon': 'fa-skull-crossbones',
            'description': '对人体健康或自然环境造成直接或潜在危害的生活废弃物',
            'items': [
                {'name': '电池类', 'examples': '废电池、蓄电池、纽扣电池'},
                {'name': '灯管类', 'examples': '废荧光灯管、节能灯、LED灯'},
                {'name': '药品类', 'examples': '过期药品、药品包装、药瓶'},
                {'name': '油漆类', 'examples': '废油漆、油漆桶、杀虫剂'},
                {'name': '化妆品', 'examples': '过期化妆品、指甲油、染发剂'}
            ],
            'tips': ['轻放防止破损', '不要混入其他垃圾', '密封包装防止泄漏'],
            'south_tip': '潮湿环境可能导致电池等有害物品泄漏，请尽快投放。'
        }
    }
    
    detail = details.get(category)
    if not detail:
        return jsonify({'error': '分类不存在'}), 404
    
    return jsonify(detail)

@app.route('/points')
def points_page():
    if 'user_id' not in session:
        return redirect(url_for('login_page'))
    return render_template('points.html', active_page='points')

@app.route('/leaderboard')
def leaderboard_page():
    return render_template('leaderboard.html', active_page='leaderboard')

@app.route('/api/points')
def api_get_points():
    if 'user_id' not in session:
        return jsonify({'error': '未登录'}), 401
    
    user_id = session['user_id']
    user_points = PointsService.get_or_create_user_points(user_id)
    
    if user_points:
        exp_for_next = PointsService.get_exp_for_next_level(user_points['level'])
        exp_progress = user_points['exp'] % exp_for_next if exp_for_next > 0 else 0
        
        return jsonify({
            'success': True,
            'points': {
                'total': user_points['total_points'],
                'level': user_points['level'],
                'exp': user_points['exp'],
                'exp_for_next': exp_for_next,
                'exp_progress': exp_progress,
                'continuous_days': user_points['continuous_days'],
                'total_checkins': user_points['total_checkins'],
                'total_recognitions': user_points['total_recognitions'],
                'total_questions': user_points['total_questions']
            }
        })
    
    return jsonify({'success': False, 'error': '获取积分信息失败'})

@app.route('/api/points/checkin', methods=['POST'])
def api_checkin():
    if 'user_id' not in session:
        return jsonify({'error': '未登录'}), 401
    
    user_id = session['user_id']
    result = PointsService.checkin(user_id)
    
    return jsonify(result)

@app.route('/api/points/records')
def api_point_records():
    if 'user_id' not in session:
        return jsonify({'error': '未登录'}), 401
    
    user_id = session['user_id']
    limit = request.args.get('limit', 20, type=int)
    records = PointsService.get_point_records(user_id, limit)
    
    return jsonify({'success': True, 'records': records})

@app.route('/api/badges')
def api_get_badges():
    if 'user_id' not in session:
        return jsonify({'error': '未登录'}), 401
    
    user_id = session['user_id']
    badges = PointsService.get_user_badges(user_id)
    
    unlocked_count = sum(1 for b in badges if b.get('unlocked'))
    
    return jsonify({
        'success': True,
        'badges': badges,
        'unlocked_count': unlocked_count,
        'total_count': len(badges)
    })

@app.route('/api/leaderboard')
def api_leaderboard():
    period = request.args.get('period', 'weekly')
    limit = request.args.get('limit', 10, type=int)
    
    leaderboard = PointsService.get_leaderboard(period, limit)
    
    user_rank = None
    if 'user_id' in session:
        user_id = session['user_id']
        for item in leaderboard:
            if item['id'] == user_id:
                user_rank = item['rank']
                break
    
    return jsonify({
        'success': True,
        'leaderboard': leaderboard,
        'user_rank': user_rank,
        'period': period
    })

@app.route('/stats')
def stats_page():
    if 'user_id' not in session:
        return redirect(url_for('login_page'))
    return render_template('stats.html', active_page='stats')

@app.route('/api/stats/overview')
def api_stats_overview():
    if 'user_id' not in session:
        return jsonify({'error': '未登录'}), 401
    
    user_id = session['user_id']
    overview = StatsService.get_overview(user_id)
    
    return jsonify({'success': True, 'overview': overview})

@app.route('/api/stats/category')
def api_stats_category():
    if 'user_id' not in session:
        return jsonify({'error': '未登录'}), 401
    
    user_id = session['user_id']
    days = request.args.get('days', 7, type=int)
    
    stats = StatsService.get_category_stats(user_id, days)
    total = sum(stats.values())
    
    return jsonify({
        'success': True,
        'stats': stats,
        'total': total,
        'days': days
    })

@app.route('/api/stats/trend')
def api_stats_trend():
    if 'user_id' not in session:
        return jsonify({'error': '未登录'}), 401
    
    user_id = session['user_id']
    days = request.args.get('days', 7, type=int)
    
    trend = StatsService.get_daily_trend(user_id, days)
    
    return jsonify({
        'success': True,
        'trend': trend,
        'days': days
    })

@app.route('/api/stats/contribution')
def api_stats_contribution():
    if 'user_id' not in session:
        return jsonify({'error': '未登录'}), 401
    
    user_id = session['user_id']
    days = request.args.get('days', 30, type=int)
    
    contribution = StatsService.get_contribution(user_id, days)
    
    return jsonify({
        'success': True,
        'contribution': contribution,
        'days': days
    })

@app.route('/api/stats/heatmap')
def api_stats_heatmap():
    if 'user_id' not in session:
        return jsonify({'error': '未登录'}), 401
    
    user_id = session['user_id']
    days = request.args.get('days', 30, type=int)
    
    heatmap = StatsService.get_heatmap(user_id, days)
    
    return jsonify({
        'success': True,
        'heatmap': heatmap,
        'days': days
    })

@app.route('/api/stats/compare')
def api_stats_compare():
    if 'user_id' not in session:
        return jsonify({'error': '未登录'}), 401
    
    user_id = session['user_id']
    compare = StatsService.get_community_compare(user_id)
    
    return jsonify({'success': True, 'compare': compare})

with app.app_context():
    init_db()
    load_model()

if __name__ == '__main__':
    print("=" * 60)
    print("智能家居垃圾分类系统 - 珠海家庭版")
    print("=" * 60)
    print("访问地址: http://localhost:5000")
    print("南方湿热气候适配")
    print("长辈友好设计")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=True)

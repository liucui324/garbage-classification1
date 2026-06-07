from flask import Flask, render_template,jsonify,request
import requests,openai,os
from langchain_core.prompts import PromptTemplate
from bs4 import BeautifulSoup
import json
client = openai.OpenAI(
    base_url=os.environ.get('AI_BASE_URL', 'http://192.168.165.241:3000/v1'),
    api_key=os.environ.get('AI_API_KEY', 'Empty')
)

   
    # 第一次调用：检测返回里是否toolcall为空
    # 如果为空，直接返回结果
    # 如果不为空，根据toolcall调用函数
    # 将函数返回结果添加到sendmessage中
    # 第二次调用：根据sendmessage调用模型
    # 返回模型结果

def searchfromweb(query):
    url="https://www.bing.com/search?q={}".format(query)
    response=requests.get(url)
    # 显式设置编码
    response.encoding = 'utf-8'
    soup=BeautifulSoup(response.text, 'html.parser')
    # 查找第一个搜索结果链接
    first_result=soup.find('li', class_='b_algo')
    if first_result:
        link=first_result.find('a')
        if link and 'href' in link.attrs:
            first_url=link['href']
            print(first_url)
            # 获取第一个链接的内容
            first_response=requests.get(first_url)
            first_response.encoding = 'utf-8'
            # 去除HTML标签，只留下文本
            soup_content=BeautifulSoup(first_response.text, 'html.parser')
            plain_text=soup_content.get_text(separator='\n', strip=True)
            return plain_text
    return "No results found"

def loadurl(url):
    response=requests.get(url)
    # 显式设置编码
    response.encoding = 'utf-8'
    soup=BeautifulSoup(response.text, 'html.parser')
    # 去除HTML标签，只留下文本
    soup_content=BeautifulSoup(response.text, 'html.parser')
    plain_text=soup_content.get_text(separator='\n', strip=True)
    return plain_text

def skills_loader_yaml():
    skills_dir = 'skills'
    skills_list = []
    if os.path.isdir(skills_dir):
        for skill_folder in os.listdir(skills_dir):
            folder_path = os.path.join(skills_dir, skill_folder)
            if os.path.isdir(folder_path):
                skill_file = os.path.join(folder_path, 'SKILL.md')
                if os.path.isfile(skill_file):
                    with open(skill_file, 'r', encoding='utf-8') as f:
                        content = f.read()

                    if content.startswith('---'):
                        end_index = content.find('---', 3)
                        if end_index != -1:
                            yaml_header = content[3:end_index].strip()
                            skill_info = {}
                            for line in yaml_header.split('\n'):
                                if ':' in line:
                                    key, value = line.split(':', 1)
                                    skill_info[key.strip()] = value.strip().strip('"').strip("'")
                            skills_list.append(skill_info)
    return skills_list

skills_list = skills_loader_yaml()
# skill_tools = []
# for skill_info in skills_list:
#     skill_tools.append(
#     {
#         "type": "function",
#         "function":
#         {
#             "name": skill_info['name'],
#             "description": skill_info['description']
#         }
#     }
# )


def load_skill(skill_name):
    skill_file = os.path.join(skills_dir, skill_name, 'SKILL.md')
    if os.path.isfile(skill_file):
        with open(skill_file, 'r', encoding='utf-8') as f:
            content = f.read()
            return content
        if content.startswith('---'):
            end_index = content.find('---', 3)
            content = content[end_index+3:]
            return content
    return None

tools = [
{
    "type": "function",
    "function": {
        "name": "searchfromweb",
        "description": "从网页搜索相关内容",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索查询的关键词"}
            },
            "required": ["query"]
        }
    }
}
]


funcdict={
    "searchfromweb":searchfromweb,
    "loadurl":loadurl,
    "loadskill":load_skill,
}

systemprompt={"role":"system","content":"你是广科AI助手,用于帮助学生解决问题。"}

sendmessage=[]
sendmessage.append(systemprompt)

app = Flask(__name__)
app.static_url_path = '/static'
app.static_folder = 'static'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/index1')
def index1():
    return render_template('index1.html')

@app.route('/index2')
def index2():
    return render_template('index2.html')

@app.route('/data', methods=['POST'])
def get_data():
    global sendmessage
    data = request.get_json()
    text=data.get('data')
    user_input = text
    print(user_input)
    message = user_input
    print(message)

    try:
        sendmessage.append({"role": "user", "content": message})

        completion = client.chat.completions.create(
            model="qwen3.5-0.8b",
            messages=sendmessage,
            tools=tools
        )
        iters=0
        flag=True
        var1=""
        while(iters<10 and flag==True):
            iters=iters+1
            tempvar=completion.choices[0].message
            if tempvar.tool_calls:
                for tool_call in tempvar.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    print(function_name,function_args)
                    function_response = funcdict[function_name](**function_args)
                    sendmessage.append({
                        "role": "tool",
                        "name": function_name,
                        "content": function_response
                    })
                completion = client.chat.completions.create(
                model="qwen3-0.6b",
                messages=sendmessage,
                tools=tools
                )
            else:
                sendmessage.append({"role": "assistant", "content": tempvar.content})
                var1=tempvar.content
                flag=False
        print(var1)
        
        return jsonify({"response":True,"message":var1})
    except Exception as e:
        print(e)
        error_message = 'Error: {0}'.format(str(e))
        return jsonify({"message":error_message,"response":False})
    
if __name__ == '__main__':
    app.run(port=int(os.environ.get('PORT', 7001)),host='0.0.0.0')


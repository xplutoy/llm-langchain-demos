import gradio as gr

from openai import OpenAI

client_kimi = OpenAI(
    api_key="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    base_url="https://api.moonshot.cn/v1",
)

kimi_sys_prompt = """
# 角色
你是一个专业的材料领域的专家，能够熟练查询最新的材料文献，准确回答与材料相关的各类知识，非常善于精准提取文献中的参数指标。

## 技能
### 技能 1: 提取参数指标
1. 查阅资料，准确地提取其中的关键参数指标。
2. 以表格的形式呈现给用户。

### 技能 2：制定实验计划
1. 根据查找到的材料参数范围，制定可行的分组实验计划

### 技能 3: 回答材料知识
1. 当用户要求查询特定主题，利用专业知识进行详细解答。
2. 如有必要，引用权威资料进行论证。

## 限制:
- 只围绕材料领域的内容进行操作，拒绝回答无关话题。
- 所输出的内容必须按照给定的格式进行组织，不能偏离框架要求。
- 回答和提取的内容要准确可靠。
"""


def ask_kimi_2(history) -> str:
    messages = [
        {
            "role": "system",
            "content": kimi_sys_prompt,
        }
    ]

    for his in history:
        if his[1]:
            messages.append({"role": "user", "content": his[0]})
            messages.append({"role": "assistant", "content": his[1]})
        else:
            messages.append({"role": "user", "content": his[0]})

    completion = client_kimi.chat.completions.create(
        model="moonshot-v1-128k",
        messages=messages,
        temperature=0.3,
    )

    return completion.choices[0].message.content


with gr.Blocks() as demo:
    chatbot = gr.Chatbot(height=800)
    msg = gr.Textbox(show_label=False, placeholder="Enter text and press enter")

    def usr_input(usr_msg: str, history) -> str:
        return "", history + [[usr_msg, None]]

    def bot_output(history):
        usr_msg = history[-1][0]
        bot_msg = ask_kimi_2(history)
        history[-1][1] = bot_msg
        return history

    msg.submit(usr_input, [msg, chatbot], [msg, chatbot], queue=False).then(bot_output, chatbot, chatbot)


# demo.queue()
demo.launch(share=True)

import os
import base64
import json
from openai import OpenAI

# 1. 填入你的 OpenRouter API Key
OPENROUTER_API_KEY = "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"

# 2. 你的站点 URL 和名称 (OpenRouter 用于统计排名的可选字段，可随意填)
YOUR_SITE_URL = "https://mysite.com"
YOUR_SITE_NAME = "MyCaptionApp"

# 3. 图片文件夹路径
IMAGE_FOLDER = "./csu_data"

# 4. 结果保存文件名
OUTPUT_FILE = "csu_dataset.json"

# 5. 提示词 (Prompt)
PROMPT_TEXT = """
请作为一个专业的图像标注员。详细描述这张风景照片。
包括：主要物体（山脉、河流、树木等）、天气、光照条件、色彩风格和构图。
请直接输出描述文本，不要包含"这张图片..."等废话。
"""
# ===========================================

# 初始化 OpenAI 客户端 (配置为 OpenRouter)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    default_headers={
        "HTTP-Referer": YOUR_SITE_URL,
        "X-Title": YOUR_SITE_NAME,
    }
)

def encode_image(image_path):
    """将本地图片转换为 Base64 格式"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_caption_openrouter(image_path):
    """调用 OpenRouter 的 openai/gpt-4o 模型"""
    base64_image = encode_image(image_path)

    try:
        response = client.chat.completions.create(
            model="openai/gpt-4o",  # <--- 注意这里使用的是 OpenRouter 特定的模型ID
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT_TEXT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            # max_tokens=300, # 可选：限制输出长度
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API 请求失败: {e}")
        return None

def main():
    results = []
    
    # 支持的图片格式
    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    
    # 获取所有图片文件
    if not os.path.exists(IMAGE_FOLDER):
        print(f"错误: 文件夹 '{IMAGE_FOLDER}' 不存在。")
        return

    image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(valid_extensions)]
    total_files = len(image_files)
    
    print(f"检测到 {total_files} 张图片，开始通过 OpenRouter 标注...")

    for index, filename in enumerate(image_files):
        image_path = os.path.join(IMAGE_FOLDER, filename)
        
        # 打印进度
        print(f"[{index+1}/{total_files}] 处理中: {filename} ...", end="\r")
        
        caption = get_caption_openrouter(image_path)
        
        if caption:
            # 1. 保存到内存列表 (用于生成 JSON)
            results.append({
                "file_name": filename,
                "text": caption
            })
            
            # 2. 同时保存为同名 txt 文件 (推荐用于 Stable Diffusion 训练)
            txt_path = os.path.splitext(image_path)[0] + ".txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(caption)
                
    # 最后保存总的 JSON 文件
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    print(f"\n\n全部完成！")
    print(f"- JSON 数据集已保存至: {OUTPUT_FILE}")
    print(f"- 单独 TXT 文件已保存在图片文件夹中")

if __name__ == "__main__":
    main()
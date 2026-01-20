import json
import os
import whisper
import yt_dlp
from pathlib import Path
from tqdm import tqdm

# ================= 配置区 =================
INPUT_JSON = "search_results.json"  # 你的原始搜索结果文件
OUTPUT_JSON = "search_results_with_text.json"  # 输出文件
URL_FIELD = "video_urls"  # JSON中存放视频链接的字段名 (根据你的数据修改，可能是 url, video_url 等)
ID_FIELD = "id"  # 唯一ID字段 (用于临时文件名)
MODEL_SIZE = "base"  # Whisper模型: tiny, base, small, medium, large (推荐 base 或 small)


# =========================================

def download_audio(video_url, output_filename):
    """使用 yt-dlp 下载音频"""
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': output_filename,  # 临时文件名模板
        'quiet': True,
        'no_warnings': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        return f"{output_filename}.mp3"  # yt-dlp 会自动加上扩展名
    except Exception as e:
        print(f"\n[Error] 下载失败 {video_url}: {e}")
        return None


def main():
    # 1. 加载 Whisper 模型
    print(f"正在加载 Whisper 模型 ({MODEL_SIZE})...")
    model = whisper.load_model(MODEL_SIZE)

    # 2. 读取数据
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 如果是列表结构
    items = data if isinstance(data, list) else data.get("items", [])

    print(f"开始处理 {len(items)} 条视频数据...")

    # 3. 循环处理
    success_count = 0
    for item in tqdm(items):
        url = item.get(URL_FIELD)
        uid = item.get(ID_FIELD, "temp")

        # 跳过没有链接的数据
        if not url:
            continue

        # 临时音频路径
        temp_audio_name = f"temp_{uid}"

        # A. 下载音频
        audio_path = download_audio(url, temp_audio_name)

        if audio_path and os.path.exists(audio_path):
            try:
                # B. Whisper 转录
                # initial_prompt 可以引导模型输出繁体/简体或特定术语，这里可选
                result = model.transcribe(audio_path)
                transcribed_text = result['text']

                # C. 关键一步：将文本回填到 content 字段
                # 你的 Step 00 脚本会自动识别 "content" 字段
                item['content'] = transcribed_text.strip()
                item['source_type'] = 'video_asr'  # 标记来源

                success_count += 1

            except Exception as e:
                print(f"\n[Error] 转录失败 {url}: {e}")
            finally:
                # D. 清理临时文件 (非常重要，否则硬盘会爆)
                if os.path.exists(audio_path):
                    os.remove(audio_path)
        else:
            # 下载失败或非视频链接，保留原样或标记
            item['content_status'] = 'download_failed'

    # 4. 保存结果
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"\n处理完成！成功转换 {success_count} 条视频。")
    print(f"结果已保存至: {OUTPUT_JSON}")
    print("现在可以将此文件放入 data/ 目录供 Step 00 使用了。")


if __name__ == "__main__":
    main()
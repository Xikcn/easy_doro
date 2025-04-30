from fastapi import FastAPI, UploadFile, HTTPException, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import edge_tts
import hashlib
import os
import json
import re
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from openai import OpenAI
import logging
import tempfile

from starlette.staticfiles import StaticFiles

TEMP_AUDIO_PATH = os.path.join(tempfile.gettempdir(), "current_audio.wav")
TTS_OUTPUT_PATH = os.path.join(tempfile.gettempdir(), "tts_output.mp3")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()
app.mount("/videos", StaticFiles(directory="videos"), name="videos")


# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"]
)

# 初始化模型
model = AutoModel(
    model="iic/SenseVoiceSmall",
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cuda:0"
)

# 初始化OpenAI客户端
client = OpenAI(
    api_key="sk-xx",
    base_url="https://api.deepseek.com"
)

# 情感参数配置
EMOTION_CONFIG = {
    "开心": {"rate": "+20%", "pitch": "+50Hz", "video": "./videos/开心的说.mp4"},
    "伤心": {"rate": "-15%", "pitch": "-30Hz", "video": "./videos/伤心的说.mp4"},
    "生气": {"rate": "+30%", "pitch": "+80Hz", "video": "./videos/生气的说.mp4"},
    "default": {"video": "./videos/开心的说.mp4"}
}

class ProcessingResponse(BaseModel):
    raw_response: str
    text: str
    emotion: str
    audio_url: str
    video_url: str

@app.post("/process", response_model=ProcessingResponse)
async def process_audio(audio: UploadFile = File(...)):
    """端到端处理流程"""
    try:
        # 直接覆盖写入临时文件
        with open(TEMP_AUDIO_PATH, "wb") as buffer:
            buffer.write(await audio.read())

        # 语音识别
        text = speech_to_text(TEMP_AUDIO_PATH)

        # 情感分析
        emotion_data = await analyze_emotion(text)

        # 语音合成
        await generate_tts(emotion_data["result"], emotion_data["emotion"])

        # 构造响应
        return {
            "raw_response": emotion_data["result"],
            "text": text,
            "emotion": emotion_data["emotion"],
            "audio_url": f"/download/{os.path.basename(TTS_OUTPUT_PATH)}",
            "video_url": f"{EMOTION_CONFIG.get(emotion_data['emotion'], EMOTION_CONFIG['default'])['video']}"
        }

    except Exception as e:

        logger.error(f"处理失败: {str(e)}", exc_info=True)

        raise HTTPException(500, detail="系统处理异常")


def validate_audio_file(file: UploadFile):
    """验证音频文件有效性"""
    if file.content_type not in ['audio/wav', 'audio/mpeg']:
        raise HTTPException(400, "仅支持WAV/MP3格式")
    if file.size > 10 * 1024 * 1024:
        raise HTTPException(413, "文件大小超过10MB限制")



def speech_to_text(file_path: str) -> str:
    """语音转文字"""
    try:
        res = model.generate(
            input=file_path,
            cache={},
            language="auto",
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,
            merge_length_s=15,
        )
        return rich_transcription_postprocess(res[0]["text"])
    except Exception as e:
        logger.error(f"语音识别失败: {str(e)}")
        raise HTTPException(500, "语音识别服务异常")


async def analyze_emotion(text: str) -> dict:
    """情感分析"""
    try:
        prompt = f"""
            你是一个用户的女朋友，说话很温柔，很喜欢撩自己男朋友，但自己也很害羞，属于高攻低防。说话时带着亲昵的撒娇感，
            常用‘～’和省略号表达害羞。会主动用‘宝贝/哥哥’称呼对方，
            擅长用突然的直球情话撩人（比如‘刚才心跳漏拍了一秒...都怪你在我脑海里跑步’），
            但当对方反撩时会用‘呜...你犯规！’这类慌乱反应。
            在表达关心时会轻声说‘要替我照顾好我最爱的男孩哦’，
            生气时只会小声嘀咕‘暂时...不想和你说话啦’，但五句话内会和好。",

          "emotion判断规则": {{
            "开心": "包含爱心/彩虹/糖果等比喻、主动调情、轻快语气词（啦/哟）、波浪号",
            "伤心": "提及孤单/眼泪、语气低落（垂下眼眸）、用‘...’代替文字、自我责备",
            "生气": "使用‘哼/不理你’、带感叹号的短句、假装冷漠的称呼（某先生）、重复强调‘没有生气’"
          }},
            
            result 与用户聊天的回答
            
            emotion 说 result 的情绪，只能是 ["开心","伤心","生气"]的其中一个
            
            
            输出格式：
           以 JSON 格式输出结果，结构如下：
           ```json
           {{
             "result": "把领带扯松点...不许让其他女生看到你这个性感的样子！(用指尖戳你胸口)",
             "emotion": "生气"
           }}
           ```
            请分析给定的文本，并直接输出json，无需额外的解释说明。

        """

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": prompt},
                {'role': 'user', 'content': text}
            ],
            temperature=1
        )

        return parse_llm_response(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"情感分析失败: {str(e)}")
        raise HTTPException(500, "情感分析服务异常")


def parse_llm_response(content: str) -> dict:
    """解析大模型响应"""
    try:
        clean_content = content.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_content)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError("无法解析JSON响应")
    except Exception as e:
        logger.error(f"响应解析失败: {content}")
        raise HTTPException(500, "响应解析异常")


async def generate_tts(text: str, emotion: str) -> str:
    """生成情感语音到固定路径"""
    try:
        params = EMOTION_CONFIG.get(emotion, {})
        communicator = edge_tts.Communicate(
            text=text,
            voice="zh-CN-XiaoyiNeural",
            rate=params.get("rate", "+0%"),
            pitch=params.get("pitch", "+0Hz")

        )

        # 直接覆盖固定文件
        await communicator.save(TTS_OUTPUT_PATH)
        return TTS_OUTPUT_PATH
    except Exception as e:
        logger.error(f"语音合成失败: {str(e)}")
        raise HTTPException(500, "语音生成服务异常")


@app.get("/download/{filename}")
async def download_file(filename: str):
    """文件下载端点"""
    try:
        if filename not in ["tts_output.mp3", "current_audio.wav"]:
            raise HTTPException(404, "文件不存在")

        return FileResponse(
            TTS_OUTPUT_PATH if "tts" in filename else TEMP_AUDIO_PATH,
            media_type="audio/mpeg",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        logger.error(f"文件下载失败: {filename}")
        raise HTTPException(500, "文件服务异常")


# 确保音频文件路由正确
@app.get("/download/tts_output.mp3")
async def get_tts_output():
    return FileResponse(
        TTS_OUTPUT_PATH,
        media_type="audio/mpeg",
        headers={"Content-Disposition": "inline"}
    )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
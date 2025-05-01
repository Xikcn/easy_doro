from typing import List, Dict
from fastapi import FastAPI, UploadFile, HTTPException, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from openai import OpenAI
from pydantic import BaseModel
import edge_tts
import os
import json
import re
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import logging
import tempfile
from starlette.staticfiles import StaticFiles

TEMP_AUDIO_PATH = os.path.join(tempfile.gettempdir(), "current_audio.wav")
TTS_OUTPUT_PATH = os.path.join(tempfile.gettempdir(), "tts_output.mp3")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()
app.mount("/videos", StaticFiles(directory="videos"), name="videos")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"]
)

# Initialize ASR model
model = AutoModel(
    model="iic/SenseVoiceSmall",
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cuda:0"
)

# Emotion configuration
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


# Initialize OpenAI client
client = OpenAI(
    api_key="sk-xx",
    base_url="https://api.deepseek.com"
)

class ConversationMemory:
    def __init__(self, max_messages=5):
        self.messages: List[Dict] = []
        self.max_messages = max_messages
        self.summary = ""

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > self.max_messages:
            self._summarize()

    def _summarize(self):
        # Simple summarization logic
        conversation = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.messages])
        self.summary = f"Previous conversation summary: {conversation[:500]}..."
        self.messages = self.messages[-self.max_messages//2:]

    def get_context(self):
        if self.summary:
            return [{"role": "system", "content": self.summary}] + self.messages
        return self.messages


# Global memory instance
memory = ConversationMemory()


@app.post("/process", response_model=ProcessingResponse)
async def process_audio(audio: UploadFile = File(...)):
    """End-to-end processing pipeline"""
    try:
        # Save uploaded file
        with open(TEMP_AUDIO_PATH, "wb") as buffer:
            buffer.write(await audio.read())

        # Speech to text
        text = speech_to_text(TEMP_AUDIO_PATH)
        memory.add_message("user", text)

        # Emotion analysis using direct API call
        emotion_data = await analyze_emotion(text)

        # Generate TTS
        await generate_tts(emotion_data["result"], emotion_data["emotion"])
        memory.add_message("assistant", emotion_data["result"])

        # 确保情绪类型是小写或大写一致
        emotion = emotion_data["emotion"].strip()

        # 获取对应的视频路径
        video_config = EMOTION_CONFIG.get(emotion, EMOTION_CONFIG["default"])
        video_path = video_config["video"]

        return {
            "raw_response": emotion_data["result"],
            "text": text,
            "emotion": emotion,
            "audio_url": f"/download/{os.path.basename(TTS_OUTPUT_PATH)}",
            "video_url": video_path
        }

    except Exception as e:
        logger.error(f"Processing failed: {str(e)}", exc_info=True)
        raise HTTPException(500, detail="System processing error")
def validate_audio_file(file: UploadFile):
    """Validate audio file"""
    if file.content_type not in ['audio/wav', 'audio/mpeg']:
        raise HTTPException(400, "Only WAV/MP3 formats supported")
    if file.size > 10 * 1024 * 1024:
        raise HTTPException(413, "File size exceeds 10MB limit")


def speech_to_text(file_path: str) -> str:
    """Speech to text"""
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
        logger.error(f"ASR failed: {str(e)}")
        raise HTTPException(500, "ASR service error")


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
async def analyze_emotion(text: str) -> dict:
    """Analyze emotion using direct API call with memory"""
    try:
        messages = [
            {
                "role": "system",
                "content": prompt
            }
        ] + memory.get_context()

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=1
        )

        result = parse_llm_response(response.choices[0].message.content)
        # 确保情绪值去除前后空格
        result["emotion"] = result["emotion"].strip()
        return result
    except Exception as e:
        logger.error(f"Emotion analysis failed: {str(e)}")
        raise HTTPException(500, "Emotion analysis service error")

def parse_llm_response(content: str) -> dict:
    """Parse LLM response"""
    try:
        clean_content = content.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_content)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError("Cannot parse JSON response")
    except Exception as e:
        logger.error(f"Response parsing failed: {content}")
        raise HTTPException(500, "Response parsing error")


async def generate_tts(text: str, emotion: str) -> str:
    """Generate TTS with emotion"""
    try:
        params = EMOTION_CONFIG.get(emotion, {})
        communicator = edge_tts.Communicate(
            text=text,
            voice="zh-CN-XiaoyiNeural",
            rate=params.get("rate", "+0%"),
            pitch=params.get("pitch", "+0Hz")
        )

        await communicator.save(TTS_OUTPUT_PATH)
        return TTS_OUTPUT_PATH
    except Exception as e:
        logger.error(f"TTS generation failed: {str(e)}")
        raise HTTPException(500, "TTS service error")


@app.get("/download/{filename}")
async def download_file(filename: str):
    """File download endpoint"""
    try:
        if filename not in ["tts_output.mp3", "current_audio.wav"]:
            raise HTTPException(404, "File not found")

        return FileResponse(
            TTS_OUTPUT_PATH if "tts" in filename else TEMP_AUDIO_PATH,
            media_type="audio/mpeg",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        logger.error(f"File download failed: {filename}")
        raise HTTPException(500, "File service error")


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
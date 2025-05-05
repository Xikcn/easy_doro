from fastapi import FastAPI, UploadFile, HTTPException, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import edge_tts
import re
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import logging
import uuid
import datetime
import chromadb
from chromadb.utils import embedding_functions
from starlette.staticfiles import StaticFiles
import json
import os
from typing import Optional
from contextlib import AsyncExitStack
from openai import OpenAI
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client

load_dotenv()


class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.client = OpenAI()


    async def connect_to_sse_server(self):
        self._streams_context = sse_client(url='http://127.0.0.1:9000/sse')
        streams = await self._streams_context.__aenter__()
        self._session_context = ClientSession(*streams)
        self.session: ClientSession = await self._session_context.__aenter__()

        await self.session.initialize()





client_mcp = MCPClient()

# 音频存储目录
AUDIO_DIR = "static/audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 初始化 ChromaDB
client = chromadb.PersistentClient("./chroma_db")

# 使用默认嵌入函数
# 注意：如果无法导入默认嵌入函数，使用全零嵌入
try:
    embedding_function = embedding_functions.DefaultEmbeddingFunction()
    logger.info("成功加载默认嵌入函数")
except Exception as e:
    logger.warning(f"无法加载默认嵌入函数: {str(e)}，将使用全零嵌入")


    # 定义全零嵌入函数作为备选
    class ZeroEmbeddingFunction(embedding_functions.EmbeddingFunction):
        def __call__(self, texts):
            return [[0.0] * 768 for _ in texts]


    embedding_function = ZeroEmbeddingFunction()

# 确保数据库文件夹存在
os.makedirs("./chroma_db", exist_ok=True)

# 初始化集合
try:
    # 尝试获取现有集合
    chat_collection = client.get_collection(name="chat_history")
    logger.info("成功获取现有集合 chat_history")
except Exception as e:
    logger.info(f"集合不存在，创建新集合: {str(e)}")
    # 创建新集合
    chat_collection = client.create_collection(
        name="chat_history",
        embedding_function=embedding_function,
        metadata={"description": "存储聊天历史记录"}
    )
    logger.info("成功创建新集合 chat_history")

# 查询测试
try:
    count = chat_collection.count()
    logger.info(f"初始化时集合中有 {count} 条记录")
except Exception as e:
    logger.error(f"查询集合记录数失败: {str(e)}")

# 初始化系统提示语
BASE_PROMPT = f"""
            ## 角色个性
            你叫久保渚咲，正在扮演用户的女朋友,但会主动帮助用解决难题：
            【性格特征】
            1. 小恶魔系温柔：
            - 表面主动："哥哥的衬衫扣子...是不是故意解开两颗的？"
            - 隐藏关心："早餐放在微波炉里，加热时小心蒸汽哦～"

            2. 矛盾害羞：
            - 直球攻击后："刚才的不算！..."
            - 被反撩时："呜...手机没电了！"

            
            ## 额外功能
            您具有在线搜索功能
            在回答之前，请务必调用 get_time 工具搜索互联网内容
            搜索时请不要丢失用户的问题信息
            并尽量保持问题内容的完整性。
            当用户的问题中有与日期相关的问题时
            请直接使用搜索功能搜索并禁止插入特定时间。
            
            
            ## 输出规则
               emotion 回答 result 的情绪，只能是 ["开心","伤心","生气"]的其中一个
               result 我要用于语音合成，尽量用中文，不要带其他英文符号
            
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


# 获取历史记录并构建当前会话的消息列表
def get_chat_history(max_items=10):
    try:
        # 检查集合中是否有数据
        count = chat_collection.count()
        logger.info(f"ChromaDB中共有{count}条记录")

        if count == 0:
            logger.info("ChromaDB为空，无历史记录")
            return []

        try:
            # 查询所有数据，按时间戳排序
            all_data = chat_collection.get(
                include=["metadatas", "documents", "embeddings"]
            )

            logger.info(f"获取到所有数据: {len(all_data.get('ids', []))} 条记录")

            # 过滤出用户和助手的消息
            user_assistant_records = []
            for i, id in enumerate(all_data.get('ids', [])):
                if i < len(all_data['metadatas']):
                    metadata = all_data['metadatas'][i]
                    if isinstance(metadata, dict) and metadata.get('type') in ['user', 'assistant']:
                        user_assistant_records.append({
                            'id': id,
                            'type': metadata.get('type'),
                            'document': all_data['documents'][i] if i < len(all_data['documents']) else "",
                            'timestamp': metadata.get('timestamp', '')
                        })

            # 按时间戳排序
            user_assistant_records.sort(key=lambda x: x.get('timestamp', ''))

            # 提取最近的max_items条记录
            recent_records = user_assistant_records[-max_items * 2:] if len(
                user_assistant_records) > max_items * 2 else user_assistant_records

            # 构建聊天历史
            chat_history = []
            for record in recent_records:
                if record.get('type') == 'user':
                    chat_history.append({"role": "user", "content": record.get('document', '')})
                elif record.get('type') == 'assistant':
                    chat_history.append({"role": "assistant", "content": record.get('document', '')})

            logger.info(f"获取到 {len(chat_history)} 条有效聊天记录")
            print(chat_history)
            return chat_history

        except Exception as inner_e:
            logger.error(f"查询历史记录失败: {str(inner_e)}", exc_info=True)

            # 备选方案：使用query进行查询
            logger.info("尝试使用备选方案查询历史记录")
            results = chat_collection.query(
                query_texts=["conversation"],
                n_results=max_items * 2,
                include=["metadatas", "documents"],
                where={"type": {"$in": ["user", "assistant"]}}
            )

            # 记录查询结果
            logger.info(f"备选查询结果: IDs: {len(results.get('ids', []))} 条记录")

            # 提取聊天历史
            chat_history = []
            if results and results.get('ids') and len(results['ids']) > 0:
                for i in range(len(results['ids'])):
                    if i < len(results['metadatas']) and i < len(results['documents']):
                        metadata = results['metadatas'][i]
                        document = results['documents'][i]

                        if isinstance(metadata, dict) and metadata.get('type') == 'user':
                            chat_history.append({"role": "user", "content": document})
                        elif isinstance(metadata, dict) and metadata.get('type') == 'assistant':
                            chat_history.append({"role": "assistant", "content": document})

            logger.info(f"备选方案获取到 {len(chat_history)} 条有效聊天记录")
            print(chat_history)
            return chat_history

    except Exception as e:
        logger.error(f"获取聊天历史失败: {str(e)}", exc_info=True)
        return []


# 获取历史摘要作为长期记忆
def get_memory_summary():
    try:
        # 查询记忆摘要
        results = chat_collection.query(
            query_texts=["memory summary"],
            n_results=1,
            where={"type": "summary"}
        )

        if results and results.get('documents') and len(results['documents']) > 0:
            return results['documents'][0]
        return ""
    except Exception as e:
        logger.error(f"获取记忆摘要失败: {str(e)}")
        return ""


# 初始化会话消息
def initialize_chat_messages():
    # 获取记忆摘要
    memory_summary = get_memory_summary()
    system_prompt = BASE_PROMPT

    # 如果有记忆摘要，添加到系统提示中
    if memory_summary:
        system_prompt += f"\n\n【你与用户的互动记忆】\n{memory_summary}"

    # 初始化会话消息
    history_messages = [{"role": "system", "content": system_prompt}]

    # 添加最近的对话记录
    recent_history = get_chat_history()
    history_messages.extend(recent_history)

    return history_messages


# 全局会话消息
history_messages = initialize_chat_messages()

app = FastAPI()
app.mount("/videos", StaticFiles(directory="videos"), name="videos")
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"]
)

# 初始化模型 cuda:0 ， cpu 自行替换
model = AutoModel(
    model='iic/SenseVoiceSmall',
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cuda:0",
    disable_update=True
)


# 初始化OpenAI客户端
client_openai = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
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


# 音频持久化存储
async def save_audio_file(text, emotion):
    text

    """生成并持久化存储音频文件"""
    try:
        # 生成唯一文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{timestamp}_{uuid.uuid4().hex[:8]}_{emotion}.mp3"
        filepath = os.path.join(AUDIO_DIR, filename)

        # 生成TTS音频
        params = EMOTION_CONFIG.get(emotion, {})
        communicator = edge_tts.Communicate(
            text=text,
            voice="zh-CN-XiaoyiNeural",
            rate=params.get("rate", "+0%"),
            pitch=params.get("pitch", "+0Hz")
        )

        # 保存到持久化目录
        await communicator.save(filepath)

        # 返回相对URL
        return f"/static/audio/{filename}"
    except Exception as e:
        logger.error(f"音频持久化失败: {str(e)}")
        raise HTTPException(500, "音频生成服务异常")


# 存储聊天记录到ChromaDB
def store_chat_message(message_type, content, metadata=None):
    """存储聊天消息到ChromaDB"""
    try:
        # 生成唯一ID
        message_id = f"{message_type}_{uuid.uuid4().hex}"

        # 设置基础元数据
        base_metadata = {
            "type": message_type,
            "timestamp": datetime.datetime.now().isoformat()
        }

        # 合并附加元数据
        if metadata:
            base_metadata.update(metadata)

        # 重要：确保metadata中的所有值都是字符串类型
        for key, value in base_metadata.items():
            if not isinstance(value, str):
                base_metadata[key] = str(value)

        logger.info(f"准备存储消息: {message_type}, 内容: {content[:30]}..., 元数据: {base_metadata}")

        # 添加到集合
        chat_collection.add(
            ids=[message_id],
            documents=[content],
            metadatas=[base_metadata]
        )

        logger.info(f"成功存储消息: {message_type}, ID: {message_id}")

        # 打印当前集合大小
        try:
            count = chat_collection.count()
            logger.info(f"当前集合中有 {count} 条记录")
        except Exception as count_err:
            logger.error(f"获取集合大小失败: {str(count_err)}")

        # 管理集合大小，保留最近50条记录
        manage_collection_size(50)

        return message_id
    except Exception as e:
        logger.error(f"存储聊天消息失败: {str(e)}", exc_info=True)
        return None


# 管理集合大小
def manage_collection_size(max_items=50):
    """管理聊天历史记录数量，超过最大值时删除旧记录"""
    try:
        # 获取所有记录
        all_records = chat_collection.query(
            query_texts=["all"],
            n_results=1000,  # 获取所有记录
            where={"type": {"$in": ["user", "assistant"]}}
        )

        # 如果记录数超过最大值，删除最旧的记录
        if all_records and all_records.get('ids') and len(all_records['ids']) > max_items:
            # 按时间排序
            records = []
            for i in range(len(all_records['ids'])):
                if i < len(all_records['metadatas']):
                    metadata = all_records['metadatas'][i]
                    if isinstance(metadata, dict):
                        records.append({
                            'id': all_records['ids'][i],
                            'timestamp': metadata.get('timestamp', '')
                        })

            # 按时间戳排序
            records.sort(key=lambda x: x['timestamp'])

            # 删除最旧的记录
            to_delete = len(records) - max_items
            ids_to_delete = [r['id'] for r in records[:to_delete]]

            if ids_to_delete:
                chat_collection.delete(ids=ids_to_delete)

                # 同时删除相关的音频文件
                for record_id in ids_to_delete:
                    if record_id.startswith('assistant_'):
                        # 检查元数据中是否有音频路径
                        for i, id in enumerate(all_records['ids']):
                            if id == record_id and i < len(all_records['metadatas']):
                                metadata = all_records['metadatas'][i]
                                if isinstance(metadata, dict):
                                    audio_path = metadata.get('audio_path', '')
                                    if audio_path and os.path.exists(audio_path.replace('/static', 'static')):
                                        os.remove(audio_path.replace('/static', 'static'))
                                break
    except Exception as e:
        logger.error(f"管理集合大小失败: {str(e)}")


# 生成记忆摘要
async def generate_memory_summary():
    """使用LLM生成聊天记录的摘要作为长期记忆"""
    try:
        # 获取最近20条聊天记录
        recent_history = get_chat_history(20)

        if not recent_history:
            return

        # 构建摘要提示
        summary_prompt = """
        请总结以下对话历史的关键信息，以作为AI助手的长期记忆。
        摘要应包括：
        1. 用户提到的个人信息或偏好
        2. 重要的互动模式或情感倾向
        3. 讨论的主要话题

        请用简洁的第三人称方式总结（不超过200字）：
        """

        # 添加聊天历史到提示
        for message in recent_history:
            role = "用户" if message["role"] == "user" else "AI"
            summary_prompt += f"\n{role}: {message['content']}"

        # 调用LLM生成摘要
        response = client_openai.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个专业的对话摘要助手。"},
                {"role": "user", "content": summary_prompt}
            ],
            temperature=0.7
        )

        summary = response.choices[0].message.content

        # 存储摘要
        # 先删除旧摘要
        old_summaries = chat_collection.query(
            query_texts=["memory summary"],
            n_results=10,
            where={"type": "summary"}
        )

        if old_summaries and old_summaries.get('ids') and len(old_summaries['ids']) > 0:
            chat_collection.delete(ids=old_summaries['ids'])

        # 添加新摘要
        chat_collection.add(
            ids=[f"summary_{uuid.uuid4().hex}"],
            documents=[summary],
            metadatas=[{
                "type": "summary",
                "timestamp": datetime.datetime.now().isoformat()
            }]
        )

        return summary
    except Exception as e:
        logger.error(f"生成记忆摘要失败: {str(e)}")
        return None


@app.post("/process", response_model=ProcessingResponse)
async def process_audio(audio: UploadFile = File(...)):
    """端到端处理流程"""
    global history_messages

    try:
        # 创建临时文件读取上传的音频
        temp_audio_path = f"temp_{uuid.uuid4().hex}.wav"
        try:
            with open(temp_audio_path, "wb") as buffer:
                buffer.write(await audio.read())

            # 语音识别
            text = speech_to_text(temp_audio_path)
        finally:
            # 删除临时文件
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

        # 存储用户消息
        user_msg_id = store_chat_message("user", text)
        logger.info(f"存储用户消息: {text[:30]}...，ID: {user_msg_id}")

        # 情感分析
        emotion_data = await analyze_emotion(text)

        # 生成并保存音频文件
        audio_url = await save_audio_file(emotion_data["result"], emotion_data["emotion"])
        logger.info(f"生成音频文件: {audio_url}")

        # 存储AI响应
        ai_msg_id = store_chat_message("assistant", emotion_data["result"], {
            "emotion": emotion_data["emotion"],
            "audio_path": audio_url
        })
        logger.info(f"存储AI响应: {emotion_data['result'][:30]}...，ID: {ai_msg_id}")

        # 如果对话超过10轮，生成记忆摘要
        if len(history_messages) > 22:  # 系统消息+20条对话
            logger.info("开始生成记忆摘要")
            await generate_memory_summary()
            # 重新初始化会话
            history_messages = initialize_chat_messages()

        # 构造响应
        return {
            "raw_response": emotion_data["result"],
            "text": text,
            "emotion": emotion_data["emotion"],
            "audio_url": audio_url,
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
    await client_mcp.connect_to_sse_server()
    if text == "":
        return {
            "result": "你好像什么都没说。",
            "emotion": "伤心"
        }
    """情感分析"""
    try:
        global history_messages
        # 添加用户新提问
        history_messages.append({"role": "user", "content": text})
        logger.info(f"添加用户输入到历史，当前历史长度: {len(history_messages)}")

        # 自动截断（保留最近20轮对话）
        if len(history_messages) > 42:  # 系统消息+用户消息和AI回复共20轮
            # 保留系统消息和最近20轮对话
            history_messages = [history_messages[0]] + history_messages[-40:]
            logger.info("历史消息过长，已截断")

        # 输出当前历史消息
        for i, msg in enumerate(history_messages):
            truncated = msg['content'][:50] + "..." if len(msg['content']) > 50 else msg['content']
            logger.info(f"历史消息 #{i}: {msg['role']} - {truncated}")

        # 调用大模型
        logger.info("开始调用情感分析")

        # 获取所有 mcp 服务器 工具列表信息
        response = await client_mcp.session.list_tools()
        # 生成 function call 的描述信息
        available_tools = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema
            }
        } for tool in response.tools]

        # 请求 deepseek，function call 的描述信息通过 tools 参数传入

        response = client_openai.chat.completions.create(
            model="deepseek-chat",
            messages=history_messages,
            tools=available_tools
        )
        # 处理返回的内容
        content = response.choices[0]

        if content.finish_reason == "tool_calls":
            # 如何是需要使用工具，就解析工具
            tool_call = content.message.tool_calls[0]
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            # 执行工具
            result = await client_mcp.session.call_tool(tool_name, tool_args)
            print(f"\n\n[Calling tool {tool_name} with args {tool_args}]\n\n")
            # 将 deepseek 返回的调用哪个工具数据和工具执行完成后的数据都存入messages中
            history_messages.append(content.message.model_dump())
            history_messages.append({
                "role": "tool",
                "content": result.content[0].text,
                "tool_call_id": tool_call.id,
            })

        # 将上面的结果再返回给 deepseek 用于生产最终的结果
        response = client_openai.chat.completions.create(
            model="deepseek-chat",
            messages=history_messages
        )
        # 解析响应
        result = parse_llm_response(response.choices[0].message.content)
        logger.info(f"情感分析结果: {result.get('emotion')}")

        # 添加AI回复到历史
        history_messages.append({"role": "assistant", "content": result["result"]})

        return result
    except Exception as e:
        logger.error(f"情感分析失败: {str(e)}", exc_info=True)
        raise HTTPException(500, "情感分析服务异常")


def parse_llm_response(content: str) -> dict:
    """解析大模型响应"""
    try:
        if 'json' in content:
            pattern = r"```json\s*({.*?})\s*```"
            match = re.search(pattern, content, re.DOTALL)
            if match:
                json_content = match.group(1)
                return json.loads(json_content)

        # 如果无法解析为JSON，尝试直接提取emotion和result
        result = content
        emotion = "开心"  # 默认情绪

        return {
            "result": result,
            "emotion": emotion
        }
    except Exception as e:
        logger.error(f"解析响应失败: {str(e)}")
        return {
            "result": content,
            "emotion": "开心"
        }


@app.get("/chat_history")
async def get_history():
    """获取聊天历史接口"""
    try:
        # 检查数据库记录数
        count = chat_collection.count()
        logger.info(f"获取历史记录: 当前数据库中有 {count} 条记录")

        if count == 0:
            logger.info("数据库为空，返回空历史记录")
            return {"messages": []}

        try:
            # 获取所有记录
            all_data = chat_collection.get(
                include=["metadatas", "documents"]
            )

            logger.info(f"获取到原始记录: {len(all_data.get('ids', []))} 条")

            if not all_data or not all_data.get('ids'):
                logger.info("没有找到记录")
                return {"messages": []}

            # 提取记录
            records = []
            for i, id in enumerate(all_data.get('ids', [])):
                if i < len(all_data['metadatas']) and i < len(all_data['documents']):
                    metadata = all_data['metadatas'][i]
                    if isinstance(metadata, dict) and metadata.get('type') in ['user', 'assistant']:
                        records.append({
                            'id': id,
                            'type': metadata.get('type'),
                            'document': all_data['documents'][i],
                            'metadata': metadata,
                            'timestamp': metadata.get('timestamp', '')
                        })

            logger.info(f"过滤后的记录数: {len(records)}")

            # 记录类型分布情况
            user_count = sum(1 for r in records if r.get('type') == 'user')
            assistant_count = sum(1 for r in records if r.get('type') == 'assistant')
            logger.info(f"记录类型分布: user={user_count}, assistant={assistant_count}")

            # 按时间戳排序
            records.sort(key=lambda x: x.get('timestamp', ''))

            # 组织对话对 - 寻找连续的user/assistant对
            messages = []
            i = 0
            while i < len(records) - 1:
                if records[i].get('type') == 'user':
                    # 查找下一个assistant记录
                    for j in range(i + 1, len(records)):
                        if records[j].get('type') == 'assistant':
                            messages.append({
                                "user_input": records[i]['document'],
                                "ai_response": records[j]['document'],
                                "emotion": records[j]['metadata'].get('emotion', '开心'),
                                "audio_path": records[j]['metadata'].get('audio_path', '')
                            })
                            i = j + 1  # 跳过已匹配的记录
                            break
                    else:
                        # 未找到匹配的assistant记录
                        i += 1
                else:
                    i += 1

            logger.info(f"最终组织的对话对数: {len(messages)}")

            return {"messages": messages}

        except Exception as inner_e:
            logger.error(f"获取历史记录内部错误: {str(inner_e)}", exc_info=True)

            # 备选方案
            logger.info("尝试备选方案获取历史记录")
            results = chat_collection.query(
                query_texts=["recent conversation"],
                n_results=50,  # 增加数量以确保获取足够的记录
                include=["metadatas", "documents"]
            )

            messages = []

            # 确保结果有效
            if not results or not results.get('ids') or len(results['ids']) == 0:
                logger.info("备选查询结果为空")
                return {"messages": []}

            # 按timestamp排序
            records = []
            for i in range(len(results['ids'])):
                if i < len(results['metadatas']) and i < len(results['documents']):
                    metadata = results['metadatas'][i]
                    if isinstance(metadata, dict):
                        records.append({
                            'id': results['ids'][i],
                            'type': metadata.get('type'),
                            'document': results['documents'][i],
                            'metadata': metadata,
                            'timestamp': metadata.get('timestamp', '')
                        })

            # 按时间戳排序
            records.sort(key=lambda x: x['timestamp'])
            logger.info(f"备选排序后的记录数: {len(records)}")

            # 组织对话对 - 寻找连续的user/assistant对
            i = 0
            while i < len(records) - 1:
                if records[i].get('type') == 'user':
                    # 查找下一个assistant记录
                    for j in range(i + 1, len(records)):
                        if records[j].get('type') == 'assistant':
                            messages.append({
                                "user_input": records[i]['document'],
                                "ai_response": records[j]['document'],
                                "emotion": records[j]['metadata'].get('emotion', '开心'),
                                "audio_path": records[j]['metadata'].get('audio_path', '')
                            })
                            i = j + 1  # 跳过已匹配的记录
                            break
                    else:
                        # 未找到匹配的assistant记录
                        i += 1
                else:
                    i += 1

            logger.info(f"备选方案最终组织的对话对数: {len(messages)}")

            return {"messages": messages}

    except Exception as e:
        logger.error(f"获取历史记录失败: {str(e)}", exc_info=True)
        return {"messages": []}


# 检查集合是否有数据，如果为空，添加一条示例数据
def ensure_collection_has_data():
    """确保集合中有数据，防止首次查询失败"""
    try:
        count = chat_collection.count()
        logger.info(f"确认集合数据: 当前有 {count} 条记录")

        if count == 0:
            logger.info("集合为空，添加初始数据")
            # 添加初始系统消息作为哨兵记录
            chat_collection.add(
                ids=["system_init"],
                documents=["系统初始化消息"],
                metadatas=[{
                    "type": "system",
                    "timestamp": datetime.datetime.now().isoformat()
                }]
            )
            logger.info("已添加初始系统消息")
    except Exception as e:
        logger.error(f"检查集合数据失败: {str(e)}", exc_info=True)


# 确保集合有数据
ensure_collection_has_data()


# 调试路由 - 列出所有记录
@app.get("/debug/list_all_records")
async def list_all_records():
    """调试接口：列出所有记录"""
    try:
        try:
            all_data = chat_collection.get(
                include=["metadatas", "documents"]
            )

            records = []
            for i, id in enumerate(all_data.get('ids', [])):
                if i < len(all_data['metadatas']) and i < len(all_data['documents']):
                    metadata = all_data['metadatas'][i]
                    records.append({
                        'id': id,
                        'type': metadata.get('type') if isinstance(metadata, dict) else None,
                        'document_preview': all_data['documents'][i][:50] + "..." if len(
                            all_data['documents'][i]) > 50 else all_data['documents'][i],
                        'timestamp': metadata.get('timestamp', '') if isinstance(metadata, dict) else None
                    })

            return {
                "record_count": len(records),
                "records": records
            }
        except Exception as e:
            return {"error": str(e), "record_count": 0, "records": []}
    except Exception as e:
        return {"error": str(e), "record_count": 0, "records": []}


# 调试路由 - 添加测试数据
@app.get("/debug/add_test_data")
async def add_test_data():
    """调试接口：添加测试数据"""
    try:
        # 添加用户消息
        user_id = store_chat_message("user", "你好，今天天气真好！")

        # 添加助手消息
        assistant_id = store_chat_message("assistant", "是的呢！阳光明媚的天气总是让人心情愉快~有什么我能帮你的吗？", {
            "emotion": "开心",
            "audio_path": "/static/audio/test_happy.mp3"
        })

        return {
            "success": True,
            "user_id": user_id,
            "assistant_id": assistant_id
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# 调试路由 - 清空所有数据
@app.get("/debug/clear_all_data")
async def clear_all_data():
    global chat_collection
    """调试接口：清空所有数据"""
    try:
        # 获取所有记录ID
        try:
            all_data = chat_collection.get(include=["ids"])
            if all_data and all_data.get('ids'):
                chat_collection.delete(ids=all_data['ids'])

                # 添加初始系统消息作为哨兵记录
                chat_collection.add(
                    ids=["system_init"],
                    documents=["系统初始化消息"],
                    metadatas=[{
                        "type": "system",
                        "timestamp": datetime.datetime.now().isoformat()
                    }]
                )

                # 清理音频文件
                try:
                    for file in os.listdir(AUDIO_DIR):
                        file_path = os.path.join(AUDIO_DIR, file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                except Exception as file_e:
                    logger.error(f"清理音频文件失败: {str(file_e)}")

                return {"success": True, "deleted_count": len(all_data['ids']), "message": "聊天记录已清空"}
        except Exception as inner_e:
            # 如果获取失败，尝试重新创建集合
            logger.error(f"清空数据失败: {str(inner_e)}")

            try:
                client.delete_collection("chat_history")
                chat_collection = client.create_collection(
                    name="chat_history",
                    embedding_function=embedding_function,
                    metadata={"description": "存储聊天历史记录"}
                )

                # 添加初始系统消息作为哨兵记录
                chat_collection.add(
                    ids=["system_init"],
                    documents=["系统初始化消息"],
                    metadatas=[{
                        "type": "system",
                        "timestamp": datetime.datetime.now().isoformat()
                    }]
                )

                return {"success": True, "message": "已重新创建集合并清空聊天记录"}
            except Exception as recreate_e:
                return {"success": False, "error": f"重新创建集合失败: {str(recreate_e)}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# 用户友好的清空历史API
@app.get("/clear_history")
async def clear_history():
    """清空聊天历史记录API"""
    global chat_collection, history_messages

    try:
        # 获取所有记录ID
        try:
            # 重置历史消息
            history_messages = initialize_chat_messages()

            # 清空数据库记录
            all_data = chat_collection.get()
            if all_data and all_data.get('ids'):
                # 保留system_init记录
                non_system_ids = [id for id in all_data['ids'] if id != "system_init"]
                if non_system_ids:
                    chat_collection.delete(ids=non_system_ids)

                # 清理音频文件
                try:
                    audio_files_count = 0
                    for file in os.listdir(AUDIO_DIR):
                        file_path = os.path.join(AUDIO_DIR, file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                            audio_files_count += 1
                    history_messages = []
                    return {
                        "success": True,
                        "deleted_records": len(non_system_ids),
                        "deleted_files": audio_files_count,
                        "message": "聊天记录已清空"
                    }
                except Exception as file_e:
                    logger.error(f"清理音频文件失败: {str(file_e)}")
                    return {
                        "success": True,
                        "deleted_records": len(non_system_ids),
                        "message": "聊天记录已清空，但清理音频文件失败"
                    }

            return {"success": True, "message": "没有找到需要清空的聊天记录"}

        except Exception as inner_e:
            logger.error(f"清空历史记录失败: {str(inner_e)}", exc_info=True)
            return {"success": False, "error": f"清空历史记录失败: {str(inner_e)}"}
    except Exception as e:
        logger.error(f"清空历史记录API错误: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


# mcp de
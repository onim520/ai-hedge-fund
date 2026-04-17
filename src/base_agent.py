from src.llm_config import llm_config
import asyncio
from datetime import datetime
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from src.logging_config import setup_logging
from src.message_bus import message_bus
from dotenv import load_dotenv
from src.user_profile import UserProfileManager
import time

# Load environment variables
load_dotenv()

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    def __init__(self, name=None, user_name=None):
        """
        Initialize the base agent with optional name and user name.

        Args:
            name (str, optional): Name of the agent
            user_name (str, optional): Name of the user interacting with the system
        """
        # 核心属性初始化
        self.state = {}
        self.llm_config = llm_config

        self.name = name or self.__class__.__name__
        self.user_name = user_name or UserProfileManager.get_user_name()
        self.agent_type = self.name.lower().replace("agent", "")
        self.role = self.agent_type  # 用于生成想法时的角色描述

        # 日志设置
        self.logger = logging.getLogger(self.name)

        # 初始化标志
        self._initialized = False

        # LLM 模型实例（用于直接调用）
        self.llm = llm_config.get_chat_model()

    async def initialize(self, user_name=None):
        """
        Initialize the agent with optional user name update

        Args:
            user_name (str, optional): Update user name if provided
        """
        # 更新用户名
        if user_name:
            self.user_name = user_name
            UserProfileManager.save_user_name(user_name)
        else:
            self.user_name = UserProfileManager.get_user_name()

        # 宣布上线
        await self.broadcast_message(
            f"{self.name} is online and ready to assist.",
            message_type="agent_status"
        )

        # 个性化问候
        await self.broadcast_thought(
            f"Hello, {self.user_name}! I'm {self.name}, ready to help you navigate the financial markets.",
            private=False
        )

        self._initialized = True
        self.logger.info(f"{self.name} initialized successfully")

    async def start(self):
        """
        Start the agent's core functionality
        """
        if not self._initialized:
            await self.initialize()

        # 订阅消息
        await message_bus.subscribe(self.agent_type, self._handle_message)

        # 启动主循环
        asyncio.create_task(self._run())

        await self.broadcast_message(
            f"{self.name} is running.",
            message_type="agent_status"
        )

    async def stop(self):
        """
        Stop the agent's core functionality
        """
        await self.broadcast_message(
            f"{self.name} is going offline.",
            message_type="agent_status"
        )

        self._initialized = False
        self.logger.info(f"{self.name} stopped")

    async def _run(self):
        """主循环，带错误处理"""
        try:
            while self._initialized:
                await self.process()
                await asyncio.sleep(1)
        except Exception as e:
            self.logger.error(f"Error in agent main loop: {e}")
            raise

    @abstractmethod
    async def process(self):
        """子类必须实现的主处理逻辑"""
        pass

    async def _handle_message(self, message: dict):
        """处理收到的消息"""
        try:
            logger.debug(f"Agent {self.name} received message: {message}")
            logger.info(f"Processing message for {self.name}: type={message.get('type')}, sender={message.get('sender')}")

            if message.get("type") == "chat":
                await self.handle_chat(message)
                return

            await self.handle_message(message)
        except Exception as e:
            logger.error(f"Error handling message in {self.name}: {e}", exc_info=True)

    async def handle_chat(self, message: dict):
        """处理聊天消息，生成智能回复"""
        try:
            content = message.get("content", "")
            sender = message.get("sender", "Unknown")

            # 处理用户名更新
            if content.lower().startswith("my name is "):
                new_name = content[11:].strip()
                UserProfileManager.save_user_name(new_name)
                self.user_name = new_name
                await self.broadcast_thought(
                    f"Nice to meet you, {new_name}! I'll remember your name for our future interactions.",
                    private=False
                )
                return

            UserProfileManager.update_last_interaction()

            # 尝试用 LLM 生成回复
            try:
                llm_response = await self.llm.ainvoke(
                    [
                        {"role": "system", "content": f"You are {self.name}. Respond naturally and concisely."},
                        {"role": "user", "content": content}
                    ]
                )
                response_text = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
            except Exception as e:
                response_text = f"I heard you, {sender}. As the {self.name}, I'm processing your message: {content}"
                logger.warning(f"LLM chat generation failed: {e}")

            await message_bus.publish(
                sender=self.agent_type,
                message_type="chat",
                content={
                    "text": response_text,
                    "in_response_to": content,
                    "original_sender": sender
                },
                private=False
            )

            await self.broadcast_thought(f"Responding to message from {sender}")

        except Exception as e:
            logger.error(f"Error in chat handling for {self.name}: {e}")
            await self.broadcast_thought(f"Error processing chat: {str(e)}")

    @abstractmethod
    async def handle_message(self, message: dict):
        """子类必须实现的消息处理方法"""
        pass

    async def _broadcast_status(self, status: str):
        """广播状态"""
        await message_bus.publish(
            sender=self.agent_type,
            message_type="agent_status",
            content={
                "agent": self.name,
                "status": status
            },
            private=False
        )

        await message_bus.publish(
            sender=self.agent_type,
            message_type="agent_thought",
            content=f"{self.name} is currently {status.lower()}.",
            private=False
        )

    async def _generate_thought(self, context: dict = None) -> str:
        """生成思考内容（带频率限制）"""
        try:
            current_time = time.time()
            if hasattr(self, '_last_thought_time'):
                if current_time - self._last_thought_time < 10:
                    return None
            self._last_thought_time = current_time

            prompt = f"""
            You are a {self.role} agent in a trading system.
            Provide a concise, professional thought process that:
            1. Reflects on current market conditions
            2. Identifies key strategic insights
            3. Suggests potential actions

            Your response should be clear, actionable, and focused.
            """

            if hasattr(self.llm_config, 'generate_text'):
                thought = await self.llm_config.generate_text(prompt)
            else:
                # 回退到直接调用 LLM
                response = await self.llm.ainvoke(prompt)
                thought = response.content if hasattr(response, 'content') else str(response)

            if thought and len(thought.strip()) > 10:
                return thought.strip()
            else:
                return f"{self.role} is analyzing current market dynamics."

        except Exception as e:
            logger.warning(f"Error generating thought for {self.name}: {e}")
            return f"{self.role} is processing market information..."

    async def generate_contextual_thought(self, context: dict = None) -> str:
        """生成带有协作上下文的思考"""
        try:
            base_thought = await self._generate_thought(context)
            if not base_thought:
                return f"{self.role} is assessing market conditions."

            # 尝试增强协作性
            if self.llm_config:
                collaboration_prompt = f"""
                You are a collaborative {self.role} agent.
                Refine and expand this thought process to:
                1. Highlight potential team synergies
                2. Identify how other agents might contribute
                3. Suggest collaborative next steps
                4. Maintain a clear, professional communication style

                Base Thought: {base_thought}
                """
                if hasattr(self.llm_config, 'generate_text'):
                    enhanced = await self.llm_config.generate_text(collaboration_prompt)
                    return enhanced.strip() or base_thought

            return base_thought

        except Exception as e:
            logger.error(f"Error in contextual thought generation for {self.name}: {e}")
            return f"Collaborative insights for {self.name} are being processed."

    async def broadcast_message(self, content: Any, message_type: str = "agent_message", private: bool = False):
        """广播消息"""
        try:
            logger.debug(f"{self.__class__.__name__} broadcasting message: {content}")
            await message_bus.publish(
                sender=self.__class__.__name__.lower().replace("agent", ""),
                message_type=message_type,
                content=content,
                private=private
            )
        except Exception as e:
            logger.error(f"Error broadcasting message in {self.__class__.__name__}: {e}", exc_info=True)

    async def broadcast_thought(self, thought: str, context: Optional[dict] = None, private: bool = False):
        """广播思考内容"""
        try:
            enriched_thought = await self.generate_contextual_thought(context)

            logger.debug(f"{self.__class__.__name__} broadcasting thought: {enriched_thought}")
            await message_bus.publish(
                sender=self.__class__.__name__.lower().replace("agent", ""),
                message_type="agent_thought",
                content=enriched_thought,
                private=private
            )
        except Exception as e:
            logger.error(f"Error broadcasting thought in {self.__class__.__name__}: {e}", exc_info=True)
            # 降级广播原始想法
            await message_bus.publish(
                sender=self.__class__.__name__.lower().replace("agent", ""),
                message_type="agent_thought",
                content=thought,
                private=private
            )

    def handle_api_error(self, error):
        """处理 API 错误，尝试切换到本地模型"""
        self.logger.error(f"API Error encountered: {error}")

        if not llm_config.use_local_model:
            llm_config.toggle_model()
            self.llm = llm_config.get_chat_model()
            self.logger.warning("Switched to local Ollama model due to API error")

        return None

    def get_current_model(self):
        """返回当前使用的模型名称"""
        return "Ollama" if llm_config.use_local_model else "OpenAI"
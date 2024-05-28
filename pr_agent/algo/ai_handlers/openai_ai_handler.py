from pr_agent.algo.ai_handlers.base_ai_handler import BaseAiHandler
import openai
from openai import AsyncOpenAI
from openai import APIError, RateLimitError, APITimeoutError
from retry import retry

from pr_agent.config_loader import get_settings
from pr_agent.log import get_logger

OPENAI_RETRIES = 5


class OpenAIHandler(BaseAiHandler):
    def __init__(self):
        # Initialize OpenAIHandler specific attributes here
        try:
            super().__init__()
            openai.api_key = get_settings().openai.key
            if get_settings().get("OPENAI.ORG", None):
                openai.organization = get_settings().openai.org
            if get_settings().get("OPENAI.API_TYPE", None):
                if get_settings().openai.api_type == "azure":
                    self.azure = True
                    openai.azure_key = get_settings().openai.key
            if get_settings().get("OPENAI.API_VERSION", None):
                openai.api_version = get_settings().openai.api_version
            if get_settings().get("OPENAI.API_BASE", None):
                openai.base_url = get_settings().openai.api_base

        except AttributeError as e:
            raise ValueError("OpenAI key is required") from e
        
        self._client = AsyncOpenAI(
            base_url=openai.base_url,
            api_key=openai.api_key
        )
    @property
    def deployment_id(self):
        """
        Returns the deployment ID for the OpenAI API.
        """
        return get_settings().get("OPENAI.DEPLOYMENT_ID", None)
    
    @retry(exceptions=(APIError, APITimeoutError, AttributeError, RateLimitError),
           tries=OPENAI_RETRIES, delay=2, backoff=2, jitter=(1, 3))
    async def chat_completion(self, model: str, system: str, user: str, temperature: float = 0.2):
        try:
            # deployment_id = self.deployment_id
            get_logger().info("System: ", system)
            get_logger().info("User: ", user)
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

            chat_completion = await self._client.chat.completions.create(
                model=model,
                # deployment_id=deployment_id,
                messages=messages,
                temperature=temperature,
                max_tokens=2560
            )
            resp = chat_completion.choices[0].message.content
            finish_reason = chat_completion.choices[0].finish_reason
            usage = chat_completion.usage
            get_logger().info("AI response", response=resp, messages=messages, finish_reason=finish_reason,
                            model=model, usage=usage)
            get_logger().info(f"resp: {resp}")
            return resp, finish_reason       
        except (APIError, APITimeoutError) as e:
            get_logger().error("Error during OpenAI inference: ", e)
            raise
        except (Exception) as e:
            get_logger().error("Unknown error during OpenAI inference: ", e)
            raise

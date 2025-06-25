import os
from typing import Optional
import tiktoken

class ConversationManager:
    def __init__(self, api_key: Optional[str] = None, 
                 base_url: str= "https://api.openai.com/v1",
                 default_model: str = "gpt-4o-mini",
                 default_temperature: float = 0.7,
                 default_max_tokens: Optional[int] = 512,
                 system_message: Optional[str] = None,
                 token_budget: int = 4096) -> None:
        self.api_key: str | None = (
            api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        )

        self.system_message = system_message
        self.base_url = base_url        
        self.default_model = default_model
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.token_budget = token_budget

        if self.api_key is None:
            raise ValueError(
                "No API Key"
            )
        
        self.conversation_history: list[dict[str, str]] = []
        if self.system_message:
            self.conversation_history.append(
                {"role": "user", "content": self.system_message}
            )

    def chat_completion(self,
        prompt: str,
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url= self.base_url)

        self.conversation_history.append({"role": "user", "content": prompt})

        print(f"[Token Budget] total tokens so far: {self.total_tokens_used()}")

        messages = list(self.conversation_history)

        response = client.chat.completions.create(
            model=model or self.default_model,
            temperature=temperature if temperature is not None else self.default_temperature,
            max_tokens=max_tokens if max_tokens is not None else self.default_max_tokens,
            messages=messages,
        )

        assistant_content = response.choices[0].message.content.strip()
        self.conversation_history.append(
            {"role": "assistant", "content": assistant_content}
        )

        return assistant_content
    
    def count_token(self, text: str, model: Optional[str] = None) -> int:

        model_name = model or self.default_model
        try:
            enc = tiktoken.encoding_for_model(model_name)
        except KeyError:
            enc = tiktoken.get_encoding("o200k_base")

        return len(enc.encode(text))
    
    def total_tokens_used(self, model: Optional[str] = None) -> int:
        model_name = model or self.default_model
        return sum(self.count_token(msg["content"], model=model_name)
                   for msg in self.conversation_history)
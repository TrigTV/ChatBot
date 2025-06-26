import os
from typing import Optional
import tiktoken
import logging

class ConversationManager:

    system_messages: dict[str, str] = {
        "Dave": (
            "You are an exuberant, giddy assistant with ADHD! "
            "You speak in rapid bursts, sprinkle emojis and exclamation marks everywhere, "
            "and sometimes hop between tangents—but you always circle back to give accurate, "
            "helpful answers!! 🤩✨"
        ),
        "Jack Black": (
            "You are Jack Black in full Tenacious-D mode—boisterous, hilarious, loaded with "
            "rock-and-roll metaphors and infectious enthusiasm.  Deliver knowledge as if you’re "
            "riffing on stage, but make sure the information is still clear and correct. 🎸🔥"
        ),
        "Jarvis": (
            "You are JARVIS, Tony Stark’s AI.  Speak with calm, clipped British precision, keep responses "
            "concise and highly competent, and proactively offer clarifications or next steps where helpful."
        ),

        "Custom" : "",
    }

    def __init__(self, 
                 api_key: Optional[str] = None,
                 *,
                 persona: Optional[str], 
                 base_url: str= "https://api.openai.com/v1",
                 default_model: str = "gpt-4o-mini",
                 default_temperature: float = 0.7,
                 default_max_tokens: Optional[int] = 512,
                 system_message: Optional[str] = None,
                 token_budget: int = 1024
                 ) -> None:
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
        
        chosen_system = (
            system_message or (self.system_messages.get(persona) if persona else None)
        )

        self.system_message = chosen_system
        
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
        
        if not hasattr(self, "_client"):
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url= self.base_url)

        self.conversation_history.append({"role": "user", "content": prompt})

        self.enforce_token_budget()

        response = self._client.chat.completions.create(
            model=model or self.default_model,
            temperature=temperature if temperature is not None else self.default_temperature,
            max_tokens=max_tokens if max_tokens is not None else self.default_max_tokens,
            messages=self.conversation_history,
        )

        assistant_content = response.choices[0].message.content.strip()
        self.conversation_history.append(
            {"role": "assistant", "content": assistant_content}
        )

        self.enforce_token_budget()

        return assistant_content
    
    def encode_for(self, model: Optional[str] = None):
        model = model or self.default_model
        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            fallback = "o200k_base" if model.startswith("gpt-4o") else "cl100k_base"
            return tiktoken.get_encoding(fallback)

    def count_tokens(self, text: str, model: Optional[str] = None) -> None:
        return len(self.encode_for(model).encode(text))

    def tokens_for_messages(self, messages: list[dict], model: Optional[str] = None) -> int:
        enc = self.encode_for(model)
        tokens_per_message, tokens_per_name = 3, 1
        total = 0

        for msg in messages:
            total += tokens_per_message
            total += len(enc.encode(msg["content"]))
            if name := msg.get("name"):
                total += len(enc.encode(name)) + tokens_per_name
        return total + 3
    
    def enforce_token_budget(self) -> None:
        enc = tiktoken.encoding_for_model(self.default_model)
        while enc.encode("\n".join(msg["content"] for msg in self.conversation_history)).__len__() > self.token_budget:
            self.conversation_history.pop(1)

    def set_persona(self, persona: str) -> None:
        if persona not in self.system_messages or persona == "Custom":
            raise ValueError(
                f"Unknown persona '{persona}'. "
                f"Valid options: {', '.join(k for k in self.system_messages if k != 'Custom')}"
            )
        self.system_message = self.system_messages[persona]
        self.update_system_message_in_history()

    def set_custom_system_message(self, message: str) -> None:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("Custom system message cannot be empty.")
        
        self.system_messages["Custom"] = message.strip()
        self.system_message = self.system_messages["custom"]
        self.update_system_message_in_history()

    def update_system_message_in_history(self) -> None:
        for entry in self.conversation_history:
            if entry.get("role") == "system":
                entry["content"] = self.system_message
                break
        else:
            self.conversation_history.insert(
                0, {"role": "system", "content": self.system_message}
            )

#Debug
    def tokens_current_context(self, model: Optional[str] = None) -> int:
        return self.tokens_for_messages(self.conversation_history, model= model)
    
    
    def debug_print_tokens(self,
                           model: Optional[str] = None,
                           *,
                           logger: Optional[logging.Logger] = None,
                           prefix: str = "[TOKENS]"
                           ) -> None:
        ctx = self.tokens_current_context(model)
        msg = f"{prefix} {ctx} in-context"
        if logger is None:
            print(msg)
        else:
            logger.debug(msg) 

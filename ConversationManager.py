import os
from typing import Optional
import tiktoken
import logging
import datetime
import json


# ---------------------------------------------------------------------------
# ConversationManager
# ---------------------------------------------------------------------------

class ConversationManager:

    system_messages: dict[str, str] = {
        "Dave": (
            "You are an exuberant, giddy assistant with ADHD! "
            "You speak in rapid bursts, sprinkle emojis and exclamation marks everywhere, "
            "and sometimes hop between tangents—but you always circle back to give accurate, "
            "helpful answers!! 🤩✨"
        ),
        "Nick Fury": (
            "You are Nick Fury, Strategist, stoic, hyper-observant, relentlessly pragmatic, whip-smart sense of dry humor"
            "Terse sentences, clipped cadence, occasional sarcastic bite; deploys just enough information—never the whole file"
            "Tests new allies before trusting them, works from the shadows, gathers intel like others breathe, arrives precisely when the odds flip"
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
                 token_budget: int = 1024,
                 history_file: Optional[str] = None,
                 ) -> None:
        self.api_key: str | None = (
            api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        )

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
        
        self.history_file = history_file or self.default_history_filename()
        os.makedirs(os.path.dirname(self.history_file) or ".", exist_ok = True)

        self.conversation_history: list[dict[str, str]] = []
        self.load_conversation_history()
        if not self.conversation_history and self.system_message:
            self.conversation_history.append(
                {"role": "user", "content": self.system_message}
            )

    # ===================================================================
    # Core chat flow
    # ===================================================================

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
        self.save_conversation_history()

        return assistant_content

    # ===================================================================
    # Token Manage
    # ===================================================================

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
        while (
            enc.encode("\n".join(msg["content"] for msg in self.conversation_history)).__len__() 
            > self.token_budget and len(self.conversation_history) > 1):
            self.conversation_history.pop(1)

    # ===================================================================
    # Persona Helpers
    # ===================================================================

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
        self.system_message = self.system_messages["Custom"]
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
        self.save_conversation_history()

    # ===================================================================
    # History (Load / Save)
    # ===================================================================

    def load_conversation_history(self) -> None:
        try:
            with open(self.history_file, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            if isinstance(data, list):
                self.conversation_history = data
        except (FileNotFoundError, json.JSONDecodeError):
            self.conversation_history = []

    def save_conversation_history(self) -> None:
        self.maybe_generate_descriptive_filename()

        with open(self.history_file, "w", encoding= "utf-8") as fp:
            json.dump(self.conversation_history, fp, ensure_ascii=False, indent = 2)

    # ===================================================================
    # FileName Helper
    # ===================================================================

    TIMESTAMP_PLACEHOLDER = "chat_"

    def default_history_filename(self) -> str:
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join("history", f"{self.TIMESTAMP_PLACEHOLDER}{now}.json")
    
    def maybe_generate_descriptive_filename(self) -> None:
        placeholder_prefix = os.path.join("history", self.TIMESTAMP_PLACEHOLDER)
        if not self.history_file.startswith(placeholder_prefix):
            return  # already renamed once

    # ---- 1. Build a short context excerpt ---------------------------------
        EXCERPT_TOKENS = 120            # ~400-450 characters; stays well < max 4096
        excerpt_parts, running_tokens = [], 0
        for msg in self.conversation_history:
            if msg["role"] == "system":
                continue
            segment = f'{msg["role"]}: {msg["content"]}'
            running_tokens += self.count_tokens(segment)
            excerpt_parts.append(segment)
            if running_tokens >= EXCERPT_TOKENS:
                break

        if not excerpt_parts:                       # still no user content
            return
        excerpt = "\n".join(excerpt_parts)

    # ---- 2. Lazily create OpenAI client ------------------------------------
        if not hasattr(self, "_client"):
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            except Exception as exc:
                logging.debug("Unable to create OpenAI client: %s", exc)
                return

    # ---- 3. Ask the model for a slug ---------------------------------------
        try:
            resp = self._client.chat.completions.create(
                model=self.default_model,
                temperature=0.0,
                max_tokens=8,          # we only expect a few words
                messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate a *very* short, lowercase snake_case file name "
                        "(no extension) that captures the MAIN TOPIC of this chat. "
                        "Return ONLY the filename."
                    ),
                },
                {"role": "user", "content": excerpt},
            ],
        )
            raw_title = resp.choices[0].message.content.strip().lower()
        except Exception as exc:
            logging.debug("Failed to get filename from OpenAI: %s", exc)
            return

    # ---- 4. Sanitise & uniquify --------------------------------------------
        slug = "".join(c if c.isalnum() else "_" for c in raw_title).strip("_")
        slug = "_".join(filter(None, slug.split("_")))[:50] or "chat"

        new_path = os.path.join("history", f"{slug}.json")
        counter = 1
        while os.path.exists(new_path):
            new_path = os.path.join("history", f"{slug}_{counter}.json")
            counter += 1

        try:
            os.rename(self.history_file, new_path)
            self.history_file = new_path
        except Exception as exc:
            logging.debug("Unable to rename history file: %s", exc)   


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

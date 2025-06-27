from __future__ import annotations

import glob
import os
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from dotenv import load_dotenv, find_dotenv
from ConversationManager import ConversationManager

load_dotenv("OPENAI_API_KEY.env")

HISTORY_DIR = "history"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# -----------------------------------------------------------------------------
# GUI application
# -----------------------------------------------------------------------------

class ChatGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Chatbot9000")
        self.geometry("900x600")

        self.cm: ConversationManager | None = None
        self._history_files: list[str] = []

        self.build_layout()
        self.refresh_history_list()

    def build_layout(self) -> None:
        sidebar = ttk.Frame(self, padding=8)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)

        persona_label = ttk.Label(sidebar, text="Persona:")
        persona_label.pack(anchor=tk.W)

        self.persona_var = tk.StringVar(value="Choose…")
        self.persona_combo = ttk.Combobox(
            sidebar,
            textvariable=self.persona_var,
            state="readonly",
            width=20,
            values=list(ConversationManager.system_messages.keys()),
        )
        self.persona_combo.pack(fill=tk.X)
        self.persona_combo.bind("<<ComboboxSelected>>", self._on_persona_change)

        ttk.Separator(sidebar, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)

        hist_label = ttk.Label(sidebar, text="Saved chats:")
        hist_label.pack(anchor=tk.W)

        self.hist_list = tk.Listbox(sidebar, height=15, activestyle="dotbox")
        self.hist_list.pack(fill=tk.BOTH, expand=True)

        hist_btn_frame = ttk.Frame(sidebar)
        hist_btn_frame.pack(fill=tk.X, pady=4)
        ttk.Button(hist_btn_frame, text="Load", command=self._load_selected_history).pack(
            side=tk.LEFT, expand=True, fill=tk.X
        )
        ttk.Button(hist_btn_frame, text="New", command=self._start_new_chat).pack(
            side=tk.LEFT, expand=True, fill=tk.X
        )

        main = ttk.Frame(self, padding=8)
        main.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.chat_display = tk.Text(main, state="disabled", wrap="word")
        self.chat_display.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        entry_frame = ttk.Frame(main)
        entry_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.user_entry = ttk.Entry(entry_frame)
        self.user_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        self.user_entry.bind("<Return>", lambda _e: self._on_send())

        send_btn = ttk.Button(entry_frame, text="Send", command=self._on_send)
        send_btn.pack(side=tk.RIGHT)

    # -----------------------------------------------------------------------------
    # File History
    # -----------------------------------------------------------------------------
    def refresh_history_list(self) -> None:
        os.makedirs(HISTORY_DIR, exist_ok=True)
        files = sorted(glob.glob(os.path.join(HISTORY_DIR, "*.json")))
        self._history_files = files

        self.hist_list.delete(0, tk.END)
        for f in files:
            self.hist_list.insert(tk.END, os.path.basename(f))

    def load_selected_history(self) -> None:
        selection = self.hist_list.curselection()
        if not selection:
            messagebox.showinfo("No selection", "Please select a history file from the list.")
            return

        file_path = self._history_files[selection[0]]
        self.init_cm(history_file=file_path)
        self.cm.load_conversation_history()
        self.redraw_chat_display()

    def start_new_chat(self) -> None:
        self.init_cm(history_file=None)
        self.redraw_chat_display()
        self.refresh_history_list()

    # -----------------------------------------------------------------------------
    # Persona Handling
    # -----------------------------------------------------------------------------
    def on_persona_change(self, _event: object = None) -> None:
        sel = self.persona_var.get()

        if sel == "Custom":
            custom_msg = simpledialog.askstring("Custom Persona", "Enter system message:")
            if custom_msg is None:
                return

            if self.cm is None:
                self.init_cm(history_file=None, custom_msg=custom_msg)
            else:
                self.cm.set_custom_system_message(custom_msg)

            self.append_system_line("[Custom persona applied]")
            return

        if sel in ("", "Choose…"):
            return

        if self.cm is None:
            return

        try:
            self.cm.set_persona(sel)
        except ValueError as exc:
            messagebox.showerror("Persona Error", str(exc))
            return

        self.append_system_line(f"[Persona switched to {sel}]")

    def on_send(self) -> None:
        text = self.user_entry.get().strip()
        if not text or self.cm is None:
            return

        self.user_entry.delete(0, tk.END)
        self.append_chat_line("You", text)

        try:
            ai_text = self.cm.chat_completion(text)
        except Exception as exc: 
            messagebox.showerror("Error", f"API call failed:\n{exc}")
            return

        self.append_chat_line("Assistant", ai_text)
        self.refresh_history_list()

    # -----------------------------------------------------------------------------
    # Display Util
    # -----------------------------------------------------------------------------
    def append_chat_line(self, speaker: str, msg: str) -> None:
        self.chat_display.configure(state="normal")
        self.chat_display.insert(tk.END, f"{speaker}: {msg}\n\n")
        self.chat_display.configure(state="disabled")
        self.chat_display.see(tk.END)

    def append_system_line(self, msg: str) -> None:
        self.chat_display.configure(state="normal")
        self.chat_display.insert(tk.END, f"— {msg} —\n\n")
        self.chat_display.configure(state="disabled")
        self.chat_display.see(tk.END)

    def redraw_chat_display(self) -> None:
        self.chat_display.configure(state="normal")
        self.chat_display.delete("1.0", tk.END)
        if self.cm is None:
            self.chat_display.configure(state="disabled")
            return

        for msg in self.cm.conversation_history:
            role = msg["role"].capitalize()
            self.chat_display.insert(tk.END, f"{role}: {msg['content']}\n\n")
        self.chat_display.configure(state="disabled")
        self.chat_display.see(tk.END)

    # -----------------------------------------------------------------------------
    # CM Helpr
    # -----------------------------------------------------------------------------
    def init_cm(self, *, history_file: str | None, custom_msg: str | None = None) -> None:
        persona_name = self.persona_var.get()
        if persona_name in {"", "Choose…", "Custom"}:
            persona_name = None 

        self.cm = ConversationManager(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            persona=persona_name,
            system_message=custom_msg,
            history_file=history_file,
        )

        if history_file is None:
            self.chat_display.configure(state="normal")
            self.chat_display.delete("1.0", tk.END)
            self.chat_display.configure(state="disabled")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    app = ChatGUI()
    app.mainloop()


if __name__ == "__main__":
    main()

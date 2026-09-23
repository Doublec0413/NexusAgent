import os
import sys
import time
import asyncio
import random
import queue
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from prompt_toolkit import PromptSession, print_formatted_text
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.styles import Style
from prompt_toolkit.application import get_app, run_in_terminal

from nexusagent.core.agent import create_agent_app, _stream_token_text
from nexusagent.core.config import DB_PATH
from nexusagent.core.bus import task_queue, purge_cancelled_heartbeats, is_cancelled_heartbeat
from nexusagent.core.heartbeat import pacemaker_loop

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def type_line(text: str, delay: float = 0.008):
    for ch in text:
        print(ch, end='', flush=True)
        time.sleep(delay)
    print()

def print_banner():
    clear_screen()

    CYAN = '\033[38;5;51m'
    PURPLE = '\033[38;5;141m'
    SILVER = '\033[38;5;250m'
    DIM = '\033[2m'
    BOLD = '\033[1m'
    RESET = '\033[0m'
    WHITE = '\033[37m'

    logo = f"""{CYAN}{BOLD}
                              {PURPLE}🔮{RESET}{CYAN}{BOLD}

    _   __                     ___                    __
   / | / /__  _  ____  _______/   | ____ ____  ____  / /_
  /  |/ / _ \\| |/_/ / / / ___/ /| |/ __ `/ _ \\/ __ \\/ __/
 / /|  /  __/>  </ /_/ (__  ) ___ / /_/ /  __/ / / / /_
/_/ |_/\\___/_/|_|\\__,_/____/_/  |_\\__, /\\___/_/ /_/\\__/
                                 /____/
{RESET}"""

    sub_title = f"{WHITE}{BOLD} 🔮 Welcome to {PURPLE}{BOLD}NexusAgent{RESET}{WHITE}{BOLD} — 下一代透明智能体 {RESET}"

    quotes = [
        "It works on my machine.",
        "It compiles! Ship it.",
        "Git commit, push, pray.",
        "There's no place like 127.0.0.1.",
        "sudo make me a sandwich.",
        "Works fine in dev.",
        "May the source be with you.",
        "Ctrl+C, Ctrl+V, Deploy.",
        "Hello, World."
    ]
    quote = random.choice(quotes)
    meta = f" {SILVER}✦{RESET} {CYAN}{quote}{RESET}"

    tip = (
        f"{PURPLE} ✦ {RESET}"
        f"{SILVER}{PURPLE}{BOLD}NexusAgent{RESET} 已完成启动。输入命令开始，输入 {PURPLE}/exit{RESET}{SILVER} 退出。{RESET}\n"
    )

    print(logo)
    print(sub_title)
    print() 
    time.sleep(0.12)
    print(meta)
    print() 
    type_line(tip, delay=0.004)


def cprint(text="", end="\n"):
    print_formatted_text(ANSI(str(text)), end=end)


def _format_agent_reply(content: str) -> str:
    lines = content.split("\n")
    formatted = f"  \033[38;5;141m❯\033[0m \033[38;5;250m{lines[0]}"
    for line in lines[1:]:
        formatted += f"\n    {line}"
    return formatted + "\033[0m"


_STREAM_END = object()


class LiveStreamSession:
    """
    在单次 run_in_terminal 会话中，从队列读取 token 并实时写入真实终端。
    绕过 patch_stdout 对无换行增量输出的覆盖问题。
    """

    def __init__(self):
        self._queue = queue.Queue()
        self._task = None
        self._active = False
        self._buffer = ""
        self._had_tokens = False

    @property
    def active(self) -> bool:
        return self._active

    @property
    def had_tokens(self) -> bool:
        return self._had_tokens

    @property
    def buffer(self) -> str:
        return self._buffer

    async def start(self):
        if self._active:
            return
        self._active = True
        self._buffer = ""
        self._had_tokens = False
        self._queue = queue.Queue()
        self._task = asyncio.create_task(self._display_loop())

    async def _display_loop(self):
        def blocking():
            out = sys.__stdout__
            out.write("  \033[38;5;141m❯\033[0m \033[38;5;250m")
            out.flush()
            indent_next = False
            while True:
                item = self._queue.get()
                if item is _STREAM_END:
                    break
                for ch in item:
                    if indent_next:
                        out.write("    ")
                        indent_next = False
                    out.write(ch)
                    out.flush()
                    if ch == "\n":
                        indent_next = True
            out.write("\033[0m\n")
            out.flush()

        await run_in_terminal(blocking, in_executor=True)

    def push(self, token: str):
        if not self._active or not token:
            return
        self._had_tokens = True
        self._buffer += token
        self._queue.put(token)

    async def finish(self):
        if not self._active:
            return
        self._queue.put(_STREAM_END)
        if self._task:
            await self._task
        self._active = False
        self._task = None


async def print_agent_reply(content: str):
    """流式失败时的完整回复兜底。"""
    def do_write():
        sys.__stdout__.write(_format_agent_reply(content) + "\n")
        sys.__stdout__.flush()

    await run_in_terminal(do_write, in_executor=False)


async def print_agent_report(content: str):
    """多 Agent 报告等长文本输出。"""
    def do_write():
        sys.__stdout__.write(_format_agent_reply(content) + "\n")
        sys.__stdout__.flush()

    await run_in_terminal(do_write, in_executor=False)


async def async_main():
    print_banner()
    
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    load_dotenv(env_path)
    
    current_provider = os.getenv("DEFAULT_PROVIDER", "aliyun")
    current_model = os.getenv("DEFAULT_MODEL", "glm-5")

    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as memory:
        app = create_agent_app(provider_name=current_provider, model_name=current_model, checkpointer=memory)
        config = {"configurable": {"thread_id": "local_geek_master"}}

        class SpinnerState:
            action_words = [
                "Thinking...",              
                "Working...",               
                "Beep boop...",             
                "Eating bugs...",           
                "Charging battery...",      
                "Brewing coffee...",        
                "Blinking lights...",       
                "Polishing pixels...",      
                "Scanning matrix...",       
                "Warming up circuits...",   
                "Syncing data...",          
                "Pinging server..."         
            ]
            current_words = [] 
            is_spinning = False
            start_time = 0
            frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
            is_tool_calling = False 
            tool_msg = ""           

        spinner = SpinnerState()


        def get_bottom_toolbar():
            if not spinner.is_spinning:
                return ANSI("") 
            
            elapsed = time.time() - spinner.start_time
            if spinner.is_tool_calling:
                display_msg = spinner.tool_msg
            else:
                idx_word = int(elapsed) % len(spinner.current_words)
                display_msg = f"🔮 {spinner.current_words[idx_word]}"

            idx_frame = int(elapsed * 12) % len(spinner.frames)
            frame = spinner.frames[idx_frame]
            

            return ANSI(f"  \033[38;5;51m{frame}\033[0m \033[38;5;250m{display_msg}\033[0m \033[38;5;141m[{elapsed:.1f}s]\033[0m")

        prompt_message = ANSI("  \033[38;5;51m❯\033[0m ")
        placeholder_text = ANSI("\033[3m\033[38;5;242minput...\033[0m")

        async def agent_worker():
            while True:
                await purge_cancelled_heartbeats()
                user_input = await task_queue.get()
                if user_input.lower() in ["/exit", "/quit"]:
                    task_queue.task_done()
                    break

                # 任务已取消但心跳仍在队列中排队：直接丢弃，不再叫醒 Agent
                if is_cancelled_heartbeat(user_input):
                    task_queue.task_done()
                    continue
                
                spinner.current_words = spinner.action_words.copy()
                random.shuffle(spinner.current_words)
                
                spinner.start_time = time.time()
                spinner.is_spinning = True
                spinner.is_tool_calling = False
                
                inputs = {"messages": [HumanMessage(content=user_input)]}
                report_shown = False
                live = LiveStreamSession()
                try:
                    async for mode, event in app.astream(
                        inputs, config=config, stream_mode=["updates", "custom"]
                    ):
                        if mode == "custom":
                            token = event.get("token", "")
                            if token:
                                spinner.is_spinning = False
                                if not live.active:
                                    await live.start()
                                live.push(token)
                            continue

                        for node_name, node_data in event.items():
                            if node_name == "agent":
                                last_msg = node_data["messages"][-1]
                                
                                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                                    await live.finish()
                                    live = LiveStreamSession()
                                    for tc in last_msg.tool_calls:
                                        spinner.is_tool_calling = True
                                        spinner.tool_msg = f"唤醒内置工具 : {tc['name']}..."
                                        cprint(f"  ●\033[38;5;51m Tool Call: \033[0m{tc['name']}")
                                        cprint('')
                                        
                                elif last_msg.content:
                                    content = _stream_token_text(last_msg.content).strip()
                                    streamed = live.buffer.strip()
                                    had_tokens = live.had_tokens
                                    await live.finish()
                                    live = LiveStreamSession()

                                    if report_shown and len(content) < 150 and any(
                                        k in content for k in ("报告已完成", "供你审查", "如需保存", "如需调整")
                                    ):
                                        report_shown = False
                                        continue

                                    if not had_tokens or len(streamed) < len(content) * 0.85:
                                        spinner.is_spinning = False
                                        await print_agent_reply(content)
                                    
                            elif node_name == "tools":
                                spinner.is_tool_calling = False
                                for msg in node_data.get("messages", []):
                                    if getattr(msg, "name", None) != "multi_agent_collaborate":
                                        continue
                                    content = msg.content if isinstance(msg.content, str) else str(msg.content)
                                    if not content:
                                        continue
                                    if "\n---\n\n" in content:
                                        report_body = content.split("\n---\n\n", 1)[1].strip()
                                    else:
                                        report_body = content.rsplit("\n\n", 1)[-1].strip()
                                    if not report_body or report_body.startswith("多Agent协作完成"):
                                        continue
                                    await print_agent_report(report_body)
                                    report_shown = True
                                    spinner.is_spinning = False
                                
                except Exception as e:
                    await live.finish()
                    spinner.is_spinning = False
                    cprint(f"  \033[31m[ ⚠️ 引擎异常 : {e} ]\033[0m")

                await live.finish()
                spinner.is_spinning = False
                cprint() # 空出舒适的行距
                task_queue.task_done()

        async def user_input_loop():
            custom_style = Style.from_dict({
                'bottom-toolbar': 'bg:default fg:default noreverse',
            })
            
            session = PromptSession(
                bottom_toolbar=get_bottom_toolbar,
                style=custom_style,
                erase_when_done=True,
                reserve_space_for_menu=0  
            )
            
            async def redraw_timer():
                while True:
                    if spinner.is_spinning:
                        try:
                            get_app().invalidate()
                        except Exception:
                            pass
                    await asyncio.sleep(0.08)
                    
            redraw_task = asyncio.create_task(redraw_timer())
            
            while True:
                try:
                    user_input = await session.prompt_async(prompt_message, placeholder=placeholder_text)

                    user_input = user_input.strip()
                    if not user_input:
                        continue
                    

                    padded_bubble = f"  ❯ {user_input}    "
                    cprint(f"\033[48;2;38;38;38m\033[38;5;255m{padded_bubble}\033[0m\n")
                    
                    await task_queue.put(user_input)
                    if user_input.lower() in ["/exit", "/quit"]:
                        cprint("  \033[38;5;141m✦ 记忆已固化，NexusAgent 进入休眠。\033[0m")
                        break
                        
                except (KeyboardInterrupt, EOFError):
                    cprint("\n  \033[38;5;141m✦ 强制中断，NexusAgent 进入休眠。\033[0m")
                    await task_queue.put("/exit")
                    break

            redraw_task.cancel() 

        with patch_stdout():
            worker = asyncio.create_task(agent_worker())
            heartbeat_worker = asyncio.create_task(pacemaker_loop(task_queue=task_queue, check_interval=10))
            await user_input_loop()
            await task_queue.join()
            worker.cancel()
            heartbeat_worker.cancel()

def main():
    asyncio.run(async_main())

if __name__ == "__main__":
    main()
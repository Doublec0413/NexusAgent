"""NexusAgent 命令行入口：config（配置向导）、run（启动 Agent）、monitor（启动监视器）。"""

import os
import typer
import questionary
import logging
from rich.console import Console
from rich.panel import Panel
from rich.status import Status
from dotenv import set_key, load_dotenv, unset_key
import sys

from nexusagent.core.provider import get_provider
from langchain_core.messages import HumanMessage

# ── 路径初始化：无论从何处调用，工作目录与 import 路径均指向项目根 ──
ENTRY_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ENTRY_DIR)

os.chdir(PROJECT_ROOT)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── CLI 应用与终端 UI ──
app = typer.Typer(help="NexusAgent - 极客专属的赛博智能终端")
console = Console()

# Questionary 交互式问答的赛博风配色
cyber_style = questionary.Style([
    ('qmark', 'fg:#8d52ff bold'),
    ('question', 'fg:#00ffff bold'),
    ('answer', 'fg:#8d52ff bold'),
    ('pointer', 'fg:#00ffff bold'),
    ('highlighted', 'fg:#00ffff bold'),
    ('selected', 'fg:#00ffff'),
    ('instruction', 'fg:#808080'),
])

ENV_PATH = os.path.join(PROJECT_ROOT, ".env")


@app.command("config")
def config_wizard():
    """交互式配置向导：收集 Provider / 模型 / Key / Base URL，探测连通性后写入 .env。"""
    console.clear()
    console.print(Panel(
        "🔮 Welcome to [bold #8d52ff]NexusAgent[/bold #8d52ff]...\n\n☁️[dim] 请完成模型配置，我们将把密钥安全固化在本地。[/dim]",
        title="[bold white]✦  NexusAgent Config[/bold white]",
        border_style="#8d52ff"
    ))

    # ── 1. 收集 Provider ──
    provider_raw = questionary.select(
        "选择你的模型提供商 (Provider):",
        choices=["openai", "anthropic", "aliyun (openai compatible)","tencent (openai compatible)", "z.ai (openai compatible)", "other (openai compatible)", "ollama"],
        style=cyber_style,
        instruction="(按上下键选择，回车确认)"
    ).ask()

    if not provider_raw:
        console.print("[dim #8d52ff]✦   录入中断，NexusAgent 配置已取消。[/dim #8d52ff]")
        return

    # 选项文案含后缀说明，实际 provider 名取第一个词（如 "aliyun (openai compatible)" → "aliyun"）
    provider = provider_raw.split(" ")[0].strip()
    is_openai_compatible = "openai" in provider_raw.lower()

    # ── 2. 收集模型名 ──
    model_name = questionary.text(
        "输入指定的模型型号 (如 gpt-4o-mini, qwen-max, glm-4 等):",
        style=cyber_style
    ).ask()

    if model_name is None:
        console.print("[dim #8d52ff]✦   录入中断，NexusAgent 配置已取消。[/dim #8d52ff]")
        return

    # ── 3. 收集 API Key（Ollama 本地运行，无需 Key）──
    api_key = ""
    env_key = ""
    if provider != "ollama":
        if is_openai_compatible:
            env_key = "OPENAI_API_KEY"
        elif provider == "anthropic":
            env_key = "ANTHROPIC_API_KEY"

        api_key = questionary.password(
            f"输入你的 {env_key} (对应 {provider_raw}):",
            style=cyber_style
        ).ask()

        if api_key is None:
            console.print("[dim #8d52ff]✦   录入中断，NexusAgent 配置已取消。[/dim #8d52ff]")
            return

    # ── 4. 收集 Base URL（不同 Provider 提示文案不同，均可留空使用默认）──
    base_url = ""
    if provider in ["openai", "anthropic"]:
        base_url = questionary.text(
            f"输入 {provider} 代理 Base URL (直连请直接回车跳过):",
            style=cyber_style
        ).ask()
    elif provider == "ollama":
        base_url = questionary.text(
            "输入 Ollama Base URL (默认 http://localhost:11434，直接回车跳过):",
            style=cyber_style
        ).ask()
    else:
        base_url = questionary.text(
            "输入兼容 Base URL (不填直接回车将使用官方默认地址):",
            style=cyber_style
        ).ask()

    if base_url is None:
        console.print("[dim #8d52ff]✦   录入中断，NexusAgent 配置已取消。[/dim #8d52ff]")
        return

    # ── 5. 连通性探测：临时注入环境变量，发一条测试消息验证配置 ──
    console.print("\n[dim]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/dim]")

    with Status(f"[bold #8d52ff]正在连接 {provider.upper()} 引擎并发送探测包...[/bold #8d52ff]", spinner="dots", spinner_style="#00ffff"):
        try:
            if env_key and api_key:
                os.environ[env_key] = api_key
            if base_url:
                if is_openai_compatible:
                    os.environ["OPENAI_API_BASE"] = base_url
                else:
                    os.environ[f"{provider.upper()}_BASE_URL"] = base_url

            llm = get_provider(provider_name=provider, model_name=model_name)
            response = llm.invoke([HumanMessage(content="回复我'收到'。")])

            console.print(" [bold #00ffff][ 配置成功!][/bold #00ffff]")

        except Exception as e:

            console.print(f" [bold #8d52ff][ 配置失败!][/bold #8d52ff]  无法连接到模型，请检查 Key、Base URL、模型型号 或 网络！\n[dim]错误信息: {str(e)}[/dim]")
            return

    # ── 6. 持久化到 .env（先清除旧的 Base URL，避免切换 Provider 后残留冲突）──
    if not os.path.exists(ENV_PATH):
        open(ENV_PATH, 'w').close()

    logging.getLogger("dotenv.main").setLevel(logging.ERROR)

    unset_key(ENV_PATH, "OPENAI_API_BASE")
    unset_key(ENV_PATH, "ANTHROPIC_BASE_URL")
    unset_key(ENV_PATH, "OLLAMA_BASE_URL")

    if env_key and api_key:
        set_key(ENV_PATH, env_key, api_key)

    if base_url:
        if is_openai_compatible:
            set_key(ENV_PATH, "OPENAI_API_BASE", base_url)
        else:
            set_key(ENV_PATH, f"{provider.upper()}_BASE_URL", base_url)

    set_key(ENV_PATH, "DEFAULT_PROVIDER", provider)
    set_key(ENV_PATH, "DEFAULT_MODEL", model_name)

    console.print(Panel(
        f"配置已保存至 [#8d52ff]{ENV_PATH}[/#8d52ff]\n"
        f"当前默认提供商: [#8d52ff]{provider}[/#8d52ff] | 模型: [#8d52ff]{model_name}[/#8d52ff]\n\n"
        f"👉 输入 [bold #00ffff]nexusagent run[/bold #00ffff] 即可启动系统！",
        border_style="#00ffff"
    ))


def _show_boot_error():
    """启动前配置缺失时的统一错误提示。"""
    console.print(Panel(
        "[bold #00ffff]NexusAgent未完成配置![/bold #00ffff]\n\n"
        "[#8d52ff]检测到 API Key、模型或Baseurl。请重新执行以下命令完成配置：[/#8d52ff]\n"
        "[bold #00ffff]nexusagent config[/bold #00ffff]",
        title="[bold #8d52ff]⚠️ Boot Sequence Failed[/bold #8d52ff]",
        border_style="#8d52ff"
    ))


@app.command("run")
def run_agent():
    """加载 .env 并校验配置，通过后启动主 Agent 交互循环。"""
    load_dotenv(ENV_PATH)
    provider = os.getenv("DEFAULT_PROVIDER")
    model = os.getenv("DEFAULT_MODEL")
    if not provider or not model:
        _show_boot_error()
        raise typer.Exit()

    # 按 Provider 类型校验对应的 API Key
    if provider != "ollama":
        if provider in ["openai", "aliyun", "z.ai", "tencent", "other"]:
            if not os.getenv("OPENAI_API_KEY"):
                _show_boot_error()
                raise typer.Exit()

        elif provider == "anthropic":
            if not os.getenv("ANTHROPIC_API_KEY"):
                _show_boot_error()
                raise typer.Exit()

    # 延迟 import，避免未配置时加载整个 Agent 栈
    import entry.main as nexusagent_main
    nexusagent_main.main()


@app.command("monitor")
def run_monitor():
    """启动资源监视器（独立模块，缺失时给出友好提示）。"""
    try:
        import entry.monitor as nexusagent_monitor
        nexusagent_monitor.main()
    except ImportError as e:
        console.print(f"[bold red]启动失败：找不到监视器模块！[/bold red]\n[dim]请确保 monitor.py 和 cli.py 在同一目录下。\n报错信息: {e}[/dim]")


def main():
    app()


if __name__ == "__main__":
    main()

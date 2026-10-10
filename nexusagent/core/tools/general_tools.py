import ast
import math
import operator
import os
from datetime import datetime

from .base import nexusagent_tool
from ..config import MEMORY_DIR

PROFILE_PATH = os.path.join(MEMORY_DIR, "user_profile.md")

# 只放行算术运算符。属性访问、名字查找、函数调用都不在表里。
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
# 挡住 2 ** 999999999 这类整数乘方把 CPU 打满。
_MAX_POW_EXPONENT = 1000


def _safe_eval(node: ast.AST):
    """白名单求值：只接受数字常量和 _OPS 里的运算符。"""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    ):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        if type(node.op) is ast.Pow and not _is_safe_exponent(right):
            raise ValueError("指数超出允许范围")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("不支持的表达式")


def _is_safe_exponent(exponent) -> bool:
    if isinstance(exponent, bool) or not isinstance(exponent, (int, float)):
        return False
    if isinstance(exponent, float) and not math.isfinite(exponent):
        return False
    return abs(exponent) <= _MAX_POW_EXPONENT


@nexusagent_tool
def get_system_model_info() -> str:
    """
    获取当前 NexusAgent 正在运行的底层大模型（LLM）型号和提供商信息。
    当用户询问“你是基于什么模型”、“你的底层大模型是什么”、“你是GPT还是GLM”、“现在用的什么模型”等身份问题时，调用此工具。
    """
    provider = os.getenv("DEFAULT_PROVIDER", "unknown")
    model = os.getenv("DEFAULT_MODEL", "unknown")

    if provider == "unknown" or model == "unknown":
        return "无法获取当前的系统模型配置，可能是环境变量未正确加载。"

    return f"当前使用的模型提供商(Provider)是: {provider}，具体型号(Model)是: {model}。"


@nexusagent_tool
def save_user_profile(new_content: str) -> str:
    """
    更新用户的全局显性记忆档案。
    当你发现用户的偏好发生改变，或者有新的重要事实需要记录时：
    1.请先调用 read_user_profile 获取当前的完整档案。
    2.在你的上下文中，将新信息融入档案，并删去冲突或过时的旧信息。
    3.将修改后的一整篇完整 Markdown 文本作为 new_content 参数传入此工具。
    注意：此操作将完全覆盖旧文件！请确保传入的是完整的最新档案。
    """
    os.makedirs(MEMORY_DIR, exist_ok=True)
    with open(PROFILE_PATH, "w", encoding="utf-8") as f:
        f.write(new_content)

    return "记忆档案已成功覆写更新。新的人设画像已生效。"


@nexusagent_tool
def get_current_time() -> str:
    """
    获取当前的系统时间和日期。
    当用户询问“现在几点”、“今天星期几”、“今天几号”等与当前时间相关的问题时，调用此工具。
    """
    now = datetime.now()
    return f"当前本地系统时间是: {now.strftime('%Y-%m-%d %H:%M:%S')}"


@nexusagent_tool
def calculator(expression: str) -> str:
    """
    一个简单的数学计算器。
    用于计算基础的数学表达式，例如: '3 * 5' 或 '100 / 4'。
    注意：参数 expression 必须是一个合法的 Python 数学表达式字符串。
    """
    try:
        result = _safe_eval(ast.parse(expression, mode="eval"))
        return f"表达式 '{expression}' 的计算结果是: {result}"
    except Exception as e:
        return f"计算出错，请检查表达式格式。错误信息: {str(e)}"

"""向后兼容：旧代码可从 builtins 导入，新代码请用 nexusagent.core.tools。"""

from . import BUILTIN_TOOLS
from .general_tools import (
    PROFILE_PATH,
    calculator,
    get_current_time,
    get_system_model_info,
    save_user_profile,
)
from .knowledge_tools import query_knowledge_base
from .multi_agent_tools import multi_agent_collaborate
from .sandbox_tools import (
    execute_office_shell,
    list_office_files,
    read_office_file,
    write_office_file,
)
from .schedule_tools import (
    delete_scheduled_task,
    list_scheduled_tasks,
    modify_scheduled_task,
    schedule_task,
    tasks_lock,
)
from .security_tools import get_risk_score, get_security_report

__all__ = [
    "BUILTIN_TOOLS",
    "PROFILE_PATH",
    "tasks_lock",
    "multi_agent_collaborate",
    "get_current_time",
    "calculator",
    "save_user_profile",
    "list_office_files",
    "read_office_file",
    "write_office_file",
    "execute_office_shell",
    "get_system_model_info",
    "schedule_task",
    "list_scheduled_tasks",
    "delete_scheduled_task",
    "modify_scheduled_task",
    "query_knowledge_base",
    "get_security_report",
    "get_risk_score",
]

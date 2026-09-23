from .base import nexusagent_tool
from ..security_scorer import security_scorer


@nexusagent_tool
def get_security_report() -> str:
    """
    生成当前会话的安全审计报告。
    当用户询问"安全报告"、"风险评估"、"安全状况"时调用此工具。
    返回详细的安全评分、工具调用分布、风险趋势等信息。
    """
    report = security_scorer.generate_report()
    filepath = security_scorer.save_report()
    return f"{report}\n\n📁 报告已保存至: {filepath}"


@nexusagent_tool
def get_risk_score() -> str:
    """
    获取当前的实时安全风险评分。
    当用户询问"当前风险"、"安全评分"、"风险等级"时调用此工具。
    """
    data = security_scorer.get_dashboard_data()
    return (
        f"🛡️ NexusAgent 安全态势\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"风险评分: {data['risk_score']}/100\n"
        f"风险等级: {data['risk_level']}\n"
        f"工具调用总计: {data['total_tool_calls']} 次\n"
        f"被拦截操作: {data['blocked_operations']} 次\n"
        f"高风险事件: {data['high_risk_events']} 次\n"
        f"会话时长: {data['session_duration_min']} 分钟\n"
        f"━━━━━━━━━━━━━━━━━━━━"
    )

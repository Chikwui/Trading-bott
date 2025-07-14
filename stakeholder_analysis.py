"""
Stakeholder Analysis Module

Defines key stakeholders, their interests, and impact levels.
"""

def get_stakeholder_map():
    """Return a mapping of stakeholder roles to their interests and impact."""
    return {
        "Trader": {"interest": "Signal accuracy, performance", "impact": "High"},
        "Risk Manager": {"interest": "Compliance, drawdown limits", "impact": "High"},
        "IT Ops": {"interest": "Uptime, scalability", "impact": "Medium"},
        "Compliance": {"interest": "Regulatory adherence", "impact": "High"},
        "End User": {"interest": "Usability, transparency", "impact": "Low"},
    }

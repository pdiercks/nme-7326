"""initialize paper src"""

try:
    from pymor.core.logger import set_log_levels
    set_log_levels({"src": "INFO"})
except ModuleNotFoundError:
    print("pyMOR not installed.")

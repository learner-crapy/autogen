"""
This module provides the main entry point for the autogen_agentchat package.
It includes logger names for trace and event logs, and retrieves the package version.
"""

import importlib.metadata

TRACE_LOGGER_NAME = "autogen_agentchat"
"""Logger name for trace logs."""

EVENT_LOGGER_NAME = "autogen_agentchat.events"
"""Logger name for event logs."""

# 1. 这个能得到autogen—agentchat的版本吗？
# 是的，一般像下面这样使用。
# import autogen_agentchat
# print(autogen_agentchat.__version__)  # 会输出安装的autogen_agentchat版本
# 这里不写这句的话，还有其他方式查看包的版本。

# 2. 没导入，项目如何获取到的版本信息？
# 不需要导入，读取元数据，而不是代码
# 元数据一般放在： pyproject.toml
__version__ = importlib.metadata.version("autogen_agentchat")

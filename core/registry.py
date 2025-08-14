# Shim module to support legacy imports like `from core.registry import ...`.
# Delegate to the real implementation in `aigve.core.registry`.
from aigve.core.registry import *  # noqa: F401,F403

# Shim package to support legacy absolute imports like `import core`.
# Re-export from the actual package inside `aigve.core`.
from aigve.core import *  # noqa: F401,F403

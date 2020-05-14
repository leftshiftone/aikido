import os
from pathlib import Path

cache_root = os.getenv('AIKIDO_CACHE_ROOT', Path(Path.home(), ".aikido"))

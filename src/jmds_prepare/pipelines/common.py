from __future__ import annotations

from typing import Callable

from ..core.atomic import write_json_atomic as write_json_atomic

Output = Callable[[str], None]

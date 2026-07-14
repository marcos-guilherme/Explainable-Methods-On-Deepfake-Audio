import sys
from pathlib import Path

# Garante que `import brspeech_xai` funcione ao rodar pytest da raiz do repo.
_PKG_ROOT = Path(__file__).resolve().parents[1]  # .../scripts
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

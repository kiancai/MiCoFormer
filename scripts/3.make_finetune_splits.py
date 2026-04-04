from pathlib import Path
import sys


# 允许从工作区根目录直接执行 `python MiCoFormer/scripts/...`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from micoformer.cli.make_finetune_splits import main


if __name__ == "__main__":
    main()

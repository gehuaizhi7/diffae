from pathlib import Path
from typing import Iterable, List, Optional


def prune_old_checkpoints(logdir: str,
                          keep_paths: Iterable[Optional[str]] = ()) -> List[str]:
    """Remove stale checkpoint files in a run directory while keeping resume targets."""
    logdir_path = Path(logdir).resolve()
    keep = {(logdir_path / 'last.ckpt').resolve()}

    for path in keep_paths:
        if not path:
            continue
        candidate = Path(path).resolve()
        if candidate.parent == logdir_path:
            keep.add(candidate)

    removed = []
    for ckpt_path in sorted(logdir_path.glob('*.ckpt')):
        resolved = ckpt_path.resolve()
        if resolved in keep:
            continue
        ckpt_path.unlink()
        removed.append(str(ckpt_path))

    return removed

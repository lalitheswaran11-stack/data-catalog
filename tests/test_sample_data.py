"""Safety checks for the sample-data generator."""

from pathlib import Path
from runpy import run_path

import pytest

main = run_path(str(Path(__file__).parents[1] / "examples" / "make_sample_data.py"))[
    "main"
]


def test_generator_refuses_to_replace_unmarked_directory(tmp_path: Path) -> None:
    target = tmp_path / "existing"
    target.mkdir()
    sentinel = target / "keep.txt"
    sentinel.write_text("important", encoding="utf-8")

    with pytest.raises(ValueError, match="unmarked directory"):
        main(target, force=True)

    assert sentinel.read_text(encoding="utf-8") == "important"


def test_generator_requires_force_before_replacing_its_output(tmp_path: Path) -> None:
    target = tmp_path / "sample"
    main(target)

    with pytest.raises(FileExistsError, match="not empty"):
        main(target)

    main(target, force=True)
    assert (target / "trades_2026_01.csv").is_file()

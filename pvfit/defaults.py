"""Locate the same configuration templates in a checkout and an installed wheel."""

from pathlib import Path

CONFIG_NAMES = (
    "reconstruction.yaml",
    "analysis.yaml",
    "optimization.yaml",
    "allen.yaml",
    "optimization_allen.yaml",
)


def bundled_config_dir():
    package = Path(__file__).resolve().parent
    bundled = package / "_defaults"
    if bundled.is_dir():
        return bundled
    return package.parent / "configs"


def default_config_path(name):
    """Use a local configs/ file if present; otherwise use the shipped template."""
    local = Path("configs") / name
    return local if local.is_file() else bundled_config_dir() / name


def export_configs(output):
    """Create editable templates, refusing to replace any existing file."""
    output = Path(output).expanduser()
    for name in CONFIG_NAMES:
        path = output / name
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {path}")
    contents = {
        name: (bundled_config_dir() / name).read_text() for name in CONFIG_NAMES
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, text in contents.items():
        with (output / name).open("x") as stream:
            stream.write(text)

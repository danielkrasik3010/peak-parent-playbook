import os
import yaml
from pathlib import Path
from typing import Union, Optional
from dotenv import load_dotenv

from src.paths import DATA_DIR, ENV_FPATH


def read_article(art_id: str) -> str:
    """
    Fetch the contents of a single article markdown file.

    Args:
        art_id: The base filename (without extension) of the article.

    Returns:
        File content as a string.

    Raises:
        FileNotFoundError: If the file does not exist in DATA_DIR.
        IOError: If there is an issue reading the file.
    """
    file_path = Path(DATA_DIR) / f"{art_id}.md"

    if not file_path.is_file():
        raise FileNotFoundError(f"Publication not found: {file_path}")

    try:
        return file_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise IOError(f"Unable to read the article file: {file_path}") from exc


def read_all_articles(directory: Union[str, Path] = DATA_DIR) -> list[str]:
    """
    Retrieve the contents of all markdown articles in a directory.

    Args:
        directory: Path where the markdown files after scraping are stored.

    Returns:
        A list containing the content of each markdown file.
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        return []

    return [
        read_article(file.stem)
        for file in dir_path.iterdir()
        if file.suffix == ".md" and file.is_file()
    ]


def load_yaml(file_path: Union[str, Path]) -> dict:
    """
    Load and parse a YAML file.

    Args:
        file_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing the parsed YAML content.
    """
    yaml_path = Path(file_path)

    if not yaml_path.is_file():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    try:
        with yaml_path.open("r", encoding="utf-8") as stream:
            return yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise yaml.YAMLError(f"Error parsing YAML: {yaml_path}") from exc
    except OSError as exc:
        raise IOError(f"Error reading YAML file: {yaml_path}") from exc
    


def ensure_env(api_key_name: str = "OPENAI_API_KEY") -> None:
    """
    Load environment variables from .env and ensure a specific key exists.

    Args:
        api_key_name: Name of the environment variable to check.

    Raises:
        AssertionError: If the variable is missing or empty.
    """
    load_dotenv(ENV_FPATH, override=True)

    value = os.getenv(api_key_name)
    assert value, f"Missing required environment variable: '{api_key_name}'"


def write_text(
    content: str,
    destination: Union[str, Path],
    title: Optional[str] = None
) -> None:
    """
    Write text to a file, with an optional title section.

    Args:
        content: The text to write.
        destination: Path where the file will be written.
        title: Optional heading to place at the top of the file.

    Raises:
        IOError: If the file cannot be written.
    """
    dest_path = Path(destination)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with dest_path.open("w", encoding="utf-8") as handle:
            if title:
                handle.write(f"# {title}\n")
                handle.write("# " + "=" * 60 + "\n\n")
            handle.write(content)
    except OSError as exc:
        raise IOError(f"Error writing to file: {dest_path}") from exc

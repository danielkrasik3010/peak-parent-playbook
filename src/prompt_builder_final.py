"""
prompt_builder_final.py

Module for constructing and managing AI prompts for the Peak Parent Playbook (PPP) assistant.

This script provides utilities to:
- Build structured prompts from YAML configuration files
- Include role, instructions, context, style/tone, examples, and output constraints
- Incorporate input data and optional reasoning strategies
- Preview and save prompts as Markdown files for documentation or debugging
- Load YAML configuration files

Key Functions:
- lowercase_first_char: Helper to lowercase the first character of a string
- format_prompt_section: Format individual sections of a prompt
- build_prompt_from_config: Construct a full prompt based on config, input, and app settings
- print_prompt_preview: Display a preview of the constructed prompt
- save_prompt_to_md: Save the prompt to a Markdown file
- load_yaml_config: Load configuration from a YAML file

Author: Daniel Krasik
"""

import os
import yaml
from typing import Optional, Dict, Any
from datetime import datetime
from src.paths import PROMPT_CONFIG_FPATH, OUTPUTS_DIR



def lowercase_first_char(text: str) -> str:
    return text[0].lower() + text[1:] if text else text


def format_prompt_section(lead_in: str, value: Any) -> str:
    if isinstance(value, list):
        formatted_value = "\n".join(f"- {item}" for item in value)
    else:
        formatted_value = value
    return f"{lead_in}\n{formatted_value}"


def build_prompt_from_config(
    config: Dict[str, Any],
    input_data: str = "",
    app_config: Optional[Dict[str, Any]] = None,
) -> str:
    prompt_parts = []

    if role := config.get("role"):
        prompt_parts.append(f"You are {lowercase_first_char(role.strip())}.")

    instruction = config.get("instruction")
    if not instruction:
        raise ValueError("Missing required field: 'instruction'")
    prompt_parts.append(format_prompt_section("Your task is as follows:", instruction))

    if context := config.get("context"):
        prompt_parts.append(f"Here’s some background that may help you:\n{context}")

    if constraints := config.get("output_constraints"):
        prompt_parts.append(format_prompt_section("Ensure your response follows these rules:", constraints))

    if tone := config.get("style_or_tone"):
        prompt_parts.append(format_prompt_section("Follow these style and tone guidelines in your response:", tone))

    if format_ := config.get("output_format"):
        prompt_parts.append(format_prompt_section("Structure your response as follows:", format_))

    if examples := config.get("examples"):
        prompt_parts.append("Here are some examples to guide your response:")
        if isinstance(examples, list):
            for i, example in enumerate(examples, 1):
                if isinstance(example, dict):
                    user_q = example.get("user question", "")
                    response = example.get("response", "")
                    prompt_parts.append(f"Example {i}:\nUser question: {user_q}\nResponse:\n{response}")
                else:
                    prompt_parts.append(str(example))
        else:
            prompt_parts.append(str(examples))

    if goal := config.get("goal"):
        prompt_parts.append(f"Your goal is to achieve the following outcome:\n{goal}")

    if input_data:
        prompt_parts.append(
            "Here is the content you need to work with:\n"
            "<<<BEGIN CONTENT>>>\n"
            "```\n" + input_data.strip() + "\n```\n<<<END CONTENT>>>"
        )

    reasoning_strategy = config.get("reasoning_strategy")
    if reasoning_strategy and reasoning_strategy != "None" and app_config:
        strategies = app_config.get("reasoning_strategies", {})
        if strategy_text := strategies.get(reasoning_strategy):
            prompt_parts.append(strategy_text.strip())

    prompt_parts.append("Now perform the task as instructed above.")
    return "\n\n".join(prompt_parts)


def print_prompt_preview(prompt: str, max_length: int = 500) -> None:
    print("=" * 60)
    print("CONSTRUCTED PROMPT:")
    print("=" * 60)
    if len(prompt) > max_length:
        print(prompt[:max_length] + "...\n")
        print(f"[Truncated - Full prompt is {len(prompt)} characters]")
    else:
        print(prompt)
    print("=" * 60)


def save_prompt_to_md(
    prompt_text: str,
    filename: Optional[str] = None,
    outputs_dir: str = OUTPUTS_DIR
) -> str:
    os.makedirs(outputs_dir, exist_ok=True)
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"prompt_{timestamp}.md"
    if not filename.endswith(".md"):
        filename += ".md"

    filepath = os.path.join(outputs_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(prompt_text)

    return filepath


def load_yaml_config(filepath: str) -> Dict[str, Any]:
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    prompt_key = "rag_ppp_assistant_prompt" 
    # Load prompt config YAML
    prompt_config = load_yaml_config(PROMPT_CONFIG_FPATH)

    if prompt_key not in prompt_config:
        raise KeyError(f"Prompt key '{prompt_key}' not found in config.")

    prompt_data = prompt_config[prompt_key]

    app_config_path = os.path.join("code", "config", "app_config.yaml")
    if os.path.exists(app_config_path):
        app_config = load_yaml_config(app_config_path)
    else:
        app_config = None

    # Build prompt from loaded config
    prompt_text = build_prompt_from_config(prompt_data, app_config=app_config)

    # Print preview
    print_prompt_preview(prompt_text)

    # Save prompt as markdown to outputs folder
    saved_filepath = save_prompt_to_md(prompt_text, filename=f"{prompt_key}_prompt.md")
    print(f"Prompt saved to: {saved_filepath}")

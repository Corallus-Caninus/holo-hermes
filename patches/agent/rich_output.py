"""Rich-based syntax highlighter for assistant responses.
Manually parses fenced code blocks and highlights them with Pygments.
Uses a dark terminal theme with vibrant green, red, purple."""

import re
from rich.syntax import Syntax
from rich.console import Console
from rich.text import Text
from io import StringIO

_CODE_FENCE = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)


def _render(text: str, prefix: str = "") -> str:
    """Render text with syntax-highlighted code blocks to ANSI string."""
    console = Console(
        file=StringIO(),
        force_terminal=True,
        color_system="truecolor",
        width=120,
    )

    if prefix:
        console.print(Text(prefix), end="")

    # Split on code fences, render blocks with Syntax, rest as plain text
    last_end = 0
    for match in _CODE_FENCE.finditer(text):
        # Plain text before this code block
        before = text[last_end : match.start()]
        if before.strip():
            console.print(Text(before.strip()), end="")

        lang = match.group(1) or "text"
        code = match.group(2)
        try:
            highlighted = Syntax(
                code,
                lang,
                theme="monokai",
                background_color="default",
                word_wrap=True,
            )
            console.print(highlighted)
        except Exception:
            # Fallback: print raw code block as plain text
            console.print(Text(code), end="")

        last_end = match.end()

    # Remaining text after last code block
    remaining = text[last_end:]
    if remaining.strip():
        console.print(Text(remaining.strip()), end="")

    return console.file.getvalue()


def render_response(agent, text: str, log_prefix: str = "") -> None:
    """Render assistant response with syntax highlighting and print via agent."""
    if not text:
        return
    try:
        ansi = _render(text, prefix=f"{log_prefix}🤖 Assistant: " if log_prefix else "")
        agent._safe_print(ansi)
    except Exception as e:
        # Write error as safe_print message
        try:
            agent._safe_print(f"[RICH DEBUG] render_response exception: {type(e).__name__}: {e}")
        except Exception:
            pass


def render_final(text: str) -> str:
    """Render final_response with Rich highlighting and return ANSI string.
    
    Call this in the conversation loop right before returning the result.
    Returns the ANSI-highlighted version for the TUI to display.
    """
    if not text:
        return text
    try:
        return _render(text, prefix="")
    except Exception:
        return text

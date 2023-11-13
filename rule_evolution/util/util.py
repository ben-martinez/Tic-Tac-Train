from typing import Dict
import logging

# --- LOGGING ---
def get_basic_logger(name: str) -> logging.Logger:
    """
    Returns a basic logger with formatting such as:
    2023-11-21 01:11:45,035 - game_util - WARNING - Could not open minimax cache path caches/minimax_cache_dim-4,4_n-4.pkll!

    Parameters:
        name: The logger name.
    """
    logger = logging.getLogger(name)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    return logger


# --- STR TRANSFORMS ---
def fill_tag(text: str, tag: str, fill: str) -> str:
    """
    Function that performs the following string transform:
    text == 'abc tag abc' -> text == 'abc fill abc'

    Parameters:
        text: The original text.
        tag: The tag to replace - will replace instances of tag.
        fill: The content to replace tags with.
    """
    assert isinstance(text, str)
    assert isinstance(tag, str)
    assert isinstance(fill, str)

    return text.replace(tag, fill)


def fill_tags(text: str, tag_fill_map: Dict[str, str]) -> str:
    """
    Given a tag_fill_map: {'tag1': 'fill1', 'tag2' : 'fill2'},
    performs the following string transform:
    text == 'abc tag1 tag2 abc' -> text == 'abc fill1 fill2 abc'

    Parameters:
        text: The original text.
        tag_fill_map: A dictionary mapping tags to fills.
    """
    for tag in tag_fill_map:
        text = fill_tag(text, tag, tag_fill_map[tag])

    return text
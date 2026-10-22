"""Self-contained, torch-free GPT-SoVITS text frontend (first slice).

Public entry points::

    from text_frontend import TextFrontend
    result = TextFrontend().process("你好，world！", language="auto")

The package does not import torch, transformers, MNN, or the GPT-SoVITS-OIE
runtime. pypinyin is imported lazily for Chinese. English uses NumPy G2P,
NLTK POS and wordsegment with explicitly prepared offline resources. jieba
is optional. See NOTICE.txt and licenses/ for third-party attribution.
"""

from .errors import (
    EmptyTextError,
    LayoutError,
    MissingDependencyError,
    ResourceError,
    TextFrontendError,
    TokenizerError,
    UnknownWordError,
    UnsupportedLanguageError,
    UnsupportedScriptError,
)
from .frontend import FrontendResult, SegmentResult, TextFrontend
from .symbols import SYMBOLS, SYMBOL_TO_ID, VERSION, symbol_ids
from .tokenizer import TokenizedText, WordPieceTokenizer

__all__ = [
    "EmptyTextError",
    "FrontendResult",
    "LayoutError",
    "MissingDependencyError",
    "ResourceError",
    "SegmentResult",
    "SYMBOLS",
    "SYMBOL_TO_ID",
    "TextFrontend",
    "TextFrontendError",
    "TokenizedText",
    "TokenizerError",
    "UnknownWordError",
    "UnsupportedLanguageError",
    "UnsupportedScriptError",
    "VERSION",
    "WordPieceTokenizer",
    "symbol_ids",
]

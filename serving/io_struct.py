from dataclasses import dataclass, field
from typing import List, Optional

from omniserve.sampling_params import SamplingParams


@dataclass
class TokenizedGenerateReqInput:
    rid: str
    input_ids: List[int]
    sampling_params: SamplingParams
    stream: bool = False


@dataclass
class BatchTokenIDOut:
    rids: List[str]
    output_tokens: List[List[int]]
    finished: List[bool]
    final_text: List[Optional[str]] = field(default_factory=list)
    errors: List[Optional[str]] = field(default_factory=list)


@dataclass
class BatchStrOut:
    rids: List[str]
    output_strs: List[str]
    finished: List[bool]
    errors: List[Optional[str]] = field(default_factory=list)

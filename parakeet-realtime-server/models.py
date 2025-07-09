from typing import List, Optional

from pydantic import BaseModel


class TranscribedSegment(BaseModel):
    complete: bool
    words: List[str]
    start: float
    id: Optional[int] = None
    sample_count: int
    final: bool = False
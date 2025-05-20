from typing import TypedDict, List, Optional

class ContextChunk(TypedDict):
    name: str
    file: str
    source: Optional[str] 
    
class HistoryMessage(TypedDict):
    query: str
    answer: str

class ChatState(TypedDict):
    question: str
    processed_question: str
    context_chunks: List[ContextChunk]
    context :str
    final_answer: str
    messages: List[HistoryMessage]
    history_summary: str
    history : str

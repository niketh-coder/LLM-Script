from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import re
from groq import  Groq
from tqdm import tqdm
import time
import tiktoken
import json
import requests
from typing import TypedDict, Optional, List, Annotated, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver ,InMemorySaver

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" 
INDEX_DIR = "/code analyzer/vector_db"
SUMMARY_CACHE = "summary_cache.json"
UPLOADED_SUMMARY_CACHE = "summary_cache.json"
GROQ_MODEL = "llama-3.3-70b-versatile"
MAX_RECENT_MESSAGES = 2
MAX_CONTEXT_TOKENS = 6000

api_keys = ['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',
            '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' , 
            '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', #mu
            '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', #godz.07
            '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', #shubhankar
           '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' ,
            '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',
            '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',
            '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',
            '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',#arnav
            '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'  #shreyans
           ]

embedder = SentenceTransformer(MODEL_NAME)
with open(f"{INDEX_DIR}/metadata.json", "r") as f:
        metadata = json.load(f)
used_chunks = set()
count = 0 

ENCODING = tiktoken.get_encoding("cl100k_base")
def count_tokens(text):
    cnt = len(ENCODING.encode(text))
    return cnt

def search(query, top_k=3):
    index = faiss.read_index(f"{INDEX_DIR}/code_index.faiss")

    q_vec = embedder.encode([query])
    _, I = index.search(q_vec, top_k)

    global used_chunks
    for id in I[0][:20] :
        if id not in used_chunks :
            used_chunks.add(id)
            
    return [metadata[i] for i in I[0]]

def convert_chunks_to_context_format(chunks):
    context_chunks = []
    for chunk in chunks:
        if len(chunk) >= 4:
            summary, file, source, name = chunk
            context_chunks.append({
                "name": name,
                "file": file,
                "source": source
            })
    return context_chunks

def convert_referenced_items(chunks):
    referenced_chunks = []
    for chunk in chunks :
        summary, file, source, name = chunk
        referenced_chunks.append(
            {
                "name": name,
                "file": file,
            }
        )
    return referenced_chunks

def merge_context_chunks(chunks , complete_cnt = 20):
    full = convert_chunks_to_context_format(chunks[:complete_cnt])
    refs = convert_referenced_items(chunks[complete_cnt : ])

    return full + refs


def chunk_context_by_token_limit(context_chunks, max_tokens=4000):
    batches = []
    current_batch = []
    current_tokens = 0
    for chunk in context_chunks:
        # print(chunk)
        global count
        chunk_tokens = count_tokens(chunk['file']) + count_tokens(chunk['name'])
        if 'source' in chunk.keys():
            chunk_tokens += count_tokens(chunk['source'])

        count += chunk_tokens
        if current_tokens + chunk_tokens > max_tokens:
            if current_batch:
                batches.append(current_batch)
            current_batch = [chunk]
            current_tokens = chunk_tokens
        else:
            current_batch.append(chunk)
            current_tokens += chunk_tokens

    if current_batch:
        batches.append(current_batch)

    return batches

def extract_json_from_response(text):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    return json.loads(text)

def ask_llm_to_refine_context_batch(query, chunk_batch):
    system_prompt = """You are an expert developer helping to analyze a React-Django full-stack project.
Your task is to:
1. From the provided code context, extract all the code relevant to the query solution.
2. Do not remove the given context chunks too much.
3. Identify function/class names that are needed to understand the code but are not included yet. These could be:
   - Functions/classes called or instantiated in the context.
   - Functions/classes mentioned but their code is not given.

Respond strictly in the following JSON format without any other sentences:
{
  "relevant_chunks": [
    {
      "name": "...",
      "file": "...",
      "trimmed_source": "..."
    },
    ...
  ],
  "needed_definitions": [
    { "name": "...", "file": "..." },
    ...
  ]
}
Be conservative — only include names in needed definitions if you are confident are required."""

    context_input = {
        "query": query,
        "context/extra chunks": chunk_batch
    }

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(context_input, indent=2)}
    ]
    client = Groq(api_key = api_keys[4])
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0
    )
    text = extract_json_from_response(response.choices[0].message.content)
    return text

def process_all_batches(query, context_chunks, token_limit=4000):
    batches = chunk_context_by_token_limit(context_chunks, max_tokens=token_limit)

    all_relevant = []
    all_needed = []

    for i, batch in enumerate(batches):
        print(f"Processing batch {i+1}/{len(batches)}...")
        result = ask_llm_to_refine_context_batch(query, batch)
        all_relevant.extend(result.get("relevant_chunks", []))
        all_needed.extend(result.get("needed_definitions", []))

    return {
        "relevant_chunks": all_relevant,
        "needed_definitions": all_needed
    }

def get_needed_functions(name , file) :
    global metadata
    global used_chunks
    functions = []
    for i in range(len(metadata)) :
        if i in used_chunks :
            continue
        if (metadata[i][1] == file or metadata[i][1] == 'unknown') :
            if name in metadata[i][2] and name != 'imports' :
                functions.append({
                    'name': name , 
                    'file' : file ,
                    'source' : metadata[i][2]
                })
                used_chunks.add(i)

    return functions

def merge_chunks_to_string(relevant_chunks):
    merged = []
    for chunk in relevant_chunks:
        name = chunk.get("name", "unknown")
        file = chunk.get("file", "unknown")
        source = chunk.get("trimmed_source", "").strip()
        merged.append(f"// File: {file} | Chunk: {name}\n{source}\n")
    return "\n".join(merged)

def iterative_context_expansion(query, context_chunks, max_rounds=1):
    final_chunks = []
    for round_no in range(max_rounds):
        result = process_all_batches(query, context_chunks , token_limit=4000)

        final_chunks.extend(result["relevant_chunks"])
        needed_chunks = result['needed_definitions']

        if not len(needed_chunks) or round_no == max_rounds - 1:
            break

        new_chunks = []
        for chunk in needed_chunks :
            functions = get_needed_functions(chunk['name'] , chunk['file'])
            new_chunks.extend(functions)

        context_chunks = new_chunks

    context = merge_chunks_to_string(final_chunks)
    
    return context


def chunk_context(text, max_tokens=MAX_CONTEXT_TOKENS):
    print(f'chunking {count_tokens(text)}')
    lines = text.split('\n')
    chunks, current_chunk, token_count = [], [], 0

    for line in lines:
        line_tokens = count_tokens(line)
        if token_count + line_tokens > max_tokens - 200:
            chunks.append('\n'.join(current_chunk))
            current_chunk, token_count = [], 0
            print("New chunk" ,count_tokens(chunks[-1]))
        current_chunk.append(line)
        token_count += line_tokens

    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    return chunks

def recursive_qa(context, query, ask_fn, max_tokens=MAX_CONTEXT_TOKENS):
    if count_tokens(context) <= max_tokens:
        return ask_fn(context, query)

    chunks = chunk_context(context, max_tokens)
    answers = [recursive_qa(chunk, query, ask_fn, max_tokens) for chunk in chunks]

    combined = "\n\n".join(answers)
    print('combined : ' , count_tokens(combined))
    return recursive_qa(combined, query , ask_fn, max_tokens)

system_prompt = (
        "You are an expert query automation assistant. "
        "Answer the user query using the provided relevant code context. "
        "Answer should be ralted to codebase functions only avoiding general problems like internet issue etc."
        "If it is not related to codebase dont answer"
        "Give 2 separate answers"
        """1 - Give a very detailed answer for the user regarding all possible issues without technical terrms(user have no idea of codebase) .
            - tell what user may be doing wrong , make sure your answer is strictly related to codebase.
        """
        "2 - give a short answer for the developer related to code problems in frontend and backend with function and file names for reference"
    )

def ask_fn(context, query):
    user_prompt =  f"""Here is the user question: "{query}"
    
        Relevant code context:
        {context}
    """

    client = Groq(
        # api_key="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"
        api_key= api_keys[10]
        )
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
        model="llama-3.3-70b-versatile",
    )
    time.sleep(15)
    return chat_completion.choices[0].message.content
    # return "1"
    
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

def create_workflow():
    builder = StateGraph(ChatState)

    def manage_conversation_history(state: ChatState) -> ChatState:
        # Ensure necessary keys exist
        history_summary = state.get('history_summary', "")
        messages = state.get("messages", [])
    
        if len(messages) > 2 * MAX_RECENT_MESSAGES:
            messages_to_summarize = messages[:-MAX_RECENT_MESSAGES]
            recent_messages = messages[-MAX_RECENT_MESSAGES:]
    
            message_text = "\n".join(
                f"User: {msg['query']}\nAssistant: {msg['answer']}"
                for msg in messages_to_summarize
            )
    
            prompt = (
                f"Previous conversation summary: {history_summary}\n\n"
                "Please summarize this conversation fragment:\n\n"
                f"{message_text}"
            )
    
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_keys[5]}"},
                json={
                    "model": GROQ_MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a conversation summarizer. Condense the following conversation into a brief summary "
                                "that captures the key points discussed and questions asked. Focus on the essential information "
                                "that would be needed for context in future conversation."
                            )
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                }
            ).json()
    
            history_summary = response["choices"][0]["message"]["content"]
            messages = recent_messages  # discard old messages
    
       
        final_history = history_summary + "\n\n" + "\n\n".join(
            f"User: {msg['query']}\nAssistant: {msg['answer']}" for msg in messages
        )

        print("History Generated")
        state['history'] = final_history
        return state

    def preprocess_with_llama(state): 
        messages = [
            {
                "role": "system",
                "content": """
                You are an expert at improving search queries for a large language model to execute them accurately.
                Your task is to return an improved version of a given query only if one of the following is true:
                The query is in a language other than English — in that case, translate it into English.
                The query depends on the context or content of previous queries or answers — in that case, rephrase the query ONLY IF history available so it makes complete sense independently.
                If the query is already in English and is self-contained, return it unchanged.
                Do not explain your output. Return only the final, enhanced query, or the original if no enhancement is needed.
                """
            },
            {
                "role": "user",
                "content": f"Enhance this query: {state['question']} \n History : {state['history']}"
            }
        ]
        
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_keys[9]}"},
                json={
                    "model": GROQ_MODEL,
                    "messages": messages,
                    "temperature": 0.5,
                    "max_tokens": 3000
                }
            )
            response_data = response.json()

            if "choices" in response_data and len(response_data["choices"]) > 0:
                enhanced_query = response_data["choices"][0]['message']["content"].strip()
                state["processed_question"] = enhanced_query
            else:
                print("Unexpected Format:", {response_data})
                state["processed_question"] = state["question"]
            
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            state["processed_question"] = state["question"]
        
        print('preprocessed query : ' , state["processed_question"])
        return state

    def retrieve(state) :
        global used_chunks
        used_chunks.clear()
        query = state['processed_question']
        results = search(query , top_k=50)
        results = merge_context_chunks(results , complete_cnt = 20)
        state['context_chunks'] = results
        print('chunks for context retrieved')
        return state

    def iterative_query_search(state) :
        global client
        client = Groq(api_key = api_keys[9])
        query = state['processed_question']
        context_chunks = state['context_chunks']
        context = iterative_context_expansion(query ,  context_chunks ,  max_rounds=3)
        state['context'] = context
        print('final context created')
        return state

    

    
    def generate_answers(state) :
        context_with_history = state['history'] +"\n\n" +  state['context']
        final_answer = recursive_qa(context_with_history, state['processed_question'] , ask_fn)
        state['final_answer'] = final_answer
        if not state['messages'] :
            state['messages'] = []
            
        messages = state.get("messages", [])
        messages.append({
            "query": state["question"],
            "answer": final_answer
        })

        print('answer generated')
        return state

    builder.add_node("preprocess", preprocess_with_llama)
    builder.add_node("retrieve", retrieve)
    builder.add_node("iterative_query_search", iterative_query_search)
    builder.add_node("manage_conversation_history", manage_conversation_history)
    builder.add_node("generate", generate_answers)
    
    builder.set_entry_point("manage_conversation_history")
    builder.add_edge("manage_conversation_history", "preprocess")
    builder.add_edge("preprocess", "retrieve")
    builder.add_edge("retrieve", "iterative_query_search")
    builder.add_edge("iterative_query_search","generate")
    builder.add_edge("generate", END)
        
    return builder.compile(checkpointer=memory)

memory = InMemorySaver()
workflow = create_workflow()
config = {"configurable": {"thread_id": "thread_1"}}

state = {
    "question": "",
    "processed_question": "",
    "context_chunks": [],
    "context": "",
    "final_answer": "",
    "messages": [],
    "history_summary": "",
    "history": ""
}

while True :
    state["question"] = input("Enter your query (or 'exit' to quit): ")
    if state["question"].lower() == "exit":
        break
    state = workflow.invoke(state, config=config) 
    print("\nQ:", state["question"])
    print("A:", state["final_answer"])
    print('================================')
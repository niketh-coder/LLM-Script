import os
import requests
from groq import Groq
from chat_state import ChatState
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from openai import OpenAI
from langgraph.checkpoint.memory import InMemorySaver
from utils import merge_context_chunks, search, iterative_context_expansion, recursive_qa, ask_fn

load_dotenv()
MAX_RECENT_MESSAGES = 3
memory = InMemorySaver()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"
client = OpenAI(
        api_key=OPENAI_API_KEY
)
used_chunks = set()
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.2,
    max_tokens=1000,
    # openai_api_key=API_KEY,
)

def get_used_chunks():
    return used_chunks

def create_workflow():
    builder = StateGraph(ChatState)

    def manage_conversation_history(state: ChatState) -> ChatState:
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
    
            # response = requests.post(
            #     "https://api.groq.com/openai/v1/chat/completions",
            #     headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            #     json={
            #         "model": GROQ_MODEL,
            #         "messages": [
            #             {
            #                 "role": "system",
            #                 "content": (
            #                     "You are a conversation summarizer. Condense the following conversation into a brief summary "
            #                     "that captures the key points discussed and questions asked. Focus on the essential information "
            #                     "that would be needed for context in future conversation."
            #                 )
            #             },
            #             {
            #                 "role": "user",
            #                 "content": prompt
            #             }
            #         ]
            #     }
            # ).json()

            # TODO: check here once
            response = llm.invoke(message_text)
    
            # history_summary = response["choices"][0]["message"]["content"]
            history_summary = response.content
            messages = recent_messages  
    
       
        final_history = history_summary + "\n\n" + "\n\n".join(
            f"User: {msg['query']}\nAssistant: {msg['answer']}" for msg in messages
        )

        print("History Generated")
        state['history'] = final_history
        return state

    def preprocess_with_openai(state): 
        messages = [
            {
                "role": "system",
                "content": """
                You are an expert at improving search queries for a large language model to execute them accurately.
                Your task is to return an improved version of a given query only if one of the following is true:
                The query is in a language other than English — in that case, translate it into English.
                The query depends on the context or content of previous queries or answers — in that case, rephrase the query ONLY IF history available so it makes complete sense independently.
                If the query is already in English and is self-contained, return it unchanged.
                Do not explain your output. Return only the final, enhanced query, or the original query if no enhancement is needed.
                """
            },
            {
                "role": "user",
                "content": f"Enhance this query: {state['question']} \n History : {state['history']}"
            }
        ]
        
        try:
            # response = requests.post(
            #     "https://api.groq.com/openai/v1/chat/completions",
            #     headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            #     json={
            #         "model": GROQ_MODEL,
            #         "messages": messages,
            #         "temperature": 0.3,
            #         "max_tokens": 1000
            #     }
            # )
            # response_data = response.json()

            response = llm.invoke(messages)
            response_data = response.content

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
        results = search(query , top_k=5)
        results = merge_context_chunks(results , complete_cnt = 10)
        state['context_chunks'] = results
        print('chunks for context retrieved')
        return state

    def iterative_query_search(state) :
        global client
        client = OpenAI(
            api_key=OPENAI_API_KEY
        )
        query = state['processed_question']
        context_chunks = state['context_chunks']
        context = iterative_context_expansion(query ,  context_chunks ,  max_rounds=3)
        state['context'] = context
        print('final context ceated')
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

    builder.add_node("preprocess", preprocess_with_openai)
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

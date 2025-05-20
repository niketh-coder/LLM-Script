import os
from chat_workflow import create_workflow
from huggingface_hub import login
from utils import run_pipeline

login(token=os.getenv("HF_TOKEN"))

run_pipeline()
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
queries = [ 'member not adding in club','who can be memeber ' , 'what can i do' , 'explain in more depth']

for query in queries:
    state["question"] = query
    state = workflow.invoke(state, config=config) 
    print("\nQ:", query)
    # print("A:", state["final_answer"])
    print('================================')

from groq import Groq
from google import genai

def build_file_summary_prompt(file_path, call_graph, imports):
    imports_text = "\n".join(imports) if imports else "No imports found."

    cg_lines = []
    names = call_graph.get("names", {})
    for name, data in names.items():
        calls = data.get("calls", [])
        cg_lines.append(f"- {name} calls: {', '.join(calls) if calls else 'none'}")
    call_graph_text = "\n".join(cg_lines) if cg_lines else "No calls found."
    
    prompt = (
        f"File: {file_path}\n"
        f"Imports:\n{imports_text}\n\n"
        f"Call Graph:\n{call_graph_text}\n\n"
        "Based on the above, provide a concise summary describing the purpose and structure of this file."
    )
    return prompt

def build_file_summary(prompt , api_key , LLM_MODEL):   
    client = genai.Client(api_key= api_key)
            
    response = client.models.generate_content(
        model=LLM_MODEL, contents=prompt
    )
    
    return response.text
    # client = Groq(api_key=api_key)
    # response = client.chat.completions.create(
    #             model= LLM_MODEL,
    #             messages=[
    #                 {"role": "user", "content": prompt}
    #             ]
    #         )
    # summary = response.choices[0].message.content
    # return summary


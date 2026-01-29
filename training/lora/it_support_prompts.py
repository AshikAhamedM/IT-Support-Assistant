def format_it_prompt(instruction, output):
    return f"""
<|begin_of_text|><|user|>
You are an IT support assistant.
Question: {instruction}
<|assistant|>
{output}
<|end_of_text|>
"""

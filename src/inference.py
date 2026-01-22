import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_response(prompt, model, tokenizer):
    # Prompt EXACTO del entrenamiento + "Thought:" para forzar el inicio
    full_prompt = (
        f"User: {prompt}\n"
        f"You are an agent with access to tools. Analyze the request and call the correct tool if necessary.\n"
        f"Thought:" 
    )
    
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=128, 
            do_sample=True, 
            temperature=0.1,       # Bajamos temperatura para que sea más preciso
            repetition_penalty=1.2 # Evita que repita frases
        )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

def main():
    # 1. Base Model
    base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    adapter_path = "models/TinyLlama-Agent-DPO-v1" # Your trained adapter

    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # 2. Load the DPO Adapter (The "Agent Brain")
    print("Loading DPO adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    # 3. Test Cases
    test_questions = [
        "What is 25 * 40?",
        "Who is the CEO of Tesla?",
        "Schedule a meeting for Monday."
    ]

    print("\n--- AGENT INFERENCE TEST ---\n")
    for q in test_questions:
        print(f"Question: {q}")
        response = generate_response(q, model, tokenizer)
        # We only print the new part (the response), not the repeated prompt
        print(f"Agent Response:\n{response.split('necessary.')[-1].strip()}") 
        print("-" * 50)

if __name__ == "__main__":
    main()
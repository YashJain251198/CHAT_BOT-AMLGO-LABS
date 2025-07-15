from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Use open-source model (no login needed)
model_id = "google/flan-t5-base"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to("cpu")  # or 'cuda' if available

# Answer generation function
def generate_answer(context, question):
    prompt = f"Context:\n{context}\n\nQuestion:\n{question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=256)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

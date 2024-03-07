import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from accelerate import Accelerator
import time

# loading Model and optimization configuration
def load_optimized_model(model_path: str):
    accelerator = Accelerator(fp16=True)  # using FP16 for faster computation if supported
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    model, tokenizer = accelerator.prepare(model, tokenizer)
    return model, tokenizer, accelerator

#  measuring inference time, generating text, and performing sentiment analysis
def generate_and_measure(model, tokenizer, accelerator, classifier, prompt, max_length=128):
    start_time = time.time()
    input_ids = tokenizer(prompt, return_tensors="pt", max_length=max_length, truncation=True, padding="max_length").input_ids
    input_ids = input_ids.to(accelerator.device)
    output = model.generate(input_ids, max_new_tokens=128, do_sample=True)  
    end_time = time.time()
    generated_text = tokenizer.batch_decode(output, skip_special_tokens=True)
    
    # Performing  sentiment analysis on the generated text
    classification_result = classifier(generated_text[0])
    
    return generated_text, classification_result, end_time - start_time



def main():
    model_path = "/kaggle/input/mistral/pytorch/7b-instruct-v0.1-hf/1"  # Specify the model path here
    classifier = pipeline("sentiment-analysis")  # Initialize sentiment analysis pipeline
    
    model, tokenizer, accelerator = load_optimized_model(model_path)
    
    prompt = input("Please enter your prompt: ")
    generated_text, classification_result, inference_time = generate_and_measure(model, tokenizer, accelerator, classifier, prompt)
    
    print(f"Generated Text: {generated_text[0]}")
    print("Sentiment Analysis Result:", classification_result[0])
    print(f"Inference Time: {inference_time:.4f} seconds")

    # Simulating throughput measurement 
    tokens_processed = len(tokenizer.encode(prompt)) + 128  # Input tokens + Output tokens
    throughput = tokens_processed / inference_time
    print(f"Throughput: {throughput:.2f} tokens/sec")


if __name__ == "__main__":
    main()






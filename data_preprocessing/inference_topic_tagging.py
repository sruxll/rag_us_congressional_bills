from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import argparse
import sys


# define load fine-tuned tokenizer and model function
def load_model(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSeq2SeqLM.from_pretrained(path)
    return tokenizer, model

def inference(tokenizer, model, text):
    in_tensor = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    out_tensor = model.generate(in_tensor.input_ids, max_length=64, num_beams=4, early_stopping=True)
    return tokenizer.batch_decode(out_tensor, skip_special_tokens=True)[0]

def main():
    parser = argparse.ArgumentParser(
        description="Classify the policy area of a US bill using a fine‑tuned model.",
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="The US bill text to classify (wrap in quotes)."
    )
    args = parser.parse_args()

    if not args.text:
        print("Error: no text input!!!")
        print("Example usage: \n python inference_topic_tagging.py \"This bill ...\"")

    model_dir = "./flan-t5-finetuned-bill_labels/checkpoint-26904"
    tokenizer, model = load_model(model_dir)

    prediction = inference(tokenizer, model, args.text)
    print(f"Tagged topic: {prediction}")

if __name__ == "__main__":
    main()
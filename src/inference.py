import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from peft import PeftModel

# Load base model and tokenizer
base_model_name = "google-bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Load both models
base_model = AutoModelForMaskedLM.from_pretrained(base_model_name)
base_model.eval()

# Load fine-tuned model
finetuned_base = AutoModelForMaskedLM.from_pretrained(base_model_name)
finetuned_model = PeftModel.from_pretrained(finetuned_base, "./lora_arxiv_mlm_final_1")
finetuned_model.eval()

# Move to device
device = "cuda" if torch.cuda.is_available() else "cpu"
base_model = base_model.to(device)
finetuned_model = finetuned_model.to(device)


def compare_predictions(text, top_k=5):
    # Tokenize input
    encoding = tokenizer(text, return_tensors="pt")
    input_ids = encoding["input_ids"].to(device)

    # Find mask token positions
    mask_token_id = tokenizer.mask_token_id
    mask_positions = (input_ids == mask_token_id).nonzero(as_tuple=True)[1].tolist()

    if not mask_positions:
        print("No mask tokens found in the input.")
        return

    # Get predictions from both models
    with torch.no_grad():
        base_outputs = base_model(input_ids=input_ids)
        base_predictions = base_outputs.logits

        finetuned_outputs = finetuned_model(input_ids=input_ids)
        finetuned_predictions = finetuned_outputs.logits

    # Compare predictions for each mask
    for mask_idx, mask_pos in enumerate(mask_positions):
        print(f"\n==== Mask #{mask_idx+1} ====")

        # Base model predictions
        base_token_predictions = base_predictions[0, mask_pos].softmax(dim=0)
        base_top_indices = torch.topk(base_token_predictions, top_k).indices
        base_top_values = torch.topk(base_token_predictions, top_k).values

        # Finetuned model predictions
        finetuned_token_predictions = finetuned_predictions[0, mask_pos].softmax(dim=0)
        finetuned_top_indices = torch.topk(finetuned_token_predictions, top_k).indices
        finetuned_top_values = torch.topk(finetuned_token_predictions, top_k).values

        # Display side by side
        print(f"{'Base BERT':<30} | {'Fine-tuned BERT':<30}")
        print("-" * 63)

        for i in range(top_k):
            base_token = tokenizer.convert_ids_to_tokens([base_top_indices[i]])[0]
            base_score = base_top_values[i].item()

            finetuned_token = tokenizer.convert_ids_to_tokens(
                [finetuned_top_indices[i]]
            )[0]
            finetuned_score = finetuned_top_values[i].item()

            print(
                f"{i+1}. {base_token:<15} ({base_score:.4f}) | {i+1}. {finetuned_token:<15} ({finetuned_score:.4f})"
            )


# Test with scientific sentences
examples = [
    f"The {tokenizer.mask_token} model was trained on a large corpus of scientific papers.",
    f"Researchers have observed a new phenomenon in {tokenizer.mask_token} physics experiments.",
    f"The neural network architecture includes multiple {tokenizer.mask_token} layers.",
    f"Quantum {tokenizer.mask_token} has applications in secure communication.",
    f"The {tokenizer.mask_token} of machine learning models is essential for reliable results.",
    f"Advances in computational {tokenizer.mask_token} have enabled new types of simulations.",
    f"The authors proposed a novel {tokenizer.mask_token} for analyzing genetic sequences.",
]

for i, example in enumerate(examples):
    print(f"\n\nExample {i+1}: {example}")
    compare_predictions(example)

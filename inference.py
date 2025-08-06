# inference.py

import argparse
from pathlib import Path
import torch
import pandas as pd
from Bio import SeqIO
from transformers import AutoTokenizer, EsmForSequenceClassification

# Define a default batch size. Adjust based on your GPU VRAM.
BATCH_SIZE = 16

def batch_predict(sequences: list, model, tokenizer, device: torch.device):
    """
    Predicts classes and probabilities for a batch of protein sequences.

    Args:
        sequences (list): A list of protein sequences (strings).
        model: The loaded ESM-2 model.
        tokenizer: The loaded ESM-2 tokenizer.
        device (torch.device): The device (CPU or GPU) to run inference on.

    Returns:
        list: A list of dictionaries, where each dict contains:
              'predicted_label', 'probability', and 'sequence'.
    """
    # Tokenize the entire batch
    # padding=True ensures all sequences in the batch are padded to the longest sequence length
    inputs = tokenizer(sequences, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Apply softmax to convert logits to probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=-1)

    # Get the top prediction for each sequence in the batch
    predicted_class_ids = logits.argmax(dim=1)
    
    results = []
    for i in range(len(sequences)):
        sequence = sequences[i]
        predicted_id = predicted_class_ids[i].item()
        predicted_label = model.config.id2label[predicted_id]
        probability = probabilities[i, predicted_id].item()

        results.append({
            "predicted_label": predicted_label,
            "probability": probability,
            "sequence": sequence
        })

    return results

def main():
    """Main function to run the inference script."""
    parser = argparse.ArgumentParser(
        description="Predict protein class using a fine-tuned ESM-2 model. "
                    "Supports batched inference for FASTA files."
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the saved model directory.",
    )
    parser.add_argument(
        "input_data",
        type=str,
        help="A single protein sequence or the path to a FASTA file.",
    )
    parser.add_argument(
        "-o", "--output_file",
        type=str,
        default="predictions.tsv",
        help="Path to save the output TSV file (only used for FASTA input).",
    )
    parser.add_argument(
        "-b", "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size for FASTA file processing (default: {BATCH_SIZE}).",
    )
    
    args = parser.parse_args()

    # --- 1. Setup Model and Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        model_path = Path(args.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")
            
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = EsmForSequenceClassification.from_pretrained(model_path)
        model.to(device)
        model.eval()
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- 2. Process Input ---
    input_path = Path(args.input_data)

    if input_path.is_file():
        # --- FASTA File Input (Batched) ---
        print(f"Processing FASTA file: {input_path} with batch size {args.batch_size}")
        
        all_records = list(SeqIO.parse(input_path, "fasta"))
        num_records = len(all_records)
        all_results = []
        
        for i in range(0, num_records, args.batch_size):
            batch_records = all_records[i:i + args.batch_size]
            batch_sequences = [str(r.seq) for r in batch_records]
            batch_ids = [r.id for r in batch_records]

            # Get predictions for the entire batch
            batch_predictions = batch_predict(batch_sequences, model, tokenizer, device)
            
            # Combine IDs and predictions
            for j in range(len(batch_predictions)):
                batch_predictions[j]["protein_id"] = batch_ids[j]
            
            all_results.extend(batch_predictions)

            print(f"Processed {min(i + args.batch_size, num_records)}/{num_records} records.")

        # Reorder and format columns for the final TSV
        results_df = pd.DataFrame(all_results)
        results_df = results_df[["protein_id", "predicted_label", "probability", "sequence"]]
        results_df["probability"] = results_df["probability"].apply(lambda x: f"{x:.4f}")

        results_df.to_csv(args.output_file, sep='\t', index=False)
        print(f"Predictions saved to {args.output_file}")
            
    else:
        # --- Single Sequence Input (Non-batched) ---
        print("Processing single sequence...")
        sequence = args.input_data
        
        # Predict a single sequence (we still use batch_predict, just with a batch size of 1)
        results = batch_predict([sequence], model, tokenizer, device)[0]
        
        print("\n--- Prediction Result ---")
        print(f"Sequence:         {sequence[:60]}...")
        print(f"Predicted Label:  {results['predicted_label']}")
        print(f"Probability:      {results['probability']:.4f}")


if __name__ == "__main__":
    main()
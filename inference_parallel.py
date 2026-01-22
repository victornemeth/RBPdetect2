# inference.py

import argparse
import itertools
import math
from pathlib import Path
import torch
from torch.nn import DataParallel
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm
from transformers import AutoTokenizer, EsmForSequenceClassification

# --- CONFIGURATION ---
# Define the maximum length to which all sequences will be truncated.
# This prevents OOM errors and is the standard way to handle long proteins.
MODEL_MAX_LENGTH = 2500

def batch_generator(iterator, batch_size):
    """Creates batches from an iterator without loading it all into memory."""
    while True:
        batch = list(itertools.islice(iterator, batch_size))
        if not batch:
            return
        yield batch

def predict_and_format_batch(batch_records, model, tokenizer, device, max_length):
    """
    Takes a batch of Bio.SeqRecord objects, predicts, and returns a formatted DataFrame.
    """
    sequences = [str(r.seq) for r in batch_records]
    protein_ids = [r.id for r in batch_records]

    # --- THE CORE FIX ---
    # Enforce truncation to the specified max_length directly in the tokenizer.
    # This correctly handles the warning and prevents OOM errors.
    inputs = tokenizer(
        sequences,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length  # <-- Explicitly set the max length
    )
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    # Access the model's config correctly, whether it's wrapped by DataParallel or not
    model_config = model.module.config if isinstance(model, DataParallel) else model.config

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    predicted_class_ids = logits.argmax(dim=1)
    
    results = []
    for i in range(len(sequences)):
        predicted_id = predicted_class_ids[i].item()
        # Add a note if the sequence was truncated
        original_len = len(sequences[i])
        protein_id_note = protein_ids[i]

        results.append({
            "protein_id": protein_id_note,
            "predicted_label": model_config.id2label[predicted_id],
            "probability": probabilities[i, predicted_id].item(),
            # Save the original, untruncated sequence for reference
            "sequence": sequences[i]
        })
        
    return pd.DataFrame(results)

def main():
    """Main function to run the inference script."""
    parser = argparse.ArgumentParser(
        description="Memory-efficiently predict protein class using a fine-tuned ESM-2 model on multiple GPUs."
    )
    parser.add_argument("model_path", type=str, help="Path to the saved model directory.")
    parser.add_argument("input_data", type=str, help="A single protein sequence or the path to a FASTA file.")
    parser.add_argument("-o", "--output_file", type=str, default="predictions.tsv", help="Path to save the output TSV file.")
    parser.add_argument("-b", "--batch_size", type=int, default=16, help="Total batch size for processing across all GPUs.")
    # Add an argument for the max length, making the script more flexible
    parser.add_argument("--max_length", type=int, default=MODEL_MAX_LENGTH, help=f"Sequence truncation length (default: {MODEL_MAX_LENGTH}).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using main device: {device}")

    try:
        model_path = Path(args.model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = EsmForSequenceClassification.from_pretrained(model_path)
        model.to(device)
        
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs via DataParallel.")
            model = DataParallel(model)
        model.eval()
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    input_path = Path(args.input_data)

    if input_path.is_file():
        print("Counting records in FASTA file (for progress bar)...")
        with open(input_path, 'r') as f:
            num_records = sum(1 for line in f if line.startswith('>'))
        
        print(f"Found {num_records} records. Processing with total batch size {args.batch_size}...")
        print(f"** Sequences will be truncated to a max length of {args.max_length} **")

        record_iterator = SeqIO.parse(input_path, "fasta")
        batch_gen = batch_generator(record_iterator, args.batch_size)
        total_batches = math.ceil(num_records / args.batch_size)
        
        pd.DataFrame(columns=["protein_id", "predicted_label", "probability", "sequence"]).to_csv(
            args.output_file, sep='\t', index=False
        )
        
        # The processing loop is now much simpler
        for batch_records in tqdm(batch_gen, total=total_batches, desc="Predicting", unit="batch"):
            try:
                results_df = predict_and_format_batch(
                    batch_records, model, tokenizer, device, args.max_length
                )
                results_df.to_csv(
                    args.output_file, sep='\t', index=False, mode='a', header=False
                )
            except Exception as e:
                # The robust error handling for failed batches remains a good idea
                tqdm.write(f"WARNING: A batch failed with error: {e}. Skipping...")
                continue
            
        print(f"\nPredictions saved to {args.output_file}")
            
    else: # Single sequence logic
        print("Processing single sequence...")
        sequence = args.input_data
        
        results_df = predict_and_format_batch(
            [SeqIO.SeqRecord(SeqIO.Seq(sequence), id="input_seq")], model, tokenizer, device, args.max_length
        )
        result = results_df.iloc[0]

        print("\n--- Prediction Result ---")
        print(f"Protein ID:       {result['protein_id']}")
        print(f"Predicted Label:  {result['predicted_label']}")
        print(f"Probability:      {result['probability']:.4f}")

if __name__ == "__main__":
    main()

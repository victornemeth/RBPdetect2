# inference.py

import argparse
import itertools
import math
from pathlib import Path
import torch
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm
from transformers import AutoTokenizer, EsmForSequenceClassification

# --- CONFIGURATION ---
# Define the length limit above which proteins are automatically classified
LONG_PROTEIN_THRESHOLD = 2500

def batch_generator(iterator, batch_size):
    """Creates batches from an iterator without loading it all into memory."""
    while True:
        batch = list(itertools.islice(iterator, batch_size))
        if not batch:
            return
        yield batch

def predict_and_format_batch(batch_records, model, tokenizer, device):
    """
    Takes a batch of Bio.SeqRecord objects, predicts, and returns a formatted DataFrame.
    Assumes all records in the batch are safe to predict (not too long).
    """
    sequences = [str(r.seq) for r in batch_records]
    protein_ids = [r.id for r in batch_records]

    # Tokenize the entire batch (truncation is a safeguard here)
    inputs = tokenizer(sequences, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    predicted_class_ids = logits.argmax(dim=1)
    
    results = []
    for i in range(len(sequences)):
        predicted_id = predicted_class_ids[i].item()
        results.append({
            "protein_id": protein_ids[i],
            "predicted_label": model.config.id2label[predicted_id],
            "probability": probabilities[i, predicted_id].item(),
            "sequence": sequences[i]
        })
        
    return pd.DataFrame(results)

def main():
    """Main function to run the inference script."""
    parser = argparse.ArgumentParser(
        description="Memory-efficiently predict protein class using a fine-tuned ESM-2 model."
    )
    parser.add_argument("model_path", type=str, help="Path to the saved model directory.")
    parser.add_argument("input_data", type=str, help="A single protein sequence or the path to a FASTA file.")
    parser.add_argument("-o", "--output_file", type=str, default="predictions.tsv", help="Path to save the output TSV file.")
    parser.add_argument("-b", "--batch_size", type=int, default=16, help="Batch size for FASTA file processing (default: 16).")
    args = parser.parse_args()

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

    # Get the name of label 0 for our failsafe rule
    failsafe_label = model.config.id2label[0]

    input_path = Path(args.input_data)

    if input_path.is_file():
        print("Counting records in FASTA file (for progress bar)...")
        with open(input_path, 'r') as f:
            num_records = sum(1 for line in f if line.startswith('>'))
        
        print(f"Found {num_records} records. Processing with batch size {args.batch_size}...")
        print(f"** Failsafe enabled: Proteins > {LONG_PROTEIN_THRESHOLD} AA will be automatically labeled as '{failsafe_label}' **")

        record_iterator = SeqIO.parse(input_path, "fasta")
        batch_gen = batch_generator(record_iterator, args.batch_size)
        total_batches = math.ceil(num_records / args.batch_size)
        
        # Write header to the output file first
        pd.DataFrame(columns=["protein_id", "predicted_label", "probability", "sequence"]).to_csv(
            args.output_file, sep='\t', index=False
        )
        
        for batch_records in tqdm(batch_gen, total=total_batches, desc="Predicting", unit="batch"):
            predictable_records = []
            long_records_results = []

            # --- THE NEW FAILSAFE LOGIC ---
            # 1. Split the batch into predictable and "too long" proteins
            for record in batch_records:
                if len(record.seq) > LONG_protein_threshold:
                    # For long proteins, create the result dictionary directly
                    long_records_results.append({
                        "protein_id": record.id,
                        "predicted_label": f"{failsafe_label}_by_length",
                        "probability": 1.0,
                        "sequence": str(record.seq)
                    })
                else:
                    predictable_records.append(record)
            
            # 2. Get model predictions ONLY for the predictable records
            model_predictions_df = pd.DataFrame()
            if predictable_records:
                model_predictions_df = predict_and_format_batch(
                    predictable_records, model, tokenizer, device
                )
            
            # 3. Combine results from both lists
            long_records_df = pd.DataFrame(long_records_results)
            combined_df = pd.concat([model_predictions_df, long_records_df], ignore_index=True)

            # 4. Append the combined batch results to the output file
            if not combined_df.empty:
                combined_df.to_csv(
                    args.output_file, sep='\t', index=False, mode='a', header=False
                )
            
        print(f"\nPredictions saved to {args.output_file}")
            
    else:
        # --- Single Sequence Input with Failsafe ---
        print("Processing single sequence...")
        sequence = args.input_data
        
        if len(sequence) > LONG_protein_threshold:
            print(f"** Failsafe: Sequence is > {LONG_protein_threshold} AA. **")
            result_label = f"{failsafe_label}_by_length"
            result_prob = 1.0
        else:
            # Re-use the batch prediction function for a single item
            results_df = predict_and_format_batch([SeqIO.SeqRecord(SeqIO.Seq(sequence), id="input_seq")], model, tokenizer, device)
            result = results_df.iloc[0]
            result_label = result['predicted_label']
            result_prob = result['probability']

        print("\n--- Prediction Result ---")
        print(f"Sequence:         {sequence[:60]}...")
        print(f"Predicted Label:  {result_label}")
        print(f"Probability:      {result_prob:.4f}")

if __name__ == "__main__":
    main()
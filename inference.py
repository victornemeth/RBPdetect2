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

def batch_generator(iterator, batch_size):
    """
    Creates batches from an iterator without loading the whole thing into memory.
    
    This is a memory-efficient way to handle large iterables (like a FASTA file).
    """
    while True:
        # Use itertools.islice to grab a chunk of the iterator
        batch = list(itertools.islice(iterator, batch_size))
        if not batch:
            # The iterator is exhausted, stop.
            return
        yield batch

def predict_and_format_batch(batch_records, model, tokenizer, device):
    """
    Takes a batch of Bio.SeqRecord objects, predicts, and returns a formatted DataFrame.
    """
    # Extract sequences and IDs from the BioPython records
    sequences = [str(r.seq) for r in batch_records]
    protein_ids = [r.id for r in batch_records]

    # Tokenize the entire batch
    inputs = tokenizer(sequences, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Perform inference
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    predicted_class_ids = logits.argmax(dim=1)
    
    # Format results for this batch
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
    # ... (parser arguments remain the same) ...
    parser.add_argument("model_path",type=str,help="Path to the saved model directory.")
    parser.add_argument("input_data", type=str, help="A single protein sequence or the path to a FASTA file.")
    parser.add_argument("-o", "--output_file", type=str, default="predictions.tsv", help="Path to save the output TSV file (only used for FASTA input).")
    parser.add_argument("-b", "--batch_size", type=int, default=16, help="Batch size for FASTA file processing (default: 16).")
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
        # --- FASTA File Streaming Pipeline ---
        print("Counting records in FASTA file (for progress bar)...")
        with open(input_path, 'r') as f:
            num_records = sum(1 for line in f if line.startswith('>'))
        
        print(f"Found {num_records} records. Processing with batch size {args.batch_size}...")

        # Create the stream iterator from the FASTA file
        record_iterator = SeqIO.parse(input_path, "fasta")
        
        # Create our memory-efficient batch generator
        batch_gen = batch_generator(record_iterator, args.batch_size)
        
        # Calculate the number of batches for tqdm
        total_batches = math.ceil(num_records / args.batch_size)
        
        # Open output file and write header first
        output_df = pd.DataFrame(columns=["protein_id", "predicted_label", "probability", "sequence"])
        output_df.to_csv(args.output_file, sep='\t', index=False)
        
        # Process batches from the generator
        for batch_records in tqdm(batch_gen, total=total_batches, desc="Predicting", unit="batch"):
            # Get predictions for the current batch as a DataFrame
            results_df = predict_and_format_batch(batch_records, model, tokenizer, device)
            
            # Append batch results to the TSV file without keeping them in memory
            results_df.to_csv(
                args.output_file, 
                sep='\t', 
                index=False, 
                mode='a', # 'a' for append
                header=False # Do not write header again
            )
            
        print(f"\nPredictions saved to {args.output_file}")
            
    else:
        # --- Single Sequence Input (unchanged) ---
        print("Processing single sequence...")
        sequence = args.input_data
        
        # We can reuse the batch prediction function for a single item
        results_df = predict_and_format_batch([SeqIO.SeqRecord(seq=sequence, id="input_seq")], model, tokenizer, device)
        result = results_df.iloc[0]

        print("\n--- Prediction Result ---")
        print(f"Sequence:         {sequence[:60]}...")
        print(f"Predicted Label:  {result['predicted_label']}")
        print(f"Probability:      {result['probability']:.4f}")

if __name__ == "__main__":
    main()
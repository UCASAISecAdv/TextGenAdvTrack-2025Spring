import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import pipeline
from utils.model_utils import adjust_prediction_score

def get_opts():
    arg = argparse.ArgumentParser()
    arg.add_argument(
        "--your-team-name",
        type=str,
        default="your team name",
        help="Your team name"
    )
    arg.add_argument(
        "--data-path",
        type=str,
        help="Path to the CSV dataset",
        default="your dataset path"
    )
    arg.add_argument(
        "--model-type",
        type=str,
        choices=[ "your model"]
        help="Type of model to use",
        default="your model"
    )
    arg.add_argument(
        "--result-path",
        type=str,
        help="Path to save the results",
        default="your result path"
    )
    opts = arg.parse_args()
    return opts

def get_dataset(opts):
    print(f"Loading dataset from {opts.data_path}...")
    data = pd.read_csv(opts.data_path)

    # New format: prompt, text
    dataset = data[data['text'].notna()][['prompt', 'text']].copy()
    print(f"Prepared dataset with {len(dataset)} prompts")
    
    return dataset

def get_model(opts):
    print(f"Loading {opts.model_type} detector model...")
    
    '''
    You should load your model here!
    '''
    
    print("Model loaded successfully")
    return model

def run_prediction(model, dataset, model_type):
    print("Starting prediction process...")
    prompts = dataset['prompt'].tolist()
    texts = dataset['text'].tolist()
    
    text_predictions = []
    
    start_time = pd.Timestamp.now()
    
    # Process texts
    print("Processing texts...")
    for text in tqdm(texts, desc="Text predictions"):
        try:
            prediction = model(text)
            
            # Get label and score
            pred_label = prediction[0]['label']
            pred_score = prediction[0]['score']
            
            # Adjust score - higher values indicate more likely human-written text
            final_score = adjust_prediction_score(pred_label, pred_score, model_type)
                
            text_predictions.append(final_score)
        except Exception as e:
            print(f"Error processing text: {str(e)[:100]}...")
            text_predictions.append(None)
    
    end_time = pd.Timestamp.now()
    processing_time = (end_time - start_time).total_seconds()
    
    # Create results in the requested format
    results_data = {
        'prompt': prompts,
        'text_prediction': text_predictions
    }
    
    # Create results dictionary
    results = {
        "predictions_data": results_data,
        "time": processing_time
    }
    
    print(f"Predictions completed in {processing_time:.2f} seconds")
    return results

if __name__ == "__main__":
    opts = get_opts()
    dataset = get_dataset(opts)
    model = get_model(opts)
    results = run_prediction(model, dataset, opts.model_type)
    
    # Save results
    os.makedirs(opts.result_path, exist_ok=True)
    writer = pd.ExcelWriter(os.path.join(opts.result_path, opts.your_team_name + ".xlsx"), engine='openpyxl')
    
    # Create prediction dataframe with the required columns
    prediction_frame = pd.DataFrame(
        data = results["predictions_data"]
    )
    
    # Filter out rows with None values
    prediction_frame = prediction_frame[prediction_frame['text_prediction'].notnull()]
    
    time_frame = pd.DataFrame(
        data = {
            "Data Volume": [len(prediction_frame)],
            "Time": [results["time"]],
        }
    )
    
    prediction_frame.to_excel(writer, sheet_name="predictions", index=False)
    time_frame.to_excel(writer, sheet_name="time", index=False)
    writer.close()
    
    print(f"Results saved to {os.path.join(opts.result_path, opts.your_team_name + '.xlsx')}")

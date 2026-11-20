"""Train the model from the command line.

    python train_model.py --data job_offer_acceptance_prediction_dataset.xlsx
    python train_model.py --mongo            # reads MONGODB_URI from the environment
"""
import argparse

import pandas as pd

import model_utils as mu

ap = argparse.ArgumentParser()
ap.add_argument("--data", help="Path to the .xlsx or .csv dataset")
ap.add_argument("--mongo", action="store_true", help="Load from MongoDB Atlas instead (needs MONGODB_URI)")
args = ap.parse_args()

if args.mongo:
    uri = mu.get_mongo_uri()
    if not uri:
        raise SystemExit("Set the MONGODB_URI environment variable first.")
    raw = mu.load_from_mongo(uri)
elif args.data:
    raw = pd.read_csv(args.data) if args.data.lower().endswith(".csv") else pd.read_excel(args.data, sheet_name=0)
else:
    raise SystemExit("Provide --data <file> or --mongo")

bundle = mu.train_and_save(raw)
print(f"Rows used: {bundle['n_rows']}   'Yes' rate: {bundle['yes_rate']:.1%}")
print(pd.DataFrame(bundle["metrics"]).round(3).to_string(index=False))
print(f"\nSaved best model: {bundle['model_name']} -> {mu.MODEL_PATH}")

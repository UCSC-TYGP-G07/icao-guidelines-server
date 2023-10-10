from datetime import datetime
import os
import pandas as pd


def logger(csv_path: str, image: str, key_vals: dict):
    key_vals['log_time'] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    key_vals['image'] = image
    print(key_vals)
    df = pd.DataFrame(key_vals, index=[0])
    df.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path))


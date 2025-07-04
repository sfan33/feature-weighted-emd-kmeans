import pandas as pd

def load_dataset(file_path, sheet_name=None):
    xls = pd.ExcelFile(file_path)
    if sheet_name is None:
        sheet_name = xls.sheet_names[0]
    df = pd.read_excel(xls, sheet_name=sheet_name)
    return df
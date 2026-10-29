
import pandas as pd

file_path = "data/Project_datasheet_2019-2020.xlsx"

excel = pd.ExcelFile(file_path)

print("Available sheets:", excel.sheet_names)

for sheet in excel.sheet_names:
    print("\n" + "=" * 40)
    print("SHEET:", sheet)

    df = pd.read_excel(
        file_path,
        sheet_name=sheet,
        header=None
    )

    print("Rows and columns:", df.shape)
    print(df.head(15).to_string(index=False, header=False))

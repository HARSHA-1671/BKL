import pandas as pd
def find_s_algorithm_from_csv(file_path):
    df = pd.read_csv(file_path).dropna(axis=1, how="all")
    df.columns = df.columns.str.strip()
    if "enjoy sport" in df.columns:
        df.rename(columns={"enjoy sport": "label"}, inplace=True)
    print("Dataset Preview:\n", df.head())
    print("Column Names:", df.columns.tolist())
    print("Labels Found in Dataset:", set(df["label"]))
    attributes = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].values
    hypothesis = None
    for attr, label in zip(attributes, labels):
        if str(label).strip().lower() == "yes":
            if hypothesis is None:
                hypothesis = list(attr)
            else:
                for j in range(len(hypothesis)):
                    if hypothesis[j] != attr[j]:
                        hypothesis[j] = "?"
    if hypothesis is None:
        print("\nNo positive examples ('Yes') found in dataset!")
    print("\nFinal Hypothesis:", hypothesis)
file_path = "c:/Users/hp/Downloads/data.csv"
find_s_algorithm_from_csv(file_path)

import os
import glob
import pandas as pd

def load_all_datasets(folder="dataset"):

    all_data = []

    files = glob.glob(os.path.join(folder, "*.csv"))
    files += glob.glob(os.path.join(folder, "*.tsv"))

    if not files:
        raise FileNotFoundError("No CSV or TSV datasets found!")

    for file in files:

        print("Loading:", file)

        df = pd.read_csv(
            file,
            sep=None,
            engine="python",
            encoding="latin-1",
            on_bad_lines="skip"
        )

        # Standardize column names
        df.columns = [
            str(col).strip().lower()
            for col in df.columns
        ]

        # Find message and label columns
        label_columns = [
            "label", "v1", "category", "class",
            "type", "target"
        ]

        message_columns = [
            "message", "v2", "text", "sms",
            "content"
        ]

        label_col = next(
            (c for c in label_columns if c in df.columns),
            None
        )

        message_col = next(
            (c for c in message_columns if c in df.columns),
            None
        )

        # Handle headerless two-column datasets
        if label_col is None or message_col is None:

            df = pd.read_csv(
                file,
                sep=None,
                engine="python",
                encoding="latin-1",
                header=None,
                on_bad_lines="skip"
            )

            if df.shape[1] < 2:
                print("Skipping unsupported file:", file)
                continue

            df = df.iloc[:, :2]
            df.columns = ["label", "message"]

        else:

            df = df[[label_col, message_col]].copy()
            df.columns = ["label", "message"]

        # Standardize labels
        df["label"] = (
            df["label"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        df["label"] = df["label"].replace({
            "ham": "ham",
            "spam": "spam",
            "legitimate": "ham",
            "not spam": "ham"
        })

        # Keep only known labels
        df = df[df["label"].isin(["ham", "spam"])]

        df["message"] = (
            df["message"]
            .astype(str)
            .str.strip()
        )

        df = df[df["message"] != ""]

        all_data.append(df)

    if not all_data:
        raise ValueError("No valid datasets were loaded!")

    # Combine all datasets
    combined = pd.concat(
        all_data,
        ignore_index=True
    )

    # Remove duplicate messages
    combined = combined.drop_duplicates(
        subset=["message"]
    )

    # Convert labels to numbers
    combined["label"] = combined["label"].map({
        "ham": 0,
        "spam": 1
    })

    print("\nDATASET SUMMARY")
    print("-----------------------")
    print("Total messages:", len(combined))
    print("Spam messages:", sum(combined["label"] == 1))
    print("Ham messages:", sum(combined["label"] == 0))

    return combined


# Load all datasets
data = load_all_datasets()

# Prepare X and y
X = data["message"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
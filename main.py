from preprocess import load_and_preprocess_data
from training import train_model
from testing import evaluate_model

TRAIN_PATH = "./data/train.csv"
TEST_PATH = "./data/test.csv"
TEXT_COLUMN = "Text"
LABEL_COLUMN = "Label"
MODEL_PATH = "cnn_model.h5"
TOKENIZER_PATH = "cnn_tokenizer.pkl"

def main():
    print("Step 1: Loading and preprocessing data...")
    df_train, df_test = load_and_preprocess_data(TRAIN_PATH, TEST_PATH, TEXT_COLUMN)

    print("Step 2: Training the model...")
    train_model(
        df_train, 
        text_column='cleaned_text', 
        label_column=LABEL_COLUMN, 
        model_path=MODEL_PATH, 
        tokenizer_path=TOKENIZER_PATH
    )

    print("Step 3: Evaluating the model...")
    evaluate_model(
        df_test, 
        text_column='cleaned_text', 
        model_path=MODEL_PATH, 
        tokenizer_path=TOKENIZER_PATH, 
        output_path=TEST_PATH
    )

if __name__ == "__main__":
    main()

import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

def evaluate_model(df_test, text_column, model_path, tokenizer_path, output_path):
    try:
        print("Loading the model, tokenizer, and label encoder...")
        model = load_model(model_path)
        tokenizer = joblib.load(tokenizer_path)
        label_encoder = joblib.load(f"{model_path}_label_encoder.pkl")

        print("Transforming test data...")
        X_test_seq = tokenizer.texts_to_sequences(df_test[text_column])
        X_test_padded = pad_sequences(X_test_seq, maxlen=model.input_shape[1])

        print("Making predictions...")
        y_pred_probs = model.predict(X_test_padded)
        y_pred = label_encoder.inverse_transform(y_pred_probs.argmax(axis=1))

        df_test['Label'] = y_pred

        if 'cleaned_text' in df_test.columns:
            df_test = df_test.drop(columns=['cleaned_text'])

        df_test.to_csv(output_path, index=False)
        print(f"\nPredicted labels written to {output_path}")
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")

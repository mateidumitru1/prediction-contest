import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Dropout, Bidirectional, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold
import joblib

def train_model(df_train, text_column, label_column, model_path, tokenizer_path, max_vocab_size=10000, max_sequence_length=100, embedding_dim=128):
    print("Encoding labels...")
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(df_train[label_column])
    y_train = np.array(y_train)
    joblib.dump(label_encoder, f"{model_path}_label_encoder.pkl")

    print("Tokenizing and padding text data...")
    tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(df_train[text_column])
    X_train_seq = tokenizer.texts_to_sequences(df_train[text_column])
    X_train_padded = pad_sequences(X_train_seq, maxlen=max_sequence_length)
    joblib.dump(tokenizer, tokenizer_path)

    print("Building the BiLSTM model...")
    model = Sequential([
        Embedding(input_dim=max_vocab_size, output_dim=embedding_dim, input_length=max_sequence_length),
        Bidirectional(LSTM(64, dropout=0.3, recurrent_dropout=0.3, kernel_regularizer=l2(0.001))),
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.3),
        Dense(len(label_encoder.classes_), activation='softmax')
    ])

    decay_steps = len(X_train_padded) * 30 // 16
    lr_schedule = CosineDecay(initial_learning_rate=0.001, decay_steps=decay_steps, alpha=0.1)
    optimizer = Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(f"{model_path}.keras", monitor='val_accuracy', save_best_only=True, mode='max')

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, val_index in skf.split(X_train_padded, y_train):
        X_train_fold, X_val = X_train_padded[train_index], X_train_padded[val_index]
        y_train_fold, y_val = y_train[train_index], y_train[val_index]

        print("Training the BiLSTM model...")
        model.fit(
            X_train_fold, y_train_fold,
            validation_data=(X_val, y_val),
            batch_size=16, epochs=30,
            class_weight=class_weight_dict,
            callbacks=[early_stopping, model_checkpoint]
        )

    print("Saving the model and tokenizer...")
    model.save(model_path)
    print(f"Training complete. Model and tokenizer saved at: {model_path} and {tokenizer_path}")

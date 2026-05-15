import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Load dataset
data = pd.read_csv("letter-recognition.data", header=None)

# 2. Split features & labels
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# 3. Encode labels (A-Z → 0-25)
le = LabelEncoder()
y = le.fit_transform(y)

# 4. Convert to float
X = X.astype('float32')

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Build DNN model (MULTICLASS)
model = Sequential()

model.add(Dense(64, activation='relu', input_shape=(16,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(26, activation='softmax'))  # 26 classes (A-Z)


# 7. Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 8. Train model
model.fit(X_train, y_train, epochs=25, batch_size=32)

# 9. Evaluate model
loss, acc = model.evaluate(X_test, y_test)
print("Accuracy:", acc)
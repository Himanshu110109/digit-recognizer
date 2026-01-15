import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from model import create_model

df = pd.read_csv("data/train.csv")

x = df.iloc[:, 1:].values / 255.0
y = df["label"].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = create_model()
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

model.save("digit_model.h5")

import os
import joblib
import pandas as pd

MODEL_PATH = "models/iris_model.pkl"

def test_model_exists():
    assert os.path.exists(MODEL_PATH)

def test_model_loads():
    model = joblib.load(MODEL_PATH)
    assert model is not None

def test_model_prediction():
    model = joblib.load(MODEL_PATH)

    sample = pd.DataFrame(
        [[5.1, 3.5, 1.4, 0.2]],
        columns=[
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)"
        ]
    )

    prediction = model.predict(sample)

    assert prediction[0] in [0, 1, 2]

def test_prediction_count():
    model = joblib.load(MODEL_PATH)

    sample = pd.DataFrame(
        [[5.1, 3.5, 1.4, 0.2]],
        columns=[
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)"
        ]
    )

    prediction = model.predict(sample)

    assert len(prediction) == 1

import mlflow

model_name = "mlops_fasttext"
version = 1

model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/{version}"
)

list_libs = ["Antoine Palazzolo", "Julien Pramil", "Lino Galiana"]

test_data = {
    "query": list_libs,
    "k": 1
}

import pandas as pd

results = model.predict(test_data for l in )

print(results)


print(
    pd.DataFrame(results)
)
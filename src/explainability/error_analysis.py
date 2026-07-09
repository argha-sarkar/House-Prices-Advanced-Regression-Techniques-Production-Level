import pandas as pd


def largest_errors(model, X_test, y_test):
    # 1. Generate predictions
    prediction = model.predict(X_test)

    # 2. Convert X_test to a DataFrame if it isn't one already
    # We use column names 0, 1, 2... if original names are lost
    if not isinstance(X_test, pd.DataFrame):
        result = pd.DataFrame(X_test)
    else:
        result = X_test.copy()

    # 3. Now you can safely add columns
    result["Actual"] = y_test.values if isinstance(y_test, pd.Series) else y_test
    result["Prediction"] = prediction
    result["AbsoluteError"] = (result["Actual"] - result["Prediction"]).abs()

    return result.sort_values("AbsoluteError", ascending=False)

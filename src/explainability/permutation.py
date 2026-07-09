import pandas as pd
from sklearn.inspection import permutation_importance


def permutation_analysis(
    model,
    X_test,
    y_test,
):

    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=20,
        random_state=42,
    )

    return pd.DataFrame(
        {
            "Feature": X_test.columns,
            "Importance": result.importances_mean,
        }
    ).sort_values(
        by="Importance",
        ascending=False,
    )

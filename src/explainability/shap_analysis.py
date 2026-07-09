import shap


def shap_summary(
    model,
    X_train,
):

    explainer = shap.Explainer(
        model,
        X_train,
    )

    shap_values = explainer(
        X_train,
    )

    shap.summary_plot(
        shap_values,
        X_train,
        show=False,
    )

    return shap_values

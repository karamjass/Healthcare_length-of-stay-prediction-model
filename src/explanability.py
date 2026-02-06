import shap
import lime
import lime.lime_tabular

def shap_global_explainability(model, X_train):
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)
    shap.summary_plot(shap_values, X_train)

def lime_local_explainability(model, X_train, X_test, index):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        mode='classification'
    )
    explanation = explainer.explain_instance(
        X_test[index],
        model.predict_proba
    )
    explanation.show_in_notebook()

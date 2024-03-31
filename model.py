from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline


def create_model(X, y, params) -> Pipeline:

    num_cols = X.select_dtypes(exclude=["object"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    scalae: list[tuple] = [
        (
            "one_hot",
            OneHotEncoder(sparse_output=True, handle_unknown="infrequent_if_exist"),
            cat_cols,
        ),
        ("scaler", RobustScaler(), num_cols),
    ]
    transformer: ColumnTransformer = ColumnTransformer(transformers=scalae)

    model: Pipeline = Pipeline(
        steps=[
            ("transformer", transformer),
            ("estimator", SGDClassifier(**params)),
        ]
    )

    return model

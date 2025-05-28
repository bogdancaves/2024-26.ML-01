import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

df = pd.read_csv("joined_df.csv")

x = df.tail(1000).drop(columns=['Direction', 'open', 'close']) # , 'open', 'close'
y = df.tail(1000)['Direction']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

encoder = ColumnTransformer(
    transformers=[
        ('cat_encoder', 'passthrough', make_column_selector(dtype_include='object')),
        ('num_encoder', StandardScaler(), make_column_selector(dtype_include=np.number))
    ]
)

pipe = Pipeline([
    ('encoder', encoder),
    ('classifier', RandomForestClassifier(random_state=42))
])

params = [
    {
        'encoder__cat_encoder': [OneHotEncoder(sparse_output=False, handle_unknown='infrequent_if_exist')],
        'encoder__num_encoder': [StandardScaler(copy=False), MinMaxScaler(copy=False)],
        'classifier__n_estimators': [100, 200, 500, 1000],
        'classifier__criterion': ['gini', 'entropy', 'log_loss'],
        'classifier__min_samples_split': [2, 3, 5]
        # 'classifier__max_features': ['sqrt', 'log2'],
        # 'classifier__max_samples': [1.0],
        # 'classifier__max_depth': [None],
        # 'classifier__bootstrap': [True],
        # 'classifier__min_samples_leaf': [1],
    }
]

grid_search = GridSearchCV(
    estimator=pipe, 
    param_grid=params, 
    scoring='accuracy',
    cv=3, 
    n_jobs=-1, 
    refit=True,
    verbose=4
)

grid_search.fit(x_train, y_train)

best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred_best)
print(f"Accuracy: {accuracy:.2f}")

report = classification_report(y_test, y_pred_best)
print("\nClassification Report:")
print(report)

joblib.dump(best_model, "artifact.joblib")
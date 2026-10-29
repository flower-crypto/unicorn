
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

np.random.seed(42)
n = 5000
fraud = np.random.rand(n) < 0.10
avg_amount = np.random.lognormal(4, 0.9, n)

amount = np.where(
    fraud,
    avg_amount * np.random.lognormal(0.9, 0.35, n),
    avg_amount * np.random.lognormal(0.0, 0.3, n)
)

hour = np.where(
    fraud & (np.random.rand(n) < 0.6),
    np.random.randint(0, 6, n),
    np.random.randint(6, 24, n)
)

distance = np.where(
    fraud,
    np.random.exponential(13, n),
    np.random.exponential(10, n)
)

txn_24h = np.where(
    fraud,
    np.random.poisson(3, n),
    np.random.poisson(2, n)
)

data = pd.DataFrame({
    'amount': amount,
    'avg_amount': avg_amount,
    'hour': hour,
    'distance_km': distance,
    'txn_last_24h': txn_24h,
    'Class': fraud.astype(int)
})

print(data['Class'].value_counts())


def evaluate(df):
    X, y = df.drop('Class', axis=1), df['Class']

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    sc = StandardScaler()
    X_tr, X_te = sc.fit_transform(X_tr), sc.transform(X_te)

    model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )

    pred = model.fit(X_tr, y_tr).predict(X_te)

    return [
        accuracy_score(y_te, pred),
        precision_score(y_te, pred),
        recall_score(y_te, pred),
        f1_score(y_te, pred)
    ]


before = evaluate(data)

eng = data.copy()
eng['amount_ratio'] = eng['amount'] / eng['avg_amount']
eng['is_night'] = (eng['hour'] < 6).astype(int)
eng['log_amount'] = np.log1p(eng['amount'])
eng['distance_x_txn'] = eng['distance_km'] * eng['txn_last_24h']

after = evaluate(eng)

names = ['Accuracy', 'Precision', 'Recall', 'F1-score']

print(pd.DataFrame({'Before': before, 'After': after}, index=names).round(3))

x = np.arange(4)

plt.bar(x - 0.2, before, 0.4, label='Before', color='red')
plt.bar(x + 0.2, after, 0.4, label='After', color='green')

plt.xticks(x, names)
plt.ylabel('Score')
plt.legend()
plt.title('Detector performance: before vs after feature engineering')
plt.show()
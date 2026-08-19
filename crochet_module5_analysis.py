import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

search = pd.read_csv("search_results.csv")
pagerank = pd.read_csv("pagerank_results.csv")
edges = pd.read_csv("crochet_network_edges.csv")

G = nx.DiGraph()
G.add_nodes_from(search["Website"])
G.add_edges_from(edges.itertuples(index=False, name=None))

df = search.copy()
df["Google Rank"] = np.arange(1, len(df) + 1)
df["In-Degree"] = df["Website"].map(dict(G.in_degree()))
df["Out-Degree"] = df["Website"].map(dict(G.out_degree()))
df["Website Name Length"] = df["Website"].str.len()
df["URL Length"] = df["URL"].str.len()
df["Contains Crochet"] = (df["Website"].str.lower().str.contains("crochet") | df["URL"].str.lower().str.contains("crochet")).astype(int)
df["Contains Yarn"] = (df["Website"].str.lower().str.contains("yarn") | df["URL"].str.lower().str.contains("yarn")).astype(int)
df["Contains Pattern"] = (df["Website"].str.lower().str.contains("pattern") | df["URL"].str.lower().str.contains("pattern")).astype(int)
df = df.merge(pagerank, on="Website", how="left")

features = [
    "Google Rank",
    "In-Degree",
    "Out-Degree",
    "Website Name Length",
    "URL Length",
    "Contains Crochet",
    "Contains Yarn",
    "Contains Pattern"
]

X = df[features]
y = df["PageRank"]

loo = LeaveOneOut()
predictions = np.zeros(len(df))

for train_idx, test_idx in loo.split(X):
    model = LinearRegression()
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    predictions[test_idx] = model.predict(X.iloc[test_idx])

df["Predicted PageRank"] = predictions
df["Absolute Error"] = abs(df["PageRank"] - df["Predicted PageRank"])

baseline = np.array([(y.sum() - y.iloc[i]) / (len(y) - 1) for i in range(len(y))])

metrics = pd.DataFrame({
    "Model": ["Mean Baseline", "Linear Regression"],
    "MAE": [mean_absolute_error(y, baseline), mean_absolute_error(y, predictions)],
    "RMSE": [mean_squared_error(y, baseline) ** 0.5, mean_squared_error(y, predictions) ** 0.5],
    "R2": [r2_score(y, baseline), r2_score(y, predictions)]
})

df.to_csv("crochet_module5_supervised_dataset.csv", index=False)
metrics.to_csv("crochet_module5_model_metrics.csv", index=False)
df.sort_values("Absolute Error", ascending=False).head(5).to_csv("crochet_module5_five_largest_errors.csv", index=False)

plt.figure(figsize=(10, 9))
plot_df = df.sort_values("PageRank", ascending=True)
plt.barh(plot_df["Website"], plot_df["PageRank"])
plt.xlabel("PageRank")
plt.ylabel("Website")
plt.title("PageRank of Crochet Websites")
plt.tight_layout()
plt.savefig("crochet_pagerank.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(8, 6))
plt.scatter(y, predictions)
low = min(y.min(), predictions.min())
high = max(y.max(), predictions.max())
plt.plot([low, high], [low, high], linestyle="--")
plt.xlabel("Actual PageRank")
plt.ylabel("Predicted PageRank")
plt.title("Actual vs. Predicted PageRank")
plt.tight_layout()
plt.savefig("crochet_actual_vs_predicted.png", dpi=300, bbox_inches="tight")
plt.close()

error_df = df.sort_values("Absolute Error", ascending=False).head(5).sort_values("Absolute Error")
plt.figure(figsize=(9, 5))
plt.barh(error_df["Website"], error_df["Absolute Error"])
plt.xlabel("Absolute Prediction Error")
plt.ylabel("Website")
plt.title("Five Largest PageRank Prediction Errors")
plt.tight_layout()
plt.savefig("crochet_five_errors.png", dpi=300, bbox_inches="tight")
plt.close()

model = LinearRegression()
model.fit(X, y)
coef_df = pd.DataFrame({"Feature": features, "Coefficient": model.coef_}).sort_values("Coefficient")
plt.figure(figsize=(9, 6))
plt.barh(coef_df["Feature"], coef_df["Coefficient"])
plt.axvline(0, linestyle="--")
plt.xlabel("Linear Regression Coefficient")
plt.ylabel("Feature")
plt.title("Feature Relationships with PageRank")
plt.tight_layout()
plt.savefig("crochet_feature_coefficients.png", dpi=300, bbox_inches="tight")
plt.close()

print(metrics)
print(df.sort_values("Absolute Error", ascending=False)[["Website", "PageRank", "Predicted PageRank", "Absolute Error"]].head(5))

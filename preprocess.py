import pandas as pd


df = pd.read_csv("traffic_dataset.csv")

print("All junction counts:\n")
print(df["tls_id"].value_counts())

best_tls = df.groupby("tls_id")["queue"].mean().idxmax()  #best_tls = "J12"

print(f"\nUsing junction: {best_tls}")

df = df[df["tls_id"] == best_tls]


df = df[df["queue"] > 0]


df["group"] = df["step"] // 3

df = df.groupby("group")["queue"].mean().reset_index()

df.to_csv("final_dataset.csv", index=False)

print("\n✅ Final dataset created: final_dataset.csv")
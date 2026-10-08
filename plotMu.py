import matplotlib.pyplot as plt
import pandas as pd
csv_path = "results/train_mu.csv"
df = pd.read_csv(csv_path)
plt.scatter(
    df["mu1"], df["mu2"], alpha=0.5, color="teal", edgecolors="none", s=15
)

print(df.shape)

max_labels_to_show = 20

for idx, row in df.head(max_labels_to_show).iterrows():
    plt.annotate(
        text=str(idx+1),  # The text will be the row number (0, 1, 2...)
        xy=(row["mu1"], row["mu2"]),  # Coordinates of the point
        xytext=(3, 3),  # Subtle pixel offset (x, y) so text doesn't overlap the dot
        textcoords="offset points",
        fontsize=8,
        color="darkred",
        weight="bold",
    )


plt.title("VAE Latent Structure", fontsize=14, fontweight="bold")
plt.xlabel("mu1 (Dimension 1)", fontsize=12)
plt.ylabel("mu2 (Dimension 2)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

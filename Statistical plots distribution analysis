import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
np.random.seed(42)
region_a = np.random.normal(50, 10, 200)
region_b = np.random.exponential(20, 200) + 10
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Statistical Plots - Region A vs B", fontsize=14, fontweight="bold")
axes[0, 0].hist(region_a, bins=20, alpha=0.6, color="blue", label="Region A")
axes[0, 0].hist(region_b, bins=20, alpha=0.6, color="orange", label="Region B")
axes[0, 0].set_title("Histogram")
axes[0, 0].legend()
for data, label, color in [(region_a, "Region A", "blue"), (region_b, "Region B", "orange")]:
    kde_x = np.linspace(data.min(), data.max(), 300)
    kde = stats.gaussian_kde(data)
    axes[0, 1].plot(kde_x, kde(kde_x), label=label, color=color, lw=2)
    axes[0, 1].fill_between(kde_x, kde(kde_x), alpha=0.2, color=color)
axes[0, 1].set_title("KDE")
axes[0, 1].legend()
axes[1, 0].boxplot([region_a, region_b], labels=["Region A", "Region B"], patch_artist=True)
axes[1, 0].set_title("Boxplot")
for data, label in [(region_a, "Region A"), (region_b, "Region B")]:
    print(f"\n{label}")
    print(f"  Mean     : {np.mean(data):.2f}")
    print(f"  Median   : {np.median(data):.2f}")
    print(f"  Std Dev  : {np.std(data):.2f}")
    print(f"  Skewness : {stats.skew(data):.2f}")
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    outliers = data[(data < q1 - 1.5*iqr) | (data > q3 + 1.5*iqr)]
    print(f"  Outliers : {len(outliers)}")
axes[1, 1].axis("off")
axes[1, 1].text(0.1, 0.5,
    "Region A: ~Normal, symmetric\nRegion B: Right-skewed, heavy tail",
    fontsize=12, va="center")
plt.tight_layout()
plt.savefig("plots.png", dpi=150)
plt.show()
print("\nSaved: plots.png")

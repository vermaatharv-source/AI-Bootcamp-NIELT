import matplotlib.pyplot as plt
import numpy as np

# Generate x values from 0 to 10
x = np.linspace(0, 10, 200)
y_sin = np.sin(x)
y_cos = np.cos(x)

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot Sine Wave
ax1.plot(
    x,
    y_sin,
    label="Sine Wave",
    color="tab:blue",
    linewidth=2.5,
    linestyle="-",
)
ax1.axhline(0, color="grey", linestyle="--", alpha=0.7)
ax1.set_title("Trigonometric Functions Analysis", fontsize=14, fontweight="bold")
ax1.set_ylabel("Amplitude", fontsize=12)
ax1.legend(loc="upper right")
ax1.grid(True, linestyle=":", alpha=0.6)

# Annotate a peak on the sine wave
peak_x = np.pi / 2
peak_y = np.sin(peak_x)
ax1.annotate(
    "Peak ($\pi/2, 1$)",
    xy=(peak_x, peak_y),
    xytext=(peak_x + 1, peak_y - 0.5),
    arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=6),
)

# Plot Cosine Wave
ax2.plot(
    x, y_cos, label="Cosine Wave", color="tab:orange", linewidth=2.5, linestyle="--"
)
ax2.axhline(0, color="grey", linestyle="--", alpha=0.7)
ax2.set_xlabel("Radians ($x$)", fontsize=12)
ax2.set_ylabel("Amplitude", fontsize=12)
ax2.legend(loc="upper right")
ax2.grid(True, linestyle=":", alpha=0.6)

# Adjust layout and display
plt.tight_layout()
plt.show()
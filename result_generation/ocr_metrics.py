import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Prepare your data for ONE document ---
# This is dummy data. You would load your actual CSV and filter for one 'filename'.
csv_path:path = "/home/pfont/pipeline/voting_system/document_analysis.csv"
df_doc = pd.read_csv(csv_path)


# --- 2. Create the plot ---
plt.style.use('seaborn-v0_8-whitegrid') # Use a nice academic style
fig, ax = plt.subplots(figsize=(10, 7))

# Use seaborn for easy coloring and marker styling
sns.scatterplot(
    data=df_doc,
    x='num_words',
    y='garbage_lines',
    hue='base_image',      # Color by the binarization method
    style='layout_applied', # Marker shape by layout application
    s=150,                 # Marker size
    ax=ax,
    palette='colorblind'   # A colorblind-friendly palette
)

# --- 3. Annotate and Finalize ---
# Highlight the outlier
outlier = df_doc[df_doc['version_name'] == 'Global Thresh']
ax.annotate(
    'Clear Outlier\n(Low word count, high noise)',
    xy=(outlier['num_words'], outlier['garbage_lines']),
    xytext=(outlier['num_words'] + 50, outlier['garbage_lines'] + 10),
    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
    fontsize=12,
    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", lw=1, alpha=0.5)
)

# Highlight the ideal region
ax.axvspan(550, 650, ymin=0, ymax=0.2, alpha=0.1, color='green')
ax.text(560, 40, 'Ideal Candidate Region', fontsize=12, style='italic', color='darkgreen')


ax.set_title('OCR Output Variance for a Representative Document', fontsize=16, pad=20)
ax.set_xlabel('Number of Words (Quantity)', fontsize=12)
ax.set_ylabel('Number of Garbage Lines (Noise)', fontsize=12)
ax.legend(title='Preprocessing Method', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('ocr_variance_plot.pdf') # Save as PDF for LaTeX
plt.show()
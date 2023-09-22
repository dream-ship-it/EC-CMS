import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from solver import solver
from EC_CMS import EC_CMC_al
from LWCA import LWCA

sns.set_palette("Set1")
sns.set_style("ticks")
sns.set_context(rc={'axes.labelsize': 15, 'legend.fontsize': 13,
                    'xtick.labelsize': 13, 'ytick.labelsize': 13})
# Create a 8 subplot grid
dataBase = ['Caltech20', 'Ecoli', 'LS', 'FCT', 'Aggregation', 'Texture', 'UMIST', 'SPF']
fig, axes = plt.subplots(len(dataBase), 1, figsize=(15, 120))
for idx in range(len(dataBase)):
    dataName = dataBase[idx]
    ariAll = []
    a0 = LWCA()
    s0 = solver(dataName=dataName, nums = 10, seed = 2, cntTimes = 20, algorithm = a0)
    s0.run()
    res = pd.DataFrame(['alpha', 'Proposed', 'LWEA'])
    res = pd.DataFrame(columns=['alpha', 'Proposed', 'LWEA'])
    res['LWEA'] = s0.ari
    for alpha in np.arange(0, 1.05, 0.05):
        al = EC_CMC_al(alpha = alpha, lamda = 0.01)
        so = solver(dataName=dataName, nums = 10, seed = 2, cntTimes = 20, algorithm = a1)
        so.run()
        ariAll.append(so.ari)
    res['Proposed'] = np.around(np.array(ariAll), 2)
    sns_line = sns.lineplot(x = 'alpha', y= 'ARI',
                        hue="variable", style="variable", data=res,
                        markers=True, dashes=False, ax=[idx, 0]).set()
    axes[idx, 0].set_title(dataName)
# Adjust the layout for better readability
plt.tight_layout()

# Save the figure as a PDF
plt.savefig('alpha.pdf')

# Display the figure (optional)
plt.show()
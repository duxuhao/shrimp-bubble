import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.set_style("white")
df = pd.read_csv('WPEFreDTW/B16h03m56s10sep2007y.csv')
starttime = 0
endtime = 20000
df = df[df.Width > 10]
df.Width /= 0.192
print df.Width
plt.hist(np.array(df[df.Width<500].Width),10,alpha=0.75)
plt.xlabel('Width (ms)')
plt.ylabel('Shrimp Quantity (tail)')
plt.savefig('WPEFreDTW/B16h03m56s10sep2007y.png',dpi = 600)
plt.show()

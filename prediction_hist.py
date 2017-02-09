import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('predictionlog2.csv')
starttime = 0
endtime = 147
df = df[df.Width > 15]
df.Width /= 192

plt.hist(df[(df.Time <= endtime) & (df.Time >= starttime)].Width,10,normed=1,alpha=0.75)
plt.xlabel('Width (ms)')
plt.ylabel('percentage')
plt.show()

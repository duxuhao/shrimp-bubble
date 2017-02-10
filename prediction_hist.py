import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('B17h23m35s25apr2012y.csv')
starttime = 0
endtime = 720
df = df[df.Width > 10]
df.Width /= 192

plt.hist(df[(df.Time <= endtime) & (df.Time >= starttime)].Width,25,alpha=0.75)
plt.xlabel('Width (ms)')
plt.ylabel('percentage')
plt.show()

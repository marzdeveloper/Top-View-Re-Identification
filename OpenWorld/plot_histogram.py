import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("./TVPR2/100id_30intrusi.txt", sep=",", index_col=False, names=["results", "thresholds", "label"])

df = df.apply(lambda x: x.str.strip("res: ") if x.dtype == "object" else x)
df = df.apply(lambda x: x.str.strip("thr: ") if x.dtype == "object" else x)

media_target = df["results"].loc[[df.index[-2]]].astype(str).to_string().split(":")
media_non_target = df["results"].loc[[df.index[-1]]].astype(str).to_string().split(":")
df = df.drop(df.index[-2:])
df = df.drop(axis = 1, labels = 'label')
df = df.astype(float)
#df["thresholds"] = df["thresholds"].apply(lambda x: x +0.5)

df.plot(kind='bar', title = 'TVPR2 100 id 30 intrusi triplet AVG')
plt.plot([], [], ' ',label = "fattore moltiplicativo target: " + "{:.3f}".format(float(media_target[1])))
plt.plot([], [], ' ',label = "fattore moltiplicativo non target: " + "{:.3f}".format(float(media_non_target[1])))
plt.legend()
plt.show()
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

cori_week_avg = pd.read_csv("cori_week_avg.csv")
marconi_week_avg = pd.read_csv("marconi_week_avg.csv")
# perlmutter_week_avg = pd.read_csv("perlmutter_week_avg.csv")
hawk_week_avg = pd.read_csv("hawk_week_avg.csv")

plt.figure(figsize=(10, 6))

sns.lineplot(x="timestamp_hr", y="watts", data=cori_week_avg, label="Cori")
sns.lineplot(x="timestamp_hr", y="watts", data=marconi_week_avg, label="Marconi")
# sns.lineplot(x="timestamp_hr", y="watts", data=perlmutter_week_avg, label="Perlmutter")
sns.lineplot(x="timestamp_hr", y="watts", data=hawk_week_avg, label="Hawk")

plt.xlabel("Time (hours)")
plt.ylabel("Load (MW)")
plt.title("Data Center Load Profiles")
plt.legend()
plt.show()

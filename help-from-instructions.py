import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing

# Create your df here:

all_data = pd.read_csv("profiles.csv")
print(all_data.job.head())

# Plot age frequency

plt.hist(df.age, bins=20)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.xlim(16, 80)
plt.show()

# Create drinks code for drinks feature

print(all_data.drinks.value_counts())
drink_mapping = {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5}
all_data["drinks_code"] = all_data.drinks.map(drink_mapping)
all_data['drinks_code'] = all_data['drinks_code'].replace(np.nan, -1, regex=True)
print(all_data.drinks_code.value_counts())

# Create essay length feature

essay_cols = ["essay0","essay1","essay2","essay3","essay4","essay5","essay6","essay7","essay8","essay9"]
# Removing the NaNs
all_essays = all_data[essay_cols].replace(np.nan, '', regex=True)
# Combining the essays
all_essays = all_essays[essay_cols].apply(lambda x: ' '.join(x), axis=1)
all_data["essay_len"] = all_essays.apply(lambda x: len(x))

# for data in all_data["drinks_code"]:
#     if not(np.isnan(data)):
#         print(data)

# Create final feature data to run regressions and classification

feature_data = all_data[['drinks_code', 'essay_len']]

x = feature_data.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)

feature_data = pd.DataFrame(x_scaled, columns=feature_data.columns)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def panic_level(value):
    if value >= 65:
        return 1
    elif value < 65 and value >= 40:
        return 2
    else:
        return 0

data = pd.read_csv("data/Embedded_dataset.csv")
data["panic_urgency"] = data["panic_meter"].apply(lambda x: panic_level(x))
data.drop(["panic_meter"], inplace = True, axis = 1)
sns.displot(data["panic_urgency"])
plt.show()

print(data.head)

data.to_csv("data/Embedded_dataset_panic_level.csv", index = False)
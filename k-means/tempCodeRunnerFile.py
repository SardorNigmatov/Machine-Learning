import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans


dataset = pd.read_csv("./Mall_Customers.csv")

print(dataset.head(10))

x = dataset.iloc[:,[3,4]].values

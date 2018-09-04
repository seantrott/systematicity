"""Main script."""

import pandas as pd

import matplotlib.pyplot as plt
import statsmodels.formula.api as sm


aoa = pd.read_csv("data/raw/AoA_Kuperman.csv")
aoa['word'] = aoa['Word']
word_sys = pd.read_csv("data/processed/all_words_systematicity.csv")

merged = pd.merge(aoa, word_sys)
merged['Rating'] = merged['Rating.Mean']

word_classes = pd.read_csv("data/raw/celex_classes.csv", sep="\\")
word_classes['word'] = word_classes['Word']

merged = pd.merge(merged, word_classes)

model = sm.ols(formula="impact ~ Rating + Class + CobLog + word_length", data=merged).fit()
print(model.summary())

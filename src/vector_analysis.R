library(tidyverse)
library(gridExtra)

setwd("/Users/strott/work/categorization/systematicity")

df = read_csv("metrics_all.csv")

f1 <- df %>%
  ggplot(aes(x = f1_valid, fill = shuffled)) +
  geom_histogram() +
  labs(x = "F1-score",
       title = "Linear SVC F1-score") +
  facet_grid(~component) +
  theme_classy()
f1


## Onset
true_dist = mean(filter(df, shuffled == "False" & component == "onset")$f1_valid)
df.shuffle = filter(df, shuffled == "True" & component == "onset")
p1.onset = 1 - nrow(filter(df.shuffle, f1_valid <= true_dist)) / nrow(df.shuffle)

## Coda
true_dist = mean(filter(df, shuffled == "False" & component == "coda")$f1_valid)
df.shuffle = filter(df, shuffled == "True" & component == "coda")
p1.coda = 1 - nrow(filter(df.shuffle, f1_valid <= true_dist)) / nrow(df.shuffle)

## Nucleus
true_dist = mean(filter(df, shuffled == "False" & component == "nucleus")$f1_valid)
df.shuffle = filter(df, shuffled == "True" & component == "nucleus")
p1.nucleus = 1 - nrow(filter(df.shuffle, f1_valid <= true_dist)) / nrow(df.shuffle)

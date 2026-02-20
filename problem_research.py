# Ref  : https://www.kaggle.com/code/ozermehmet/synthetic-vs-original-is-the-data-lying
# Average label conflict rate (Bayes error estimate): 0.1698
# Theoretical AUC ceiling (approx): 0.8302

# ~17.0% of samples have contradictory neighbors.
# This is why 0.955 AUC seems to be the hard ceiling on this dataset.


# Synthetic vs Original: How good is the generated data?
# In this competition, we are dealing with a massive synthetic dataset (630k+ rows). But the real question is: How well does this generated data represent the original 270-row Cleveland dataset?

# In this notebook, we won't just look at simple distributions. We will measure the statistical differences between the two datasets using Kolmogorov-Smirnov tests and Wasserstein distances. Should we add the original data to our training set? Let's find out.z

# Conclusion & Strategy Recommendation
# - Should we add the original data? Adding the 270-row original dataset to the 630,000-row training data will be a drop in the ocean. However, using the original data as a separate hold-out set in your Cross-Validation (CV) strategy makes much more sense. A model that predicts well on the original data will likely be more robust on the unseen test set (which is also generated from the same distribution).
# - Problematic Features: The distributions of Cholesterol, Max HR, and Age are slightly distorted in the synthetic data. While tree-based models (XGBoost, LightGBM) are somewhat robust to this, if you plan to build a Neural Network or Logistic Regression, you should strongly consider using a Robust Scaler or applying targeted feature engineering to handle the synthetic noise.


# for feature understanding 
# https://www.kaggle.com/competitions/playground-series-s6e2/discussion/673913


# Why scores are so compressed: The "Flipped Label" Trap

# The reason the scores are so close is that we have likely hit the Bayes Error Rate of the dataset—the theoretical limit of predictability where features simply don't contain enough information to perfectly separate the classes.

# The Stats: Original vs. Synthetic. The rate of samples where the ground truth contradicts the clinical features is remarkably consistent:

# Original Dataset: 47 flipped / 270 samples (17.41%)
# Synthetic (Train): 69,872 flipped / 630,000 samples (11.09%)
# I did a deep dive into the "flipped labels" (samples where a well-tuned model is confidently wrong) and discovered this isn't a synthetic artifact; it's inherited clinical complexity.


# When Bayes error ≈ 10–15%:
# Even perfect modeling cannot exceed:
# Accuracy ceiling ≈ 85–90%
# AUC ceiling ≈ high 0.90s but tightly bounded

# ref : https://www.kaggle.com/competitions/playground-series-s6e2/discussion/673079



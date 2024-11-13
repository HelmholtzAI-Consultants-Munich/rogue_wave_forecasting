# Prediction and Explanation of Rogue Waves

## Project Description

**Problem Definition:** Rogue waves are extreme individual waves that can pose a threat to ships and offshore platforms. The prediction of such waves could help to avoid accidents. However, the underlying mechanisms of Rogue Wave formation are not fully understood yet.

**Project Goal:** Use AI models to predict the maximum relative wave hight 𝐻/𝐻𝑠 within the upcoming time window and use eXplainable AI methods to identify the parameters that enhance the rogue wave probability.

## Approach

To achieve the above defined project goal, we will:

- train a classification model to identify the parameters that are predictive of rogue waves 
    - use an ElasticNet model that is directly interpretable via model coefficients
    - use a Random Forest model if the linear model is not capable of modelling the data due to non-linearities in the feature-target relationshop and use Random Forest Feature Importance, SHAp and FGC for interpretation of the model results
- train a regression model for forcasting the maximum relative wave hight within the upcoming 10 min
    - depending on the classification results either use an ElasticNet or Random Forest Regressor
- perform feature selection to get a predictive model with a minimum amount of features
    - iterate via inner cross validation over all feature combinations
    - use outer cross validation for choosing the best model
    - test chosen model on test set
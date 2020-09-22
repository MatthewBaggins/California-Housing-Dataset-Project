# California-Housing-Dataset-Project
My work on California Housing Dataset with Feature Engineering, building pipelines with custom transformers and testing and fine-tuning Machine Learning models.


## cali-housing-eda-feature-engineering

This is the first part of my work on California Housing Dataset, which I did on Kaggle in September 2020.

I roughly followed the second chapter of the Hands-On Machine Learning Handbooks, but also came up with many ideas on my own.

In this notebook I did the following:

### 1. Feature Engineering.

Besides the very simple ones inspired by what I found in the handbook (bedrooms_fraction and so on) I realized that there are two "price hotspots"–geographical areas in which the highest prices tend to cluster. I explored this possible dependence and wrote code to calculate the precise location of these hotspots and the distance of each district from the hotspot that's closer to it. This new feature (which I named center_distance) is highly negatively correlated (-0.54) with the variable that I was trying to predict, i.e. the median prices of a house in a given district (median_house_value).

The other engineered feature I created was the distance of a district from the ocean. This new variable, ocean_distance, also turned out to be highly negatively correlated (-0.49) with median_house_value.

### 2. Created a pipeline with custom transformers that does the following:

    1) Removes the rows with "capped" values in housing_median_age and median_house_value columns (i.e. rows which initially probably contained very high values but were then "rounded" or "cut" down to some pre-set value (52 for median_housing_age and 500001 for median_house_value)
    
    2) Calculates the distance from the ocean (ocean_distance, described above)
    
    3) Imputes values missing from the total_bedrooms column
    
    4) Calculates the distance from the closest center (center_distance, described above)
    
    5) One-hot encodes the categorical variable ocean_proximity
    
    6) Adds several new potentially valuable attributes
    
    7) Normalizes all the numerical data
    
### 3. Tested three popular models provided by Scikit-learn library with their default values as well as a simple neural network.

For all the Scikit-learn's models tested, I achieved significantly better scores than reported in the handbook, which points to the predictive power of my engineered features.

The neural network achieved the best score of all the models tested, RMSE = 41682, whereas for RandomForestRegressor (second best) RMSE=52435.

I'm going to continue working on this dataset (especially with regards to developing and fine-tuning better models) in a separate notebook, which soon will be also available at my GitHub and Kaggle pages.

## 

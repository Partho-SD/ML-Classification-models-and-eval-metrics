# ML-Classification-models-and-eval-metrics
This project explores the effectiveness of different ML classification models on a diabeties dataset. The effectiveness of the models are verified through the evaluation metrics. Early detection is preferred so Recall is prioritized.
dataset link: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset

# a. Problem statement
Despite the significant prevalence and economic burden of diabetes and prediabetes in the United States, a substantial portion of affected individuals remain undiagnosed and unaware of their condition. This lack of early detection prevents timely lifestyle interventions and effective medical treatments, leading to increased rates of severe complications, reduced quality of life, and substantial healthcare costs.

Therefore, there is a critical need for a robust and accurate classification model that can leverage readily available survey data to identify individuals as healthy, prediabetic or diabetic. Such a model, utilizing the Behavioral Risk Factor Surveillance System (BRFSS) 2015 dataset, would empower public health initiatives to proactively identify at-risk populations, facilitate early diagnosis, and ultimately mitigate the individual and societal impact of diabetes. The primary challenge lies in developing a model that can effectively handle the inherent class imbalance within the dataset while achieving high predictive performance.

# b. Dataset description
Diabetes 012 Health Indicators - BRFSS 2015

I. General Information

    Dataset Name: Diabetes 012 Health Indicators - BRFSS 2015
    Source: Behavioral Risk Factor Surveillance System (BRFSS) 2015, collected annually by the Centers for Disease Control and Prevention (CDC). Dataset obtained from Kaggle.
    Purpose: The BRFSS survey aims to monitor health-related risk behaviors, chronic health conditions, and the use of preventative services among U.S. adults. This specific cleaned dataset is curated for the purpose of classifying individuals into healthy, prediabetic/diabetic categories based on self-reported health indicators.
    Timeframe: Data collected for the year 2015.
    Geographical Scope: Represents adult residents of the United States.
II. Dataset Structure and Size

    File Format: CSV (diabetes_binary_health_indicators_BRFSS2015.csv)
    Number of Records: 253,680 individual survey responses.
    Number of Features: 21 predictive features + 1 target variable.
III. Variables Description

    Target Variable:
        Diabetes_binary: Self-reported diabetes status.
        Data Type: Categorical (Binary, represented as Integer)
        Possible Values:
            0: No diabetes or only during pregnancy
            1: Diabetes or Prediabetes
    Feature Variables:
        HighBP: High Blood Pressure (Binary: 0=No, 1=Yes)
        HighChol: High Cholesterol (Binary: 0=No, 1=Yes)
        CholCheck: Cholestrol checked in 5 years (Binary: 0=No, 1=Yes )
        BMI: Body Mass Index (Numeric, float, units: kg/m^2)
        Smoker: Smoked at least 100 cigarettes in lifetime (Binary: 0=No, 1=Yes)
        Stroke: Ever had stroke (Binary: 0=No, 1=Yes)
        HeartDiseaseorAttack: had heart disease (Binary: 0=No, 1=Yes)
        PhysActivity: Physical activity in the last 30 days (Binary: 0=No, 1=Yes)
        Fruits: Consume 1 or more fruit per day (Binary: 0=No, 1=Yes)
        Veggies: consume vegetables 1 or more times per day (Binary: 0=No, 1=Yes)
        HvyAlcoholConsump: heavy drinker(men 14+ drink/week, women 7+ drinks/week) (Binary: 0=No, 1=Yes)
        AnyHealthcare: Any healthcare coverage present (Binary: 0=No, 1=Yes)
        NoDocbcCost: couldnt affort doctor due to cost in last 12 months (Binary: 0=No, 1=Yes)
        GenHlth: General Health (Ordinal: 1-5 excellent -> poor)
        MentHlth: no of days mental health bad in last 30 days (Numeric: 0 - 30)
        PhysHlth: no of days physical health bad in last 30 days (Numeric: 0 - 30)
        DiffWalk: Difficulty walking or climbing stairs (Binary: 0=No, 1=Yes) 
        Sex: Gender 0-female, 1-male (Binary: 0=No, 1=Yes) 
        Age: 13 level age category 1 = 18-24 9 = 60-64 13 = 80 or older (Ordinal: 1-13)
        Education: Education level 1 = no school/kindergarten 4 = Grade12 or high school 6=6/6+ years college(Ordinal: 1-6)
        Income: Income scale  1 = less than $10,000 5 = less than $35,000 8 = $75,000 or more (Ordinal 1-8)
IV. Data Quality and Characteristics

    Missing Values: The dataset contains no missing values
    Data Consistency: Based on self-reported survey responses, the data may be subject to recall bias or social desirability bias
    Target Variable Distribution: predominantly 0 - Healthy, Highly unbalanced
    Feature Distributions: Distributions of BMI, Age, GenHlth, Education, Income are skewed. Other features indicate health indicators, behaviours, physical characteristics and healthcare accessibility.
V. Preprocessing and Cleaning

    This dataset is a pre-processed version of the larger BRFSS 2015 raw data. It has been reduced from 330 features to a selected 21, specifically chosen for diabetes prediction. It is assumed that initial data cleaning and encoding of categorical variables have already been performed.

# c. Models used

| ML Model Name          | Accuracy | AUC    | Precision | Recall | F1    | MCC   |
|------------------------|----------|--------|-----------|--------|-------|-------|
| Logistic Regression    |0.7227    |0.8239  |0.3066     |0.7848  |0.4409 |0.3596 |
| Decision Tree          |0.7113    |0.8119  |0.2951     |0.7721  |0.4270 |0.3407 |
| kNN                    |0.8338    |0.6866  |0.3633     |0.2562  |0.3005 |0.2135 |
| Naive Bayes            |0.7209    |0.7801  |0.2933     |0.7116  |0.4151 |0.3177 |
| Random Forest(Ensemble)|0.7226    |0.8203  |0.3048     |0.7737  |0.4373 |0.3535 |
| XGBoost (Ensemble)     |0.7528    |0.8270  |0.3287     |0.7430  |0.4557 |0.3708 |

-Observations on performance of each model
| ML Model Name          | Observation about Model Performance                                                                                          |
|------------------------|------------------------------------------------------------------------------------------------------------------------------|
| Logistic Regression    |Good balance of recall (0.78) and AUC (0.82), indicating strong ability to identify diabetics and good overall discrimination.|
|                        |Precision is low, reflecting many false positives, but high recall aligns with the goal of early detection.                   |
| Decision Tree          |Similar recall (0.77) to logistic regression but slightly lower AUC and F1, suggesting more overfitting and less robust       |
|                        |generalization. Still prioritizes recall but at the cost of overall accuracy and precision.                                   |
| kNN                    |Highest accuracy (0.83) but very low recall (0.26), meaning it misses most diabetic cases.                                    |
|                        |High accuracy is misleading due to class imbalance; not suitable when recall is critical.                                     |
| Naive Bayes            |Lower recall (0.71) and AUC (0.78) than logistic regression, with the lowest precision. Simple model, but underperforms       |
|                        |compared to others, likely due to strong independence assumptions.                                                            |
| Random Forest(Ensemble)|Recall (0.77) and AUC (0.82) are close to logistic regression, with slightly lower F1 and MCC. Handles non-linearities and    |
|                        |interactions, but does not significantly outperform simpler models here.                                                      |
| XGBoost (Ensemble)     |Best F1 (0.46) and MCC (0.37) among all, with high recall (0.74) and highest AUC (0.83). Offers the best trade-off between    |
|                        |recall and precision, making it the most effective model for early detection in this context.                                 |
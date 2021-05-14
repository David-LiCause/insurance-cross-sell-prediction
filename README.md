
# Insurance Cross Sell Prediction

## Scope

Given a dataset containing customer-level data on health insurance policyholders, predict whether these customers would also be interested in purchasing vehicle insurance from this insurance company.

## Data

Source: https://www.kaggle.com/arashnic/imbalanced-data-practice

The data contains 10 features, and includes a binary outcome (whether or not the customer is interested in purchasing vehicle insurance). The outcome has moderate class imbalance (~84% negative instances, ~16% positive instances).

## Data Cleaning

 - Inspect data
 - Cast binary and ordinal features to numeric
 - One-hot encode categorical features
 - Split data into train, validation, test sets (80/10/10 split)
 - Perform feature scaling on continuous variables
 - Identify highly correlated features

## Modeling

 1. Define evaluation metrics (account for class imbalance)
    - F1 Score: Harmonic mean of precision and recall
    - AUC of Precision-Recall Curve: The area under the curve of the precision-recall curve
    - Confusion matrix: Contingency table of predictions and labels

 2. Train linear model
    - LogisticRegression class from scikit-learn library
    - Set `class_weight` parameter to handle class imbalance. Set weights of training examples to inversely proportional to their frequencies
    - **Train evaluation metrics:**
        - F1 Score: **.57**
        - Precision-Recall AUC: **.48**
        - Confusion matrix:

            |             | Predicted 0 | Predicted 1 |
            | ----------- | :-----------: | :-----------: |
            | **Actual 0**    |   73.7%     |    26.3%    |
            | **Actual 1**    |     6.8%     |     93.2%   |
    - **Validation evaluation metrics:**
        - F1 Score: **.57**
        - Precision-Recall AUC: **.49**
        - Confusion matrix:

            |             | Predicted 0 | Predicted 1 |
            | ----------- | :-----------: | :-----------: |
            | **Actual 0**    |   73.7%     |    26.3%    |
            | **Actual 1**    |     6.9%     |     93.1%   |

3. Inspect linear relationships between features and outcome
    - Logit class from statsmodels library
    - Identify features that have a statistically significant relationship with the outcome (alpha = .05)

        |     Feature        | Coefficient | P-Value |
        | ----------- | :-----------: | :-----------: |
        |  Gender  |   -0.06     |   < .001   |
        |   Age  |    -2.40    |    < .001    |
        |  Driving_License   |    1.33    |   < .001     |
        |  Previously_Insured   |   -4.22     |    < .001    |
        |   Vehicle_Age  |   0.36     |   < .001     |
        |   Vehicle_Damage  |    2.37    |    < .001    |

4. Train random forest model
    - RandomForestClassifier class from scikit-learn library
    - Set `class_weight` parameter to handle class imbalance. Set weights of training examples to inversely proportional to their frequencies
    - Used a modified grid search to tune hyperparameters (n_estimators, max_features, max_depth, class_weight, bootstrap)
    - Compute resources: Cloud instance with 16 CPU cores, 60 GB RAM
    - **Train evaluation metrics:**
        - F1 Score: **.60**
        - Precision-Recall AUC: **.52**
        - Confusion matrix:

            |             | Predicted 0 | Predicted 1 |
            | ----------- | :-----------: | :-----------: |
            | **Actual 0**    |   76.9%     |    23.1%    |
            | **Actual 1**    |     7.1%     |     92.9%   |
    - **Validation evaluation metrics:**
        - F1 Score: **.59**
        - Precision-Recall AUC: **.52**
        - Confusion matrix:

            |             | Predicted 0 | Predicted 1 |
            | ----------- | :-----------: | :-----------: |
            | **Actual 0**    |   76.5%     |    23.5%    |
            | **Actual 1**    |     9.3%     |     90.7%   |

5. Inspect non-linear relationships between features and outcome
    - Calculate feature importance for each variable in RF model

        |     Feature        | Feature Importance |
        | ----------- | :-----------: | 
        | Vehicle_Damage | 0.627 |
        | Previously_Insured | 0.171 |
        | Age | 0.095 |
        | Policy_Sales_Channel_152 | 0.032 |
        | Policy_Sales_Channel_160 | 0.013 |
        | Vintage | 0.011 |
        | Annual_Premium | 0.010 |
        | Vehicle_Age | 0.008 |
        | Policy_Sales_Channel_26 | 0.005 |
        | Policy_Sales_Channel_124 | 0.003 |
        | Region_Code_48 | 0.002 |
        | Gender | 0.002 |
        | Region_Code_28 |  0.002 |
        | Policy_Sales_Channel_157 | 0.002 |
        | Region_Code_50 | 0.001 |
        | Policy_Sales_Channel_151 | 0.001 |
        | Region_Code_0 | 0.001 |
        | Region_Code_41 | 0.001 |
        | ... | |
       

6. Train gradient boosting model
    - Booster class from XGBoost library
    - Set `scale_pos_weight` parameter to handle class imbalance. Set weights of training examples to inversely proportional to their frequencies
    - Used a modified grid search to tune hyperparameters (learning_rate, n_round, max_depth, max_delta_step, num_features)
    - Compute resources: Cloud instance with 16 CPU cores, 60 GB RAM
    - **Train evaluation metrics:**
        - F1 Score: **.62**
        - Precision-Recall AUC: **.61**
        - Confusion matrix:

            |             | Predicted 0 | Predicted 1 |
            | ----------- | :-----------: | :-----------: |
            | **Actual 0**    |   77.9%     |    22.1%    |
            | **Actual 1**    |     4.6%     |     95.4%   |
    - **Validation evaluation metrics:**
        - F1 Score: **.59**
        - Precision-Recall AUC: **.53**
        - Confusion matrix:

            |             | Predicted 0 | Predicted 1 |
            | ----------- | :-----------: | :-----------: |
            | **Actual 0**    |   76.9%     |    23.1%    |
            | **Actual 1**    |     9.8%     |     90.2%   |

7. Inspect non-linear relationships between features and outcome
    - Calculate feature importance for each variable in GBM model

        |     Feature        | Feature Importance |
        | ----------- | :-----------: | 
        | Previously_Insured | 0.525 | 
        | Vehicle_Damage | 0.377 | 
        | Policy_Sales_Channel_152    |    0.015 | 
        | Policy_Sales_Channel_160     |   0.008 | 
        | Region_Code_48   |     0.002 | 
        | Vehicle_Age   |     0.002 | 
        | Policy_Sales_Channel_26    |    0.002 | 
        | Region_Code_28   |     0.002 | 
        | Policy_Sales_Channel_157    |    0.002 | 
        | Age    |    0.002 | 
        | Region_Code_0   |     0.002 | 
        | Policy_Sales_Channel_163    |    0.001 | 
        | Policy_Sales_Channel_151   |     0.001 | 
        | Policy_Sales_Channel_124    |    0.001 | 
        | Policy_Sales_Channel_52     |   0.001 | 
        | Policy_Sales_Channel_11    |    0.001 | 
        | Policy_Sales_Channel_156     |   0.001 | 
        | Driving_License   |     0.001 | 
        | Region_Code_50    |    0.001 | 
        | Region_Code_20   |    0.001 | 
        | Policy_Sales_Channel_61    |    0.001 | 
        | Region_Code_26    |    0.001 | 
        | Policy_Sales_Channel_155    |    0.001 | 
        

8. Estimate generalizability of trained model
    - Calculate test set evaluation metrics
    - **Test evaluation metrics:**
        - F1 Score: **.59**
        - Precision-Recall AUC: **.53**
        - Confusion matrix:

            |             | Predicted 0 | Predicted 1 |
            | ----------- | :-----------: | :-----------: |
            | **Actual 0**    |   77.2%     |    22.8%    |
            | **Actual 1**    |     9.9%     |     90.1%   |

9. Export trained model
    - Save the trained XGBoost model and associated metadata to Pickle file

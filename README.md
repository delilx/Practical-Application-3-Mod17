# Practical-Application-3-Mod17
Project focused on comparing different classifiers using a dataset on marketing bank products over the telephone.

Files for PCMLAI Assignment 17.1 (“Comparing Classifiers”)

## Description of files in repository
* 'bank-additional-full.csv' –the database containing information on a Portuguese bank’s marketing campaign efforts and results, including client attributes, information about current and previous campaigns, and indicators of Portugal’s socio-economic context.
* 'assignmentmod17_DelilMartinez.ipynb' -- Jupyter Notebook containing the code and analysis for the project
* 'OptimalDecisionTree.png' – image of the resulting optimal decision tree (the best classifier)


**Link to Jupyter notebook**: [https://github.com/delilx/Practical-Application-3-Mod17/blob/main/assignmentmod17_DelilMartinez.ipynb](https://github.com/delilx/Practical-Application-3-Mod17/blob/main/assignmentmod17_DelilMartinez.ipynb)

# Comparing Classifiers
## Project Description
The data set contains information related to a marketing campaign carried out by a Portuguese bank intended to get clients to sign up to a long-term deposit instrument. In addition to the response variable ‘y’, the set contains features that refer to client attributes (age, marital status, educational level, some information on their participation in a previous campaign), some information about the campaign, plus records of a few economic indicators of the time when the campaign was run.
The purpose of the project is to work with and compare a variety of classification models that have the variable ‘y’ as target, making use of the rest of the features.

## Data and Task Understanding 
 
| Categorical  |    | Quantitative      |
| ------------- | ------------- | ------|
| job *{admin/blue-collar/entrepreneur/... }*|| age *(integer)*|  
| marital *{divorced/married/single/unknown }*| | duration*(integer)*  [^*]|  
| education *{basic.4y/basic.6y/basic.9y/high.school/... }*| | campaign *(integer)*|
| default *{no/yes/unknown}*| | pdays *(integer)*|  
| housing *{no/yes/unknown}*| | previous *(integer)*|
| loan *{no/yes/unknown}*| | emp.var.rate *(float)*|  
| contact *{cellular/telephone}*| | cons.price.idx *(float)*|  
| month *{jan/feb/mar/.../ nov/dec}*| | cons.conf.idx *(float)*|  
| day_of_week *{mon/tue/wed/thu/fri}*|  | euribor3m *(float)*|  
| poutcome *{failure/nonexistent/success}*| | nr.employed *(integer)*|  
| ------------------------------------------------------------------------------------- | ------------- | ------------------------------|
| y *{yes/no}*|   | **Note:** This is the response variable


**Issue: Class imbalance**: The first key piece of our understanding of the data is the fact that the target variable ‘y’ is not balanced, with only 4,640 of the 41,188 observations in the set (11.3\%) corresponding to the ‘yes’ label. As a result, accuracy, which tends to be the default metric for comparing classifiers, is not really adequate. We will need to decide whether to focus on the precision or on the recall of each of the models. 
|      | Predicted label 0      | Predicted label 1 |
| ------------- | ------------- |---|
|True label 0 | true negative (TN)|false positive (FP) |
| True label 1 | false negative (FN) |true positive (TP) |

<u>Classifier Errors</u>:

* FP = false positive: the model predicts that a client will sign up for the long-term deposit when they actually won't $\rightarrow$ perhaps resources are wasted in contacting the client, trying to coax them into signing, a fruitless endeavor.

* FN = false negative: the model predicts that a client will not sign up for the instrument when in fact they will $\rightarrow$ lost business.

<u> Metric Candidates</u>:

* **Precision** = $\frac{TP}{TP + FP}$ = proportion of all the 'yes' predictions that are actually correct.

* **Recall** = $\frac{TP}{TP + FN}$ = proportion of all the actual 'yes' labels that the model was able to classify correctly.

* **f1score**  = harmonic mean of precision and recall (a combined metric)

In this bank classification problem, it would seem that focusing on capitalizing on all the potential conversions (minimizing the false negatives) would be a higher priority than minimizing the false positives. After all, the bank's agents and overall promotional material for the campaign could be seen as a relatively constant resource (the bank will likely maintain a stable agent force to work throughout the campaign).

Hence, this analysis focuses on finding the classifier that maximizes recall, with the additional understanding that both precision and accuracy are also important performance metrics of any classification model.

## Data Preparation and Preprocessing
### Feature reduction
* ‘duration’ was eliminated since the original source (Bank Marketing - UCI Machine Learning Repository) indicated that this attribute “highly affects the target output” and that it should only be included for benchmark (not predictive) purposes.  
* ‘pdays’ was used as a measure to control the quality of ‘poutcome’ and subsequently eliminated, as it included the code 999 to effectively indicate that a client had not been contacted in a previous campaign.
* ‘day_of_week’ was determined to be of little informational value for classification so it was also eliminated.
* ‘age’ was found to be extremely skewed right; a logarithmic transformation was applied to produce the artificial feature ‘lnage’ to use in the models (instead of the original ‘age’).
* ‘campaign’, originally a quantitative feature representing the number of contacts carried out with a client during the current campaign was transformed into the indicator variable ‘campaignunder10’ as it was determined that the value 10 is a practical threshold that separates the target output.
* Out of the socio-economic context attributes, the features ‘emp.var.rate’ and ‘nr.employed’ were eliminated from the set because they are highly correlated with ‘euribor3m’.
### Feature processing
All categorical features were transformed into dummy variables via OneHotEncoder, whereas quantitative features were transformed into comparable scales via MinMaxScaler, as the variables cannot be reasonably taken to follow a normal distribution (which would have warranted the use of StandardScaler).

## Model Comparison
A baseline was established using a ‘dummy classifier’ that assigns the majority label in the set to every prediction, regardless of the attribute values for any given client.
The classification models compared are: logistic regression, K-nearest neighbors (KNN), decision tree, and support vector machine (SVC), each of which was first run in a simplified version, and then fine-tuned through a grid search with cross validation (GridSearchCV) across a set of hyperparameters. The details of each model fitting can be found in the Jupyter notebook included in the repository.

Results:
|Criterion      |Simple Model Winner      |Improved Model Winner | Notes |
| ------------- | ------------- |---| ---|
|Simplicity| Dummy classifier|NA | baseline model, no training required|
| ------------------ | -------------------------------------- |------------------------------------| ---------------------------------------------------------|
|Accuracy |Decision Tree |Decision Tree | |
| | accuracy = 0.8974 | accuracy = 0.8975 | not the right metric due to |
| | training time: 0.0938s | training time: 3.12s |imbalance in outcome |
| ------------------ | -------------------------------------- |------------------------------------| ---------------------------------------------------------|
|Precision |Decision Tree |SVC | |
| | precision = 0.6786 | precision = 0.6441 | in this application not  |
| | training time: 0.0938s | training time: 384.74s |looking at optimizing this metric |
| ------------------ | -------------------------------------- |------------------------------------| ---------------------------------------------------------|
|Recall |KNN |Decision Tree | |
| | recall = 0.2299 | recall = 0.2524 | this is the metric to |
| | training time: 3.14s | training time: 3.12s |focus on |
| ------------------ | -------------------------------------- |------------------------------------| ---------------------------------------------------------|
|f1 score |KNN |Decision Tree | |
| | f1 = 0.3130 | f1 = 0.3587 | this is a combined metric |
| | training time: 3.14s | training time: 3.12s |showing DT as winner as well |

Clearly the best classification model for this application is the optimized decision tree (with hyperparameters criterion = entropy and max_depth = 6), with the added advantage that it is also extremely efficient with a training time of 3.12 seconds.

Although the resulting recall may not be particularly impressive at 25.24\%, the optimized decision tree classifier also produces the second highest precision at 61.94\%, and even an accuracy that exceeds that of the basic, majority-based dummy classifier at 89.75\% (compared to 88.65\% for the dummy classifier). In fact, this optimal decision tree also maximizes the f1 score, the harmonic mean of precision and recall.

A detailed graph of the resulting optimized decision tree is also included in the repository (‘OptimalDecisionTree.png’) for reference.

## Findings of the Analysis / Actionable Items

With the understanding that the economic atmosphere of the country (Portugal) is critical in terms of individuals' financial/investment decisions, decision-makers can also focus on the following characteristics which appear to be aligned with a higher conversion rate for the long-term deposit instrument involved in the campaign:

Clients' attributes: in truth, **clients of all ages** signed up for the long-term deposit, though the distribution of ages in the successful campaign group is flatter and more highly skewed right, which is consistent with the job category **'job-retired'**. Other job categories that should be kept "in the radar" are **management** and **technician**, though the **"unemployed"** and **"unknown"** categories should not be dismissed and ought to be explored deeper, as they associate with close-to-average conversion rates. Finally, the one education level category that the model indicates should be flagged is **education_basic.9yr, which has the lowest conversion rate** among all the education level categories.

In terms of the actual contacts carried out with clients, there is a clearly higher success rate when these are done via cellphone (**contact_cellular**) versus a landline (contact_telephone). As for the **number of contacts**, if the bank agents have reached **9** contacts with a client without being able to close the conversion, our analysis suggests (and the results of the classifier confirm this) that the likelihood that such a client will sign up for the long-term deposit instrument being offered is very low -this is captured by the artificial feature campaignunder10. Thus, management may wish to set up a rule to decide whether additional resources (agents' time) ought to be invested in pursuing the conversion. Finally, as mentioned earlier, there are variations in the degree of success of the campaign throughout the year, and the classifier has identified as relevant encoded features **month_oct (a month with great success rate), month_apr (middle of the road) and month_may (the month with the lowest success rate)** to form a decision. It is useful to keep in mind that the time of the year is likely closely linked to the country's economic conditions, so some additional care is necessary when issuing generalizations about the results.
Speaking of the socio-economic context features, it seems like the campaign is more successful when **'euribor3m'** is lower (which correlates with points in time when **'nr.employed'** is also lower, and **'emp.var.rate'** takes on negative values), which suggests that clients tend to invest in the long-term deposit during somewhat pessimistic times.  

### Limitations of the Project and Potential Next Steps

Throughout the project we detected some inconsistencies in the data (for instance, clients whose 'pdays' attribute was coded as 999, meaning that they had not participated in the previous campaign, but that then had "failure" in the 'poutcome' feature). Like with any other data-intensive project, it is important to keep in mind that the quality of the input will necessarily affect the quality of the results.

A couple of ideas that appear to be easy to implement and could enrich the analysis include keeping track of the specific agent (or their experience level) associated with each client, and, since the campaign seems to have been run through personal contact (via cellphone or landline), it may even be useful to keep track of how the agent's characteristics match with those of the client -e.g. age, gender, etc.

A final observation is that although it initially appeared as if the features in the set were evenly split between categorical and quantitative, it could be argued that in fact most of the information ended up being categorical: 'duration' was dropped as suggested in the data source; 'campaign' was used only as a threshold to separate the target outcome classes; 'pdays' was partly categorical and was dropped from the modeling; (the exceptions are the clients' ages, the socio-economic context features, and the 'previous' feature). Including additional quantitative features (such as clients' income, how long they have been clients at the institution, family size, loan amounts for those who have them, etc.) may contribute to producing superior classifiers.

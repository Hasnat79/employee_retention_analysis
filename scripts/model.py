import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from joblib import dump
from joblib import load
import pickle as pkl

def plot_distribution(y_train, save_path=None):
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))

    # Assuming 'LeaveOrNot' is binary (0 and 1)
    ax = sns.countplot(x='LeaveOrNot', data=pd.DataFrame(y_train, columns=['LeaveOrNot']))

    plt.title('Distribution of LeaveOrNot in Training Data')
    plt.xlabel('LeaveOrNot')
    plt.ylabel('Count')

    total = len(y_train)
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height() / total)
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        ax.text(x, y, percentage, ha='center', va='bottom')
    plt.savefig(save_path)
    

# Loading dataset
file_path = "../data/Employee.csv"  # Replace with the actual file path
df = pd.read_csv(file_path)
# print(df.columns.tolist()) #['Education', 'JoiningYear', 'City', 'PaymentTier', 'Age', 'Gender', 'EverBenched', 'ExperienceInCurrentDomain', 'LeaveOrNot']
# print(df.count()) #4653

#----------------------------------------------------------------------------------------------------
# creating Age and Gender Groups
df['AgeGroup'] = pd.cut(df['Age'], bins=[-float('inf'), 30, float('inf')], labels=['<30', '>=30'])

# print(df.columns.tolist())
#----------------------------------------------------------------------------------------------------

# -processing-
# converting the non-numerical attributes to one-hot vecotor
df_encoded = pd.get_dummies(df,columns = ['Education','City','PaymentTier','EverBenched','ExperienceInCurrentDomain'])
# print(df_encoded.count()) #4653
# print(df_encoded.head())
# print(df_encoded.columns.tolist()) #[5 rows x 23 columns] ~ 23 attributes now
# print(len(df_encoded.columns.tolist())) #[5 rows x 23 columns] ~ 23 attributes now
#----------------------------------------------------------------------------------------------------

# Splitting the data into training and testing sets (70/30)
X = df_encoded.drop(['LeaveOrNot'],axis = 1)
y = df_encoded['LeaveOrNot']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# distribution of predictive feature 'LeaveOrNot' in training data
# print(y_train.count())#3257
# print(y_test.count())#1396
# print(pd.Series(y_train).value_counts())#0-->2133 ~ 1-->1124
plot_distribution(y_train, "../figures/X_train_y_train_leaveOrnot_distribution_plot.png")
#----------------------------------------------------------------------------------------------------

#removing protected attributes from training data X_train, X_test
protected_attributes = ['Age', 'Gender', 'AgeGroup']
X_train = X_train.drop(columns=protected_attributes) #20 ~ total features for the model
__X_test = X_test # backup for prediction analysis 
# print(__X_test.columns.tolist())
X_test = X_test.drop(columns=protected_attributes) #20 ~ total features for the model
# print(__X_test.head())
#----------------------------------------------------------------------------------------------------

# Initialize, train and save the Random Forest Classifier
for n in [50,100,200]:
    rf_clf = RandomForestClassifier(n_estimators=n, random_state=42)

    # train
    rf_clf.fit(X_train, y_train)
    dump(rf_clf,f'../checkpoints/random_forest_model_{n}_estimators.pkl')
#--------------------------------------------------------------------------------

# loading the 200 estimators model
rf_clf = load("../checkpoints/random_forest_model_200_estimators.pkl")

# prediction on X_test
y_pred = rf_clf.predict(X_test)
y_pred_proba = rf_clf.predict_proba(X_test)

# saving the predictions, predictions probabilities and ground truths in one dataframe with protected attributes
__X_test['y_test'] = y_test
__X_test['y_pred'] = y_pred
__X_test['y_pred_proba'] = y_pred_proba[:, 1]#probability of leaving
# print(__X_test.head())
#--------------------------------------------------------------------------------

# evaluation 
# Detecting bias of the model in Age
groups = ['<30', '>=30','Male','Female']
for group in groups:
    if group == '<30' or group == '>=30':
        subgroup = __X_test[__X_test['AgeGroup'] == group]['y_test']
        subgroup_pred = __X_test[__X_test['AgeGroup'] == group]['y_pred']
        subgroup_pred_proba = __X_test[__X_test['AgeGroup'] == group]['y_pred_proba']
    else: 
        subgroup = __X_test[__X_test['Gender'] == group]['y_test']
        subgroup_pred = __X_test[__X_test['Gender'] == group]['y_pred']
        subgroup_pred_proba = __X_test[__X_test['Gender'] == group]['y_pred_proba']
    
    print(f"Leave Rate in {group} group: {subgroup.mean():.2f}")
    #calculate false-positive error - type 1 error, false-negative error - type 2 error
    tn, fp, fn, tp = confusion_matrix(subgroup, subgroup_pred).ravel()
    print(f"type 1 error rate in {group} group: {fp / (fp + tn):.2f}")
    print(f"type 2 error rate in {group} group: {fn / (fn + tp):.2f}")
    
    # save calibration curve for each group
    prob_true, prob_pred = calibration_curve(subgroup,subgroup_pred_proba, n_bins=10, strategy='uniform')
    line = mlines.Line2D([0, 1], [0, 1], color='black', linestyle='--')
    fig, ax = plt.subplots()
    ax.add_line(line)
    plt.plot(prob_pred, prob_true, marker='o', label=group)
    #save plot
    plt.xlabel('Predicted Probability of the model')
    plt.ylabel('True Probabilities')
    plt.title('Calibration Curve for Different Groups')
    plt.legend()
    plt.savefig(f'../figures/calibration_curve_{group}.png')
#--------------------------------------------------------------------------------
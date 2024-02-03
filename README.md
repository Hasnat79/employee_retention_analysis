# Unmasking Bias: Employee Retention Analysis
This project aims to predict employee retention and detect bias in the model using various features like 'Education', 'JoiningYear', 'City', 'PaymentTier', 'Age', 'Gender', 'EverBenched', 'ExperienceInCurrentDomain'. The dataset used is 'Employee.csv' located in the [data](asg%201/data/Employee.csv) directory.

## Project Structure

The project has the following structure:

- `checkpoints/`: Contains the trained models.
- `data/`: Contains the dataset used for the project.
- `figures/`: Contains any figures or plots generated during the project.
- `scripts/`: Contains the scripts used for the project.

## Dataset
The employee.csv dataset has 8 features. They are: 'Education', 'JoiningYear', 'City',
'PaymentTier', 'Age', 'Gender', 'EverBenched', and 'ExperienceInCurrentDomain'. The predictive
feature for our task is ‘LeaveOrNot’. In addition, the features that have categorical features are:
‘Education’, ‘City’, ‘PaymentTier’, ‘EverBenched’, and 'ExperienceInCurrentDomain’. In our
model training, we will treat the following features as protected features: ‘Age’ and ‘Gender’. The
data distribution of the label is 65.5% (Not Leave) and 34.5% (Leave) suggesting the imbalance
of label.
![](asg%201/figures/X_train_y_train_leaveOrnot_distribution_plot.png)
Figure 1: Label Distribution

## Models

The models are trained using the Random Forest Classifier with different numbers of estimators. The trained models are saved in the [checkpoints](asg%201/checkpoints/) directory.

- [Random Forest Model with 50 estimators](asg%201/checkpoints/random_forest_model_50_estimators.pkl)
- [Random Forest Model with 100 estimators](asg%201/checkpoints/random_forest_model_100_estimators.pkl)
- [Random Forest Model with 200 estimators](asg%201/checkpoints/random_forest_model_200_estimators.pkl)

## Scripts

The main script for the project is [model.py](asg%201/scripts/model.py) which contains the code for training the models.

## Usage

To run the project, navigate to the scripts directory and run the model.py script.

```sh
cd scripts/
python model.py
```

## Bias Analysis based on Age Group  and Gender

### Bias in Age

Age Group | True Leave Rate| Type 1 (false positive)| Type 2 (false negative)|
|---|---|---|--|
|<30 |37%| 5%|40%|
|>=30 |27%| 6% |35%|
Table 1: Comparison of metrics between age groups

**Employee Below 30 years of Age:** 37% of the employees below 30 actually left the job. From
table 1, we observe that 40% of the times the model predicted an employee under 30 would not
leave but actually they did. If we look at the calibration curve (Figure 2) below, the model is
generally well-calibrated for this age group because half of the points are on the diagonal line.
Hence,there may be a slight bias in the model towards underpredicting that employees under 30
will leave.
![](asg%201/figures/calibration_curve_<30.png)
Figure 2: Age Group Calibration Curve: '<30'

**Employee Above or Equal 30 years of Age:** 27% of the employees aged 30 or above actually
left the company. Table 1 indicates that for 35% of the times the model predicted that an
employee aged 30 or above would not leave the company but they actually did. However, from
the calibration curve (Figure 3) below, we observe that the curve is mostly below the diagonal. It
means that the model is more likely to predict that an employee aged 30 or above will leave
when in reality they do not. It is over-confident or biased in predicting towards employees aged
30 or above that they will leave.**

![](asg%201/figures/calibration_curve_>=30.png)
Figure 3: Age Group Calibration Curve: '>=30'

### Bias in Gender

Gender Group| True Leave rate| Type 1 (false positive) |Type 2 (false negative)|
|---|---|---|--|
|Male |24% |6% |45%|
|Female| 48%| 5% |34%|
Table 1: Comparison of metrics between gender groups

**Male Group:** 24% of the male employees actually left. The type 1 error suggests that 6% of
times the model predicted a male employee would leave whereas they actually did not. As the
type 1 error is quite low, the model is quite good at predicting when a male employee will not
leave. On the contrary, the type 2 error is quite high. It suggests that the model often fails to
identify male employees who will leave. Moreover, the calibration curve (Figure 4) for the male is
mostly below the diagonal, which means that the model is over-predicting the probability that
male employees leave. Overall, the results suggest that there is a bias in the model towards
over-predicting that the male employees will leave.

![](asg%201/figures/calibration_curve_Male.png)
Figure 4: Gender Calibration Curve: Male

**Female Group:** Table 1 suggests that 48% of the female employees actually left. The Type 1
error (5%) is quite low which means that the model is quite good at predicting when a female
employee will not leave. On the other hand, type 2 error (34%) is quite high. It indicates that the
model often fails to identify the female employees who will leave the company. In addition, the
calibration curve (Figure 5) for the female group is mostly above the diagonal line. It means that
the model is under-predicting or under-confident that female employees will leave. Overall, there
is a bias in the model towards under-predicting that female employees will leave, meaning that it
under-identifies female employees who will actually leave.

![](asg%201/figures/calibration_curve_Female.png)
Figure 5: Gender Calibration Curve: Female

### Contact
Feel free to reach out if you have any questions or suggestions!
- 📧 Email: hasnatabdullah79@gmail.com
- 💼 LinkedIn: [Hasnat Md Abdullah ](https://www.linkedin.com/in/hasnat-md-abdullah/)

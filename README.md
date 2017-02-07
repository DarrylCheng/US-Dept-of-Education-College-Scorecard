
# US Dept of Education - College Scorecard

## Questions
### 1. Given ACT score in English, writing and Math, what are your expected earning income after you graduate?
### 2. Does studying in public college earns more than studying in private college?
### 3. Does tuition fee affect the income of graduate?
### 4. Does SAT score affect the income of graduate?
### 5. Which sector of ACT scores contributes the most to the income of graduate?

# Libraries and modules imported


```python
import pandas as pd
import numpy as np
import sqlite3 
import math
import matplotlib.pyplot as plt
import sklearn
import scipy.stats as stats
import seaborn as sns
from sklearn import preprocessing
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')
```

# Extracting data from the database file

The database file and the College Scorecard.csv are the same, we will be using the database file to extract features that will be useful in our data analysis.


```python
con = sqlite3.connect('output/database.sqlite')
data = pd.read_sql('SELECT UNITID, INSTNM, cast(SAT_AVG as int)SAT_AVG, cast(ACTENMID as int)ACTENMID, cast(ACTMTMID as int)ACTMTMID,\
                  cast(ACTWRMID as int)ACTWRMID, cast(TUITIONFEE_PROG as int)TUITIONFEE_PROG, cast(MD_EARN_WNE_P10 as int)MD_EARN_WNE_P10,\
                  cast(UNEMP_RATE as int) UNEMP_RATE, cast(MD_EARN_WNE_P6 as int)MD_EARN_WNE_P6, cast(ADM_RATE as int) ADM_RATE,\
                  cast(TUITIONFEE_IN as int) TUITIONFEE_IN,cast(TUITIONFEE_OUT as int) TUITIONFEE_OUT, CONTROL,STABBR from Scorecard', con)
df_copy = data.copy()
```

# Exploratory data analysis
Before jumping into data analysis, we must first have a thorough understanding of our data. <br>
First, we take a look at a brief summary of our data.


```python
df_copy.head(3)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>UNITID</th>
      <th>INSTNM</th>
      <th>SAT_AVG</th>
      <th>ACTENMID</th>
      <th>ACTMTMID</th>
      <th>ACTWRMID</th>
      <th>TUITIONFEE_PROG</th>
      <th>MD_EARN_WNE_P10</th>
      <th>UNEMP_RATE</th>
      <th>MD_EARN_WNE_P6</th>
      <th>ADM_RATE</th>
      <th>TUITIONFEE_IN</th>
      <th>TUITIONFEE_OUT</th>
      <th>CONTROL</th>
      <th>STABBR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100636</td>
      <td>COMMUNITY COLLEGE OF THE AIR FORCE</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Public</td>
      <td>AL</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100654</td>
      <td>ALABAMA A &amp; M UNIVERSITY</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Public</td>
      <td>AL</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100663</td>
      <td>UNIVERSITY OF ALABAMA AT BIRMINGHAM</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Public</td>
      <td>AL</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_copy.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>UNITID</th>
      <th>SAT_AVG</th>
      <th>ACTENMID</th>
      <th>ACTMTMID</th>
      <th>ACTWRMID</th>
      <th>TUITIONFEE_PROG</th>
      <th>MD_EARN_WNE_P10</th>
      <th>UNEMP_RATE</th>
      <th>MD_EARN_WNE_P6</th>
      <th>ADM_RATE</th>
      <th>TUITIONFEE_IN</th>
      <th>TUITIONFEE_OUT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.246990e+05</td>
      <td>18598.000000</td>
      <td>13222.000000</td>
      <td>13207.000000</td>
      <td>1445.000000</td>
      <td>30782.000000</td>
      <td>19311.000000</td>
      <td>32337.000000</td>
      <td>32688.000000</td>
      <td>34156.000000</td>
      <td>58024.000000</td>
      <td>56844.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>6.929501e+05</td>
      <td>1052.701043</td>
      <td>22.316669</td>
      <td>21.994018</td>
      <td>10.215225</td>
      <td>11574.881132</td>
      <td>29575.133344</td>
      <td>3.337292</td>
      <td>25054.084068</td>
      <td>0.106043</td>
      <td>10587.185768</td>
      <td>12616.974632</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.599043e+06</td>
      <td>126.892749</td>
      <td>3.548055</td>
      <td>3.290569</td>
      <td>14.849524</td>
      <td>6614.404962</td>
      <td>17312.794769</td>
      <td>1.383665</td>
      <td>14172.591658</td>
      <td>0.312428</td>
      <td>8856.046236</td>
      <td>8138.403437</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.006360e+05</td>
      <td>514.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.606670e+05</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.083180e+05</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.800940e+05</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>max</th>
      <td>4.828571e+07</td>
      <td>1599.000000</td>
      <td>36.000000</td>
      <td>35.000000</td>
      <td>530.000000</td>
      <td>88550.000000</td>
      <td>250000.000000</td>
      <td>15.000000</td>
      <td>133600.000000</td>
      <td>10.000000</td>
      <td>70024.000000</td>
      <td>70024.000000</td>
    </tr>
  </tbody>
</table>
</div>



By looking at the output of .describe(), we can make a few conclusion that.

A lot of features have missing values, this conclusion is drawn by looking at the _count_ output. <br>
Furthermore by looking at the min values of variables like MD_EARN_WNE_P6 and ADM_RATE, which is zero which are logically wrong. <br> <br>
By looking further into the data, we can see that a lot of our data are missing (Labeled NaN). 
Unfortunately, running the dropna() command on the dataframe results in loss of <b>all</b> data as each row has at least one or two missing values. Hence data cleaning will be performed individually when we want to use the data.
<hr>

## Descriptive Statistics

For each of the numerical variables, a histogram will be plotted out to see the distribution of data as well as to identify any outliers present in the variable. For non-numerical variables, barplots will be used.

We take a look at the distribution of the data to better understand our data as well as to make conclusions on data. For instance, we can find out the similarities between the data or where most of the data converge as just looking at the mean value sometimes is not accurate enough since mean can be easily affected by outliers.

For each of the graph, a short description of the data is printed out (max value, min value, standard deviation et cetera), as well as the variance of the data.<br> Variance very close to zero means that most of the data are similiar, while high variance often mean that most data is distinct. Hence by taking the variance value into consideration, we can remove features that carry little information. Important thing to note is that the data needs to be normalized before finding the variance.


```python
min_max_scaler = preprocessing.MinMaxScaler() #For normalization
# Clean NaN values from the variable we want to check
def cleanVariable(variable_to_check):
    df_SAT = df_copy.copy()
    #df_SAT = df_SAT[~np.isnan(df_SAT[variable_to_check])] #Remove NaN values
    df_SAT = df_SAT[df_SAT[variable_to_check]>0]
    return df_SAT
```


```python
variable_to_check = "SAT_AVG"
dfplt = cleanVariable(variable_to_check)
plt.hist(dfplt[variable_to_check], bins=10)
plt.xlabel("Average student SAT Scores of college")
plt.ylabel("Freq")
plt.title("Distribution of average SAT scores of students in each college")
plt.show()

print("Variance: ", np.var(min_max_scaler.fit_transform(dfplt[variable_to_check]))) 
print(dfplt[variable_to_check].describe())
print (min_max_scaler.fit_transform(dfplt[variable_to_check]))
```


![png](output_11_0.png)


    Variance:  0.0136769979888
    count    18598.000000
    mean      1052.701043
    std        126.892749
    min        514.000000
    25%        970.000000
    50%       1035.000000
    75%       1115.000000
    max       1599.000000
    Name: SAT_AVG, dtype: float64
    [ 0.4202765   0.47557604  0.5483871  ...,  0.44976959  0.4562212
      0.35391705]
    

From the histogram of the distribution of average SAT scores of students in college, we may conclude that if you have an overall SAT score of around 900~1200 you may be accepted to enroll in most college. However looking at the histogram, there are some university where the average SAT score of students enrolled are higher or lower than average. We will take a look on those colleges. <br><br>


```python
dff = df_copy.copy()
dff = dff[dff["SAT_AVG"] > 0]
dff["INSTNM"] = dff["INSTNM"].str.upper()
dff.drop_duplicates("INSTNM", inplace=True)
a = dff.sort_values('SAT_AVG', ascending=False).head(10)
sns.barplot(a["SAT_AVG"], a["INSTNM"], orient="h")
plt.ylabel("Institute name")
plt.xlabel("Average SAT Score of students enrolled")
plt.title("Top 10 college by average SAT Score")
plt.show()

b = dff.sort_values('SAT_AVG', ascending=True).head(10)
sns.barplot(b["SAT_AVG"], b["INSTNM"], orient="h")
plt.ylabel("Institute name")
plt.xlabel("Average SAT Score of students enrolled")
plt.title("Bottom 10 college by average SAT Score")
plt.show()
```


![png](output_13_0.png)



![png](output_13_1.png)


We can see that colleges/university like Harvard and Massachusetts institute of technology have students with relatively high SAT scores. This information are useful for students who wants to enter these college as to improve themselves in getting higher SAT scores to have a higher chance of getting in. <br><br>


```python
variable_to_check = "TUITIONFEE_PROG"
dfplt = cleanVariable(variable_to_check)
plt.hist(dfplt[variable_to_check], bins=10)
plt.xlabel("Tuition fee of each college")
plt.ylabel("Freq")
plt.title("Distribution of tuition fee of each college")
plt.show()

print("Variance: ", np.var(min_max_scaler.fit_transform(dfplt[variable_to_check]))) 
print(dfplt[variable_to_check].describe())
```


![png](output_15_0.png)


    Variance:  0.00558676982406
    count    30778.000000
    mean     11576.385438
    std       6613.518286
    min         70.000000
    25%       7331.000000
    50%      10350.000000
    75%      14603.000000
    max      88550.000000
    Name: TUITIONFEE_PROG, dtype: float64
    

Looking in the histogram of distribution of tuition fee, we find that the tuition fee is relatively cheap for most college as the tuition fee for most colleges are below average the of $8812. <br><br>


```python
variable_to_check = "TUITIONFEE_IN"
dfplt = cleanVariable(variable_to_check)
plt.hist(dfplt[variable_to_check], bins=20)
plt.xlabel("In-state tuition fee")
plt.ylabel("Freq")
plt.title("Distribution of in-state tuition fee of University")
plt.show()
print("Variance: ", np.var(min_max_scaler.fit_transform(dfplt[variable_to_check]))) 
print(dfplt[variable_to_check].describe())

variable_to_check = "TUITIONFEE_OUT"
dfplt = cleanVariable(variable_to_check)
plt.hist(dfplt[variable_to_check], bins=20)
plt.xlabel("Out-of-state tuition fee")
plt.ylabel("Freq")
plt.title("Distribution of out-of-state tuition fee of University")
plt.show()
print("Variance: ", np.var(min_max_scaler.fit_transform(dfplt[variable_to_check]))) 
print(dfplt[variable_to_check].describe())
```


![png](output_17_0.png)


    Variance:  0.0159950644484
    count    57984.000000
    mean     10594.489290
    std       8854.732166
    min         11.000000
    25%       3285.000000
    50%       8505.000000
    75%      15200.000000
    max      70024.000000
    Name: TUITIONFEE_IN, dtype: float64
    


![png](output_17_2.png)


    Variance:  0.013504664978
    count    56818.000000
    mean     12622.748178
    std       8135.787628
    min         15.000000
    25%       6550.000000
    50%      10896.500000
    75%      16600.000000
    max      70024.000000
    Name: TUITIONFEE_OUT, dtype: float64
    

There are only slight difference in tuition fee for in-state and out-of-state college.
Furthermore, the histogram shows that there are some outliers for both of the variable. We will take a deeper look into the variable before removing the outlier. <br><br>


```python
variable_to_check = "MD_EARN_WNE_P6"
dfplt = cleanVariable(variable_to_check)
plt.hist(dfplt[variable_to_check], bins=10)
plt.xlabel("Median student earning of each college")
plt.ylabel("Freq")
plt.title("Distribution of median student earning of each college")
plt.show()

print("Variance: ", np.var(min_max_scaler.fit_transform(dfplt[variable_to_check]))) 
print(dfplt[variable_to_check].describe())
```


![png](output_19_0.png)


    Variance:  0.00743082832795
    count     28281.000000
    mean      28958.236979
    std       10913.398108
    min        7000.000000
    25%       21800.000000
    50%       27300.000000
    75%       34300.000000
    max      133600.000000
    Name: MD_EARN_WNE_P6, dtype: float64
    

We can say that most students' salary after they graduate falls around 30k-40k. However, there are obvious outliers shown where the maximum median earning of that college's graduate is $127300. We will look into it to decide whether it is an error or not. 


```python
dff = df_copy.copy()
dff = dff[dff["MD_EARN_WNE_P6"] > 0]
dff["INSTNM"] = dff["INSTNM"].str.upper()
dff.drop_duplicates("INSTNM", inplace=True)
a = dff.sort_values('MD_EARN_WNE_P6', ascending=False).head(10)
sns.barplot(a["MD_EARN_WNE_P6"], a["INSTNM"], orient="h")
plt.ylabel("Institute name")
plt.xlabel("Median earning")
plt.title("Top 10 college by Median earning")
plt.show()
```


![png](output_21_0.png)


The outlier isn't too significant, when compared side by side with the second highest using a barplot.
This barplot also tells us health sciences or pharmacy related may be one of the highest paying jobs in the US as the top 5 colleges are all related to health and pharmacy. Further analysis is required to prove the hypothesis. <br><br>


```python
variable_to_check = "ADM_RATE"
dfplt = cleanVariable(variable_to_check)
plt.hist(dfplt[variable_to_check])
plt.xlabel("Admission rate of each college")
plt.ylabel("Freq")
plt.title("Distribution of admission of each college")
plt.show()
print("Rate of NaN: ", (len(df_copy)-len(dfplt))/len(df_copy)*100, "%")
print("Value count:")
print(dfplt[variable_to_check].value_counts())
print()
print(dfplt[variable_to_check].describe())
print("Variance: ", np.var(min_max_scaler.fit_transform(dfplt[variable_to_check]))) 
```


![png](output_23_0.png)


    Rate of NaN:  97.10422697856438 %
    Value count:
    1.0     3609
    10.0       1
    3.0        1
    Name: ADM_RATE, dtype: int64
    
    count    3611.000000
    mean        1.003046
    std         0.153416
    min         1.000000
    25%         1.000000
    50%         1.000000
    75%         1.000000
    max        10.000000
    Name: ADM_RATE, dtype: float64
    Variance:  0.000290492669096
    

Over 98% of data from ADM_RATE are NaN values, and the variance is also relatively low which means majority of the data are the exact same. Hence we decided to ignore this feature, as it will not be useful in our analysis. <br><br>


```python
variable_to_check = "ACTENMID"
dfplt = cleanVariable(variable_to_check)
plt.hist(dfplt[variable_to_check], bins=4)
plt.xlabel("ENGLISH ACT SCORE")
plt.ylabel("Freq")
plt.title("Average English ACT score of students enrolled in different institutes")
plt.show()

print("Variance: ", np.var(min_max_scaler.fit_transform(dfplt[variable_to_check]))) 
print(dfplt[variable_to_check].describe())
```


![png](output_25_0.png)


    Variance:  0.0108890489553
    count    13222.000000
    mean        22.316669
    std          3.548055
    min          2.000000
    25%         20.000000
    50%         22.000000
    75%         24.000000
    max         36.000000
    Name: ACTENMID, dtype: float64
    

Each ACT test is scored out of 36 points, with the lowest possible score of 1. Hence there is nothing wrong with this feature. 
Majority of students have an average English ACT score of 22 points.


```python
variable_to_check = "ACTMTMID"
dfplt = cleanVariable(variable_to_check)
plt.hist(dfplt[variable_to_check], bins=3)
plt.xlabel("MATH ACT SCORE")
plt.ylabel("Freq")
plt.title("Average Math ACT score of students enrolled in different institutes")
plt.show()

print("Variance: ", np.var(min_max_scaler.fit_transform(dfplt[variable_to_check]))) 
print(dfplt[variable_to_check].describe())
```


![png](output_27_0.png)


    Variance:  0.00994217229236
    count    13207.000000
    mean        21.994018
    std          3.290569
    min          2.000000
    25%         20.000000
    50%         22.000000
    75%         24.000000
    max         35.000000
    Name: ACTMTMID, dtype: float64
    

Each ACT test is scored out of 36 points, with the lowest possible score of 1.
Through the plot and looking into the data, there are one occurence where the ACT score for Math is only 1.
We decided that this is not an outlier, lowest possible score for an ACT test is 1.


```python
variable_to_check = "ACTWRMID"
dfplt = cleanVariable(variable_to_check)
plt.hist(dfplt[variable_to_check], bins=3)
plt.xlabel("WRITING ACT SCORE")
plt.ylabel("Freq")
plt.title("Average writing ACT score of students enrolled in different institutes")
plt.show()

print("Variance: ", np.var(min_max_scaler.fit_transform(dfplt[variable_to_check]))) 
print(dfplt[variable_to_check].describe())
```


![png](output_29_0.png)


    Variance:  0.000807488465614
    count    1407.000000
    mean       10.491116
    std        14.952310
    min         4.000000
    25%         7.000000
    50%         8.000000
    75%         9.000000
    max       530.000000
    Name: ACTWRMID, dtype: float64
    

Each ACT test is scored out of 36 points, with the lowest possible score of 1. There is an extreme outlier in this feature where the score is 530 where the minimum possible score is only 36. The row associated with it will be removed.


```python
#Removing the outlier
df_copy = df_copy[df_copy["ACTWRMID"] != 530]
```

After cleaning the ACT data, we want to see the top 10 institute with student of highest ACT score.


```python
dff = df_copy.copy()
dff = dff[dff["ACTENMID"] > 0]
dff["INSTNM"] = dff["INSTNM"].str.upper()
dff.drop_duplicates("INSTNM", inplace=True)
a = dff.sort_values('ACTENMID', ascending=False).head(10)
sns.barplot(a["ACTENMID"], a["INSTNM"], orient="h")
plt.ylabel("Institute name")
plt.xlabel("ENGLISH ACT SCORE")
plt.title("Top 10 college by English ACT score")
plt.show()
```


![png](output_33_0.png)



```python
dff = df_copy.copy()
dff = dff[dff["ACTMTMID"] > 0]
dff["INSTNM"] = dff["INSTNM"].str.upper()
dff.drop_duplicates("INSTNM", inplace=True)
a = dff.sort_values('ACTMTMID', ascending=False).head(10)
sns.barplot(a["ACTMTMID"], a["INSTNM"], orient="h")
plt.ylabel("Institute name")
plt.xlabel("MATH ACT SCORE")
plt.title("Top 10 college by Math ACT score")
plt.show()
```


![png](output_34_0.png)



```python
dff = df_copy.copy()
dff = dff[dff["ACTWRMID"] > 0]
dff["INSTNM"] = dff["INSTNM"].str.upper()
dff.drop_duplicates("INSTNM", inplace=True)
a = dff.sort_values('ACTWRMID', ascending=False).head(10)
sns.barplot(a["ACTWRMID"], a["INSTNM"], orient="h")
plt.ylabel("Institute name")
plt.xlabel("WRITING ACT SCORE")
plt.title("Top 10 college by Writing ACT score")
plt.show()
```


![png](output_35_0.png)


From these three barplots of the top 10 institute with highest ACT scores in writing, math, and English, we can say we are not surprised that institutes like MIT and harvard made it on the charts in Math and English ACT scores. <br><br>


```python
variable_to_check = "CONTROL"
dfplt = df_copy.copy()

sns.barplot( dfplt[variable_to_check].value_counts().index, dfplt[variable_to_check].value_counts() )
plt.ylabel("Count")
plt.title("Distribution of type of college")
plt.show()
print(dfplt[variable_to_check].value_counts())
```


![png](output_37_0.png)


    Private for-profit    51531
    Public                37943
    Private nonprofit     35201
    Name: CONTROL, dtype: int64
    

From the distribution fo type of college, we can tell that most colleges in the US are private-nonprofit. While the minority are private for-profit. <br><br>


```python
asd = df_copy.copy()
asd = asd[['INSTNM','STABBR']]
asdd = asd.groupby('STABBR').count().sort_values('INSTNM').head(10)

sns.barplot(asdd["INSTNM"] , asdd.index ,orient="h")
plt.ylabel("State Code")
plt.xlabel("Sum of colleges in state")
plt.title("Top 10 state which has most colleges")
plt.show()
```


![png](output_39_0.png)


From the graph, we know that which state has the highest number of colleges. We can conclude that Delaware has the highest number of college by refering to the graph. This enable us to know if we want to study, probably which state should we targeting at. <hr>

## Data Mining

After exploring the main characteristics of our data, we attempt to find patterns and relationships between features.


```python
inout = df_copy.copy()
inout = inout[(inout.TUITIONFEE_IN > 0) & (inout.TUITIONFEE_OUT > 0)]
inout = inout[['TUITIONFEE_IN','TUITIONFEE_OUT','CONTROL']]
sns.regplot(inout.TUITIONFEE_IN,inout.TUITIONFEE_OUT,scatter=True,fit_reg=True)
plt.xlabel("Tuition fee in-state")
plt.ylabel("Tuition fee out-state")
plt.title("Comparison between the tuition fees in-state and out-state")
plt.show()
```


![png](output_42_0.png)


It looks like there are two set of data point, one where data points lie perfectly on line of x=y, which mean that both tution fee are the same for these institution. From this graph, we can conclude that some university might have a higher tuition fee for out-state students.


```python
sns.lmplot(x="TUITIONFEE_IN",y="TUITIONFEE_OUT",col="CONTROL",data = inout,truncate=True)
plt.show()
```


![png](output_44_0.png)


From the graph above, we found out that around 30% of Private For-Profit and 55% of private non-profit share the same fee cost for both in and outstate study while only 5% of pulic institution share the same fee.
Plot shows how many point lie on the line



```python
inout.groupby('CONTROL').mean()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TUITIONFEE_IN</th>
      <th>TUITIONFEE_OUT</th>
    </tr>
    <tr>
      <th>CONTROL</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Private for-profit</th>
      <td>12866.961431</td>
      <td>12870.163045</td>
    </tr>
    <tr>
      <th>Private nonprofit</th>
      <td>17553.688145</td>
      <td>17580.178636</td>
    </tr>
    <tr>
      <th>Public</th>
      <td>3544.328542</td>
      <td>8433.536785</td>
    </tr>
  </tbody>
</table>
</div>



After analysis for all of the institution, we further our analysis to calculate the mean for both in and outstate tution fee cost.There is almost no in-state out-state fee difference for private instituions while for public institutions the out state fee is approximate twice compare with instate fee. <br><br>


```python
asd = df_copy.copy()
asd = asd[asd['UNEMP_RATE']>0]
asd = asd[['INSTNM','STABBR','UNEMP_RATE']]

asd["INSTNM"] = asd["INSTNM"].str.upper()
asd = asd.sort_values('UNEMP_RATE',ascending=False)
asd.drop_duplicates("INSTNM", inplace=True)
asd.head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>INSTNM</th>
      <th>STABBR</th>
      <th>UNEMP_RATE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12221</th>
      <td>STONE CHILD COLLEGE</td>
      <td>MT</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>6874</th>
      <td>ALASKA VOCATIONAL TECHNICAL CENTER</td>
      <td>AK</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>35414</th>
      <td>BLACKFEET COMMUNITY COLLEGE</td>
      <td>MT</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>12704</th>
      <td>UNITED EDUCATION AND COMPUTER COLLEGE</td>
      <td>CA</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>24102</th>
      <td>OGLALA LAKOTA COLLEGE</td>
      <td>SD</td>
      <td>12.0</td>
    </tr>
  </tbody>
</table>
</div>



From the table, we are able to find out the institute with the top 5 highest unemployment rate, these are probably the college that we gonna avoid as we would want a low unemployment rate college.

## Which sector of ACT scores contributes the most to the income of graduate?
_We will go deeper into it in our linear regression model_


```python
dfplt = df_copy.copy()
dfplt = dfplt[(dfplt.ACTENMID > 0)  & (dfplt.ACTMTMID > 0) & (dfplt.ACTWRMID > 0) & (dfplt['MD_EARN_WNE_P6']>0)]
math = plt.plot(dfplt['ACTMTMID'], dfplt.MD_EARN_WNE_P6, 'ro', label = "math")
writing = plt.plot(dfplt['ACTWRMID'], dfplt.MD_EARN_WNE_P6, 'go', label = "writing")
english = plt.plot(dfplt['ACTENMID'], dfplt.MD_EARN_WNE_P6, 'bo', label = "english")
plt.title("Relationship between ACT and Earning")
plt.xlabel("ACT")
plt.ylabel("Earning")
plt.legend()
plt.show()
```


![png](output_51_0.png)


From the graph above, we can know that ACT scores might have a significant affect in determine the future earning. 
Hence for now, maybe it is suitable in training our predictive model.

## Does SAT score affect the income of graduate?


```python
Sat_6yEarn = df_copy[(df_copy.SAT_AVG > 0) & (df_copy.MD_EARN_WNE_P6 > 0) ]
plt.scatter(Sat_6yEarn['SAT_AVG'], Sat_6yEarn['MD_EARN_WNE_P6'])
plt.xlabel('SAT_AVG')
plt.ylabel('MD_EARN_WNE_P6')
plt.title("Relationship  between Average SAT Score and Earning")
plt.xlabel("Average SAT Score")
plt.ylabel("Earning")
plt.show()

#clearly an increasing sign
```


![png](output_54_0.png)


From the graph above, we can know that the SAT score might affect the earning, but the plot is not that significant and contains some outliers.

## Does tuition fee affect the income  of graduate?


```python
#find if Tuition fee affect Earning
Tui_6yEarn = df_copy[(df_copy.TUITIONFEE_PROG > 0) & (df_copy.MD_EARN_WNE_P6 > 0) ]
plt.scatter(Tui_6yEarn['TUITIONFEE_PROG'], Tui_6yEarn['MD_EARN_WNE_P6'])
plt.xlabel('Tuition Fee')
plt.ylabel('Earning')
plt.title("Relationship  between Tuition Fee and Earning")
plt.show()
#not quite, and hard to see as most of the tuition fee fall in the lower range
```


![png](output_57_0.png)


From the graph above, we can know that tuition fee might not really affect the earning of graduates
This is probably due to most college has the same range of tuition fees like the descriptive analysis that we found out ealier.


```python
ACT3_6yEarn = df_copy[(df_copy.ACTENMID > 0)  & (df_copy.ACTMTMID > 0) & (df_copy.ACTWRMID > 0) & (df_copy.MD_EARN_WNE_P6 > 0) ]
acs = ACT3_6yEarn.sort_values(['ACTENMID', 'ACTMTMID', 'ACTWRMID'], ascending = False).head(10)
des = ACT3_6yEarn.sort_values(['ACTENMID', 'ACTMTMID', 'ACTWRMID'], ascending = True).head(10)
top = plt.plot(acs['MD_EARN_WNE_P6'],'ro', label="Top10")
btn = plt.plot(des['MD_EARN_WNE_P6'],'bo', label="Worst10")  

plt.title("Top10 ACT score compare with Worst10 ACT score(Earning in 6 years)")
plt.legend()
plt.ylabel('Earning')
plt.show()
#red mean the top 10 ACT score in read, writing and english
#blue mean the least 10 ACT score in read, writing and english
#clearly we can know that red earn high than blue
```


![png](output_59_0.png)


ACT scores might affect the earnings of graduates. We try to plot the comparison between the top 10 ACT score college and the worst 10 ACT score college in term of earning.

## Does studying in public college earns more than studying in private college?


```python
public = df_copy[(df_copy.CONTROL == 'Public')]
private_np = df_copy[(df_copy.CONTROL == 'Private nonprofit')]
private_p = df_copy[(df_copy.CONTROL == 'Private for-profit')]
public = public[public.MD_EARN_WNE_P6 > 0]
private_np = private_np[private_np.MD_EARN_WNE_P6 > 0]
private_p = private_p[private_p.MD_EARN_WNE_P6 > 0]
x = sum(public['MD_EARN_WNE_P6'])/len(public)
y = sum(private_np['MD_EARN_WNE_P6'])/len(private_np)
z = sum(private_p['MD_EARN_WNE_P6'])/len(private_p)
dictionary = plt.figure()

D = {u'Public':x, u'Private for non profit': y, u'Private for profit':z}

plt.title("Comparison between Public and Private College (Earning)")
plt.bar(range(len(D)), D.values(), align='center')
plt.xticks(range(len(D)), D.keys())
plt.ylabel('Earning')
plt.show()
```


![png](output_62_0.png)


The graph gives us a basic idea on which type of college might have a higher earning after graduate. From the graph above, we can conclude that enroll in the private for non profit college will most probably earn a higher income. <hr>

# Linear Regression

We try to build a linear regression model based on the ACT score to predict the future income level. The reason we choose ACT score is due to that we found out ACT might be showing significant effect on determine the income of the graduates.


```python
linear_data = df_copy.copy()
linear_data = linear_data[(linear_data.ACTENMID > 0)  & (linear_data.ACTMTMID > 0) & (linear_data.ACTWRMID > 0) & (linear_data['MD_EARN_WNE_P6']>0)]
linear_data = linear_data[['ACTENMID', 'ACTMTMID', 'ACTWRMID','MD_EARN_WNE_P6']]

from sklearn.linear_model import LinearRegression
x = linear_data.drop('MD_EARN_WNE_P6' , axis = 1)
lm = LinearRegression()
lm.fit(x,linear_data.MD_EARN_WNE_P6)

print ('Estimated intercept', len(lm.coef_))
a = pd.DataFrame(list(zip(x.columns, lm.coef_)), columns = ['Subject', 'coeficient'])
print(a)
#since ACT math is the highest coeficient, hence we plot ACT math

plt.scatter(linear_data.ACTMTMID, linear_data.MD_EARN_WNE_P6)
plt.xlabel("Math ACT score")
plt.ylabel("Earning in 6 years")
plt.title("Relationship between Math ACT score and Earning in 6 years (actual)")
plt.show()
```

    Estimated intercept 3
        Subject   coeficient
    0  ACTENMID -1189.144312
    1  ACTMTMID  2910.042421
    2  ACTWRMID    -3.014084
    


![png](output_65_1.png)


From the a dataframe above, we can conclude that math might have a highest coeficient, hence we want to look at only Math ACT score instead of all ACT score like the one we did above. Since math ACT score has the highest coeficient, we might train our model based on Math ACT score or combining the three ACT score to train our model. We will decide based on their performance later on.


```python
lm.predict(x)[0:328]
plt.scatter(linear_data.ACTMTMID, lm.predict(x))
plt.xlabel("Math ACT score")
plt.ylabel("Earning in 6 years")
plt.title("Relationship between Math ACT score and Earning in 6 years (predict)")
plt.show()
```


![png](output_67_0.png)


We try to see the relationship between Math ACT score and earning after applying the predictive model to see if it has a big difference with the actual one or not.


```python
y = linear_data.drop(['MD_EARN_WNE_P6'] , axis = 1)
lm = LinearRegression()
lm.fit(y[['ACTMTMID']],linear_data.MD_EARN_WNE_P6)

msemath = np.mean((linear_data.MD_EARN_WNE_P6 - lm.predict(y[['ACTMTMID']])) ** 2)
print("Math mean square error = ", msemath)

lm.fit(y,linear_data.MD_EARN_WNE_P6)
msefull = np.mean((linear_data.MD_EARN_WNE_P6 - lm.predict(y)) ** 2)
print("Full mean square error = ", msefull)
```

    Math mean square error =  42463329.15538637
    Full mean square error =  40171339.622049525
    

input 3 values perform better, hence we will try to build our model by considering both 3 ACT values


```python
import sklearn.cross_validation
x_train, x_test, y_train, y_test = sklearn.cross_validation.train_test_split(x, linear_data.MD_EARN_WNE_P6, test_size = 0.33, random_state = 5)
print(x_train.shape)
print(x_test.shape) 
print(y_train.shape)
print(y_test.shape)
```

    (219, 3)
    (109, 3)
    (219,)
    (109,)
    

We try to split the data into training and test set. Then, we find the shape of the training and testing to make sure our training and testing dataset is ready.


```python
lm = LinearRegression()
lm.fit(x_train, y_train)
pred_train = lm.predict(x_train)
pred_test = lm.predict(x_test)
```

We try to train our model based on all the ACT scores as we found out that by involving all the ACT scores, we can get a lower mean square error.


```python
print("Train mean square error:", np.mean((y_train - lm.predict(x_train)) ** 2))
print("Test mean square error:", np.mean((y_test - lm.predict(x_test)) ** 2))
```

    Train mean square error: 39191974.760901645
    Test mean square error: 42488041.97791104
    

We try to see the mean square error for both train and test set.


```python
plt.scatter(lm.predict(x_train),lm.predict(x_train) - y_train, c='b', s=40, alpha=0.5)
plt.scatter(lm.predict(x_test),lm.predict(x_test) - y_test, c='g', s=40)
plt.hlines(y = 0, xmin=0, xmax = 70000)
plt.title('Error Rate for Training and Testing Set')
plt.ylabel('Error')
plt.legend()
plt.show()

#most of it scatter around 0, good!
```


![png](output_77_0.png)


We try to plot our the error rate for training and testing set. According to the graph, when the plot is nearer to 0, it mean that it contain a smaller range of error. Hence the graph shown that all the plot is around the 0 line, which is a very good sign.


```python
from sklearn import linear_model
import numpy as np
lm = linear_model.LinearRegression()
lm.fit(x_train, y_train)
pred_test = lm.predict(x_test)
fit = np.polyfit(x['ACTMTMID'],linear_data.MD_EARN_WNE_P6,1)
p = np.poly1d(fit)

#since we know math has highest corelation, try to fit and compare using it
math = plt.plot(x['ACTMTMID'], linear_data.MD_EARN_WNE_P6, 'ro', label = "math")
xp = np.linspace(15, 35, 50)
best_fit = plt.plot(xp, p(xp), '-', color='green', label="best fit line")

plt.title("Relationship between ACT[Math] and Earning(actual)")
plt.xlabel("ACT[Math]")
plt.ylabel("Earning")
plt.legend()
plt.show()
```


![png](output_79_0.png)


Since we know that math has the highest coeficient, we try to plot out the best fit for math to just have an idea of how our linear regression work so that we will have confidence in our model. The graph above shows the original dataset and its best fit line, we will then use this to compare with our predict model to see if it has any significant difference.


```python
fit = np.polyfit(x_test['ACTMTMID'],lm.predict(x_test),1)
p = np.poly1d(fit)

plt.plot(x_test['ACTMTMID'], lm.predict(x_test), 'ro')
xp = np.linspace(15, 35, 50)
orange = plt.plot(xp, p(xp), '-')

plt.title("Relationship between ACT[Math] and Earning(model)")
plt.xlabel("ACT[MATH]")
plt.ylabel("Earning")
plt.show()
```


![png](output_81_0.png)


The graph above shows the best fit line of our model when applied to test set, the difference is not that significant so it prove that our linear regression is logical through comparing the both graph visualization.


```python
print("Mean square error for the model is:", np.mean((y_test - lm.predict(x_test)) ** 2)) 
print("The estimated coeficient of English is: ", lm.coef_[0])
print("The estimated coeficient of Math is: ", lm.coef_[1])
print("The estimated coeficient of Writing is: ", lm.coef_[2])
```

    Mean square error for the model is: 42488041.97791104
    The estimated coeficient of English is:  -1076.09957108
    The estimated coeficient of Math is:  2811.35207771
    The estimated coeficient of Writing is:  223.65933888
    

We want to once find out the mean square error for the model when applied to test set and the coeficient of all our attributes that are used to train our dataset.


```python
print(lm.predict([30,30,11]))
```

    [ 47890.72193148]
    

We try to predict an income based on the ACT scores, the result is logical and it prove that our model should be working in guessing the future income.

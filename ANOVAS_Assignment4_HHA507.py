# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 23:05:54 2021

@author: manle
"""

""" Loading the dataset"""

import pandas as pd

MathAlcstudents = pd.read_csv("https://raw.githubusercontent.com/manleen1210/ANOVAS_HHA507/main/MathStudents_AlcoholConsumption.csv")


""" Identifying variables to test in the ANOVAs """

list(MathAlcstudents)
['school',
 'sex',
 'age',
 'address',
 'famsize',
 'Pstatus',
 'Medu',
 'Fedu',
 'Mjob',
 'Fjob',
 'reason',
 'guardian',
 'traveltime',
 'studytime',
 'failures',
 'schoolsup',
 'famsup',
 'paid',
 'activities',
 'nursery',
 'higher',
 'internet',
 'romantic',
 'famrel',
 'freetime',
 'goout',
 'Dalc',
 'Walc',
 'health',
 'absences',
 'G1',
 'G2',
 'G3']


""" Variables to test
Dependent variable: G3 (Final Grade)
Independent variable 1: Walc (Weekend alcohol consumption) (Scale of 1 (very low) to 5 (very high))
Independent variable 2: Dalc (Weekday alcohol consumption) (Scale of 1 (very low) to 5 (very high))
Independent variable 3: Famrel (Quality of relationship with family) (Scale of 1 (very bad) to 5 (excellent)) """

"""Renaming variables to make them more understandable"""

MathAlcstudents = MathAlcstudents.rename(columns={ 'G3' : 'Final_grade'})
MathAlcstudents = MathAlcstudents.rename(columns={ 'Walc' : 'Weekend_alc'})
MathAlcstudents = MathAlcstudents.rename(columns={ 'Dalc' : 'Weekday_alc'})
MathAlcstudents = MathAlcstudents.rename(columns={ 'famrel' : 'Family_relationship_quality'})

""" Replacing numeric values with string """
MathAlcstudents['Weekend_alc'].replace({1: 'Very_low', 2: 'low', 3: 'Neutral', 4: 'High', 5: 'Very_high'}, inplace= True)
MathAlcstudents['Weekday_alc'].replace({1: 'Very_low', 2: 'low', 3: 'Neutral', 4: 'High', 5: 'Very_high'}, inplace= True)
MathAlcstudents['Family_relationship_quality'].replace({1: 'Very_bad', 2: 'bad', 3: 'Neutral', 4: 'good', 5: 'Excellent'}, inplace= True)

"""Dataset with relevant columns only"""
MathAlcstudents_relevant = MathAlcstudents[['Final_grade', 'Weekend_alc','Weekday_alc','Family_relationship_quality']]


"""Importing all relevant packages"""
import scipy.stats as stats
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.multicomp as mc
import statsmodels.api as sm

"""First ANOVA
1 way anova
1 DV - Final_grade
1 IV - Weekend_alc
Is there a difference between the "levels" of Weekend alcohol consumption and 
the final grade of the student?"""

""" Checking skewness and kurtosis """
from scipy.stats import kurtosis, skew, bartlett

Weekend_alc1 = MathAlcstudents_relevant[MathAlcstudents_relevant['Weekend_alc'] == 'Very_low']
Weekend_alc2 = MathAlcstudents_relevant[MathAlcstudents_relevant['Weekend_alc'] == 'low']
Weekend_alc3 = MathAlcstudents_relevant[MathAlcstudents_relevant['Weekend_alc'] == 'Neutral']
Weekend_alc4 = MathAlcstudents_relevant[MathAlcstudents_relevant['Weekend_alc'] == 'High']
Weekend_alc5 = MathAlcstudents_relevant[MathAlcstudents_relevant['Weekend_alc'] == 'Very_high']

Weekendskew1 = skew(Weekend_alc1['Final_grade'])
Weekendskew2 = skew(Weekend_alc2['Final_grade'])
Weekendskew3 = skew(Weekend_alc3['Final_grade'])
Weekendskew4 = skew(Weekend_alc4['Final_grade'])
Weekendskew5 = skew(Weekend_alc5['Final_grade'])

""" The negative skew across all levels of weekend alcohol consumption levels 
indicates date has outliers on the left side of the curve """

Weekendkurt1 = kurtosis(Weekend_alc1['Final_grade'])
Weekendkurt2 = kurtosis(Weekend_alc2['Final_grade'])
Weekendkurt3 = kurtosis(Weekend_alc3['Final_grade'])
Weekendkurt4 = kurtosis(Weekend_alc4['Final_grade'])
Weekendkurt5 = kurtosis(Weekend_alc5['Final_grade'])

""" Levels 1 and 2 of weekend alcohol consumption indicates a lack of outliers 
whereas levels 3,4 and 5 indicate a high number of outliers (leptokurtic) """

""" Creating a boxplot """

Grade_Weekendalc_boxplot = sns.boxplot(x='Weekend_alc', y= 'Final_grade', data=MathAlcstudents_relevant)

Weekendalc_counts = MathAlcstudents_relevant['Weekend_alc'].value_counts().reset_index()
""" The values are all different for the 5 categories and so it is unbalanced """

""" Shapiro test 
    DV ~ C(IV) """
model1 = smf.ols("Final_grade ~ C(Weekend_alc)", data = MathAlcstudents_relevant).fit()

stats.shapiro(model1.resid)
""" (statistic=0.9336082935333252, p-value=2.92812497339201e-12) """

anova_table1 = sm.stats.anova_lm(model1, typ=1)
anova_table1
                   df       sum_sq    mean_sq         F   PR(>F)
C(Weekend_alc)    4.0    61.722394  15.430599  0.733162  0.56975
Residual        390.0  8208.186467  21.046632       NaN      NaN

""" One-way ANOVA """

stats.f_oneway(MathAlcstudents_relevant['Final_grade'][MathAlcstudents_relevant['Weekend_alc'] == 'Very_low'],
               MathAlcstudents_relevant['Final_grade'][MathAlcstudents_relevant['Weekend_alc'] == 'low'],
               MathAlcstudents_relevant['Final_grade'][MathAlcstudents_relevant['Weekend_alc'] == 'Neutral'],
               MathAlcstudents_relevant['Final_grade'][MathAlcstudents_relevant['Weekend_alc'] == 'High'],
               MathAlcstudents_relevant['Final_grade'][MathAlcstudents_relevant['Weekend_alc'] == 'Very_high'])
""""F_onewayResult(statistic=0.7331623695637165, pvalue=0.5697502070573542)"""
""" The p-value is greater than 0.05 which indicates that there is no 
signficant difference between the levels of alcohol consumption on the 
weekend and the final grade"""

""" Post-hoc test """
comp1 = mc.MultiComparison(MathAlcstudents_relevant['Final_grade'], MathAlcstudents_relevant['Weekend_alc'])
post_hoc_res1 = comp1.tukeyhsd() 
tukeyway1 = post_hoc_res.summary()

"""Second ANOVA
1 way anova
1 DV - Final_grade
1 IV - Weekday_alc
Is there a difference between the "levels" of Weekday alcohol consumption and 
the final grade of the student?"""


""" Checking skewness and kurtosis """

Weekday_alc1 = MathAlcstudents_relevant[MathAlcstudents_relevant['Weekday_alc'] == 'Very_low']
Weekday_alc2 = MathAlcstudents_relevant[MathAlcstudents_relevant['Weekday_alc'] == 'low']
Weekday_alc3 = MathAlcstudents_relevant[MathAlcstudents_relevant['Weekday_alc'] == 'Neutral']
Weekday_alc4 = MathAlcstudents_relevant[MathAlcstudents_relevant['Weekday_alc'] == 'High']
Weekday_alc5 = MathAlcstudents_relevant[MathAlcstudents_relevant['Weekday_alc'] == 'Very_high']

Weekdayskew1 = skew(Weekday_alc1['Final_grade'])
Weekdayskew2 = skew(Weekday_alc2['Final_grade']) 
Weekdayskew3 = skew(Weekday_alc3['Final_grade'])
Weekdayskew4 = skew(Weekday_alc4['Final_grade'])
Weekdayskew5 = skew(Weekday_alc5['Final_grade'])

""" The negative skew across all levels of weekday alcohol consumption levels 
indicates date has outliers on the left side of the curve """

Weekdaykurt1 = kurtosis(Weekday_alc1['Final_grade'])
Weekdaykurt2 = kurtosis(Weekday_alc2['Final_grade'])
Weekdaykurt3 = kurtosis(Weekday_alc3['Final_grade'])
Weekdaykurt4 = kurtosis(Weekday_alc4['Final_grade'])
Weekdaykurt5 = kurtosis(Weekday_alc5['Final_grade'])

""" Levels 2 and 4 of weekday alcohol consumption indicates a lack of outliers 
whereas levels 1,3 and 5 indicate a high number of outliers (leptokurtic) """

""" Creating a boxplot """
Grade_Weekdayalc_boxplot = sns.boxplot(x='Weekday_alc', y= 'Final_grade', data=MathAlcstudents_relevant)

Weekdayalc_counts = MathAlcstudents_relevant['Weekday_alc'].value_counts().reset_index()
""" The values are all different for the 5 categories and so it is unbalanced """

""" Shapiro Test
    DV ~ C(IV) """
model2 = smf.ols("Final_grade ~ C(Weekday_alc)", data = MathAlcstudents_relevant).fit()

stats.shapiro(model2.resid)
""" (statistic=0.9369356632232666, p-value=6.8601300855231084e-12) """

anova_table2 = sm.stats.anova_lm(model2, typ=1)
anova_table2
                df       sum_sq    mean_sq         F    PR(>F)
C(Weekday_alc)    4.0   132.173885  33.043471  1.583605  0.177864
Residual        390.0  8137.734976  20.865987       NaN       NaN

""" One way ANOVA """

stats.f_oneway(MathAlcstudents_relevant['Final_grade'][MathAlcstudents_relevant['Weekday_alc'] == 'Very_low'],
               MathAlcstudents_relevant['Final_grade'][MathAlcstudents_relevant['Weekday_alc'] == 'low'],
               MathAlcstudents_relevant['Final_grade'][MathAlcstudents_relevant['Weekday_alc'] == 'Neutral'],
               MathAlcstudents_relevant['Final_grade'][MathAlcstudents_relevant['Weekday_alc'] == 'High'],
               MathAlcstudents_relevant['Final_grade'][MathAlcstudents_relevant['Weekday_alc'] == 'Very_high'])
"""F_onewayResult(statistic=1.5836045063367645, pvalue=0.17786362227119418)"""
""" The p-value is greater than 0.05 which indicates that there is no 
signficant difference between the levels of alcohol consumption on the 
weekdays and the final grade"""

""" Post-hoc test """
comp2 = mc.MultiComparison(MathAlcstudents_relevant['Final_grade'], MathAlcstudents_relevant['Weekday_alc'])
post_hoc_res2 = comp2.tukeyhsd() 
tukeyway2 = post_hoc_res.summary()


"""Third ANOVA
1 way anova
1 DV - Final_grade
1 IV - Family_relationship_quality
Is there a difference between the "levels" of the quality of relationship 
between the student and their family and the final grade of the student?"""

""" Checking skewness and kurtosis """

Relationshipquality1 = MathAlcstudents_relevant[MathAlcstudents_relevant['Family_relationship_quality'] == 'Very_bad']
Relationshipquality2 = MathAlcstudents_relevant[MathAlcstudents_relevant['Family_relationship_quality'] == 'bad']
Relationshipquality3 = MathAlcstudents_relevant[MathAlcstudents_relevant['Family_relationship_quality'] == 'Neutral']
Relationshipquality4 = MathAlcstudents_relevant[MathAlcstudents_relevant['Family_relationship_quality'] == 'good']
Relationshipquality5 = MathAlcstudents_relevant[MathAlcstudents_relevant['Family_relationship_quality'] == 'Excellent']

Relationshipquality_skew1 = skew(Relationshipquality1['Final_grade'])
Relationshipquality_skew2 = skew(Relationshipquality2['Final_grade']) 
Relationshipquality_skew3 = skew(Relationshipquality3['Final_grade'])
Relationshipquality_skew4 = skew(Relationshipquality4['Final_grade'])
Relationshipquality_skew5 = skew(Relationshipquality5['Final_grade'])

""" The negative skew across all levels of weekday alcohol consumption levels 
indicates date has outliers on the left side of the curve """

Relationshipquality_kurt1 = kurtosis(Relationshipquality1['Final_grade'])
Relationshipquality_kurt2 = kurtosis(Relationshipquality2['Final_grade'])
Relationshipquality_kurt3 = kurtosis(Relationshipquality3['Final_grade'])
Relationshipquality_kurt4 = kurtosis(Relationshipquality4['Final_grade'])
Relationshipquality_kurt5 = kurtosis(Relationshipquality5['Final_grade'])

""" Level 2 of the quality of relationship between students and their family
indicates a lack of outliers whereas levels 1,3, 4 and 5 indicate a 
high number of outliers (leptokurtic) """

""" Creating a boxplot """

Grade_FamilyRelationship_boxplot = sns.boxplot(x='Family_relationship_quality', y= 'Final_grade', data=MathAlcstudents_relevant)

Familyrelationship_counts = MathAlcstudents_relevant['Family_relationship_quality'].value_counts().reset_index()
""" The values are all different for the 5 categories and so it is unbalanced """

""" Shapiro Test 
    DV ~ C(IV) """
model3 = smf.ols("Final_grade ~ C(Family_relationship_quality)", data = MathAlcstudents_relevant).fit()

stats.shapiro(model3.resid)
""" (statistic=0.9366806745529175, p-value=6.4201751553971675e-12) """

anova_table3 = sm.stats.anova_lm(model2, typ=1)
anova_table3
               df       sum_sq    mean_sq         F    PR(>F)
C(Weekday_alc)    4.0   132.173885  33.043471  1.583605  0.177864
Residual        390.0  8137.734976  20.865987       NaN       NaN

""" One way ANOVA """

stats.f_oneway(MathAlcstudents_relevant['Final_grade'][MathAlcstudents_relevant['Family_relationship_quality'] == 'Very_bad'],
               MathAlcstudents_relevant['Final_grade'][MathAlcstudents_relevant['Family_relationship_quality'] == 'bad'],
               MathAlcstudents_relevant['Final_grade'][MathAlcstudents_relevant['Family_relationship_quality'] == 'Neutral'],
               MathAlcstudents_relevant['Final_grade'][MathAlcstudents_relevant['Family_relationship_quality'] == 'good'],
               MathAlcstudents_relevant['Final_grade'][MathAlcstudents_relevant['Family_relationship_quality'] == 'Excellent'])
"""F_onewayResult(statistic=0.3974329754209523, pvalue=0.8104874341858186)"""
""" The p-value is greater than 0.05 which indicates that there is no 
signficant difference between the levels of the quality of the relationship
with their family and the students' final grade"""

""" Post-hoc test """
comp3 = mc.MultiComparison(MathAlcstudents_relevant['Final_grade'], MathAlcstudents_relevant['Family_relationship_quality'])
post_hoc_res3 = comp3.tukeyhsd() 
tukeyway3 = post_hoc_res.summary()
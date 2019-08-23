## ANLY502 Mini Project
## Hongyang Zheng

## Introduction
This mini project used the Reddit comments archive dataset for Oct/Nov/Dev 2018 and Jan 2019, which includes 39 variables such as the body of the comment, the score of a comment, the author, whether this comment is collapsed and etc. By exploratory data analysis, I gained a basic understanding of the data features and data structure by tables and plots. Then I made some data transformation and convert categorical variables into features. Finally I built two kinds of classification model to predict whether a comment is collapsed. 


## Code
My code is shown in [Project.ipynb](https://github.com/gu-anly502/spring2019-miniproject-zzzzzzhy0607/blob/master/Project.ipynb)


## Methods
### Preprocessing
The original dataset contains 39 variables. By looking at the type and meaning of these variables, I found that most of them are meaningless boolean type or string type. Since I only want to keep the variables that having meaning for building models and interpretation, I droped lots of meaningless variables like: author_flair_background_color, author_flair_text_color. After this step, I had 19 variables. Next I count the missing value for the remained dataset, if there are too many missing value, using it to build model will influence the accuracy. At this step, I found that there are five variables having too many missing values so I dropped it.

### Summarize data and tables
First, I use `df.describe().show()` to show a brief summary for the data to look at the statistical summary and data features. In this step, I mainly used sample data. Since the purpose of this step is to see the structure or features of the data and using sampling data does not have influence and can be faster. I found that the `score` variable changes widely and is kind of right-skewed, so I made log scale for this variable to prepare for the data visualization.

I used SparkSQL to make tables. This table is score grouped by `collapsed`. We can see that the average score is significantly different by the status of collapsed.
```
+----------+----------+-------------------+
|min(score)|max(score)|         avg(score)|
+----------+----------+-------------------+
|    -22280|     50676|-1.5141080381505283|
|     -2898|     90192|  9.747161969346473|
+----------+----------+-------------------+
```

This table is score grouped by `no_follow`. We can see that the average score is significantly different by the status of no_follow. 
```
+------------------+----------+----------+
|        avg(score)|min(score)|max(score)|
+------------------+----------+----------+
| 2.291694561100037|    -22280|     54430|
|33.055577562149196|         0|     90192|
+------------------+----------+----------+
```
From above tables, we can conclude that `collapsed` and `no_follow` can be important factors to influence `score`.

### Data visualization
In this step, I mainly used sample data. Since the purpose of this step is to see the structure or features of the data and uisng sampling data does not have influence and can be faster. I mainly used boxplot, scatterplot and heat map.

![image](https://github.com/gu-anly502/spring2019-miniproject-zzzzzzhy0607/blob/master/histogram.png)

From this histogram we can see that most log-scaled score are in the interval [0,1]. After making log transformation, the data is still a little right-skewed. 

![image](https://github.com/gu-anly502/spring2019-miniproject-zzzzzzhy0607/blob/master/scatter%20plot.png)

This boxplot describes the relationship between log-scaled score and controversiality. It can be obviously found that when controversiality = 1, the log-scaled score is no more than 5. Therefore, when there exists a controversiality, this comment will receive a lower score.

![image](https://github.com/gu-anly502/spring2019-miniproject-zzzzzzhy0607/blob/master/heatmap.png)

This heatmap represents the correlation between two variables in the dataset. When the color is deeper, the negative relationship is bigger, while when the color is lighter, the positive relationship is bigger. We can see that log-scaled score and no_follow have a very strong negative correlation. There are also many other pairs of variable having correlation such as collapsed and no_follow, score and glided.


### Convert data type and create pipline
In this step, I convert Boolean type variables `no_follow` and ` subreddit_type` into string and then convert them together into features using ` StringIndexer` and ` OneHotEncoder`. I also converted Boolean type ` collapsed` into integer 0 or 1 so we can predict it in the model part. 


### Classification models
Based on previous exploratory step, I am interested in predicting whether a comment will collapse. I first took 1% sample of the whole dataset which should be of size 5GB. I think this is a reasonable size because when I tried the model using the sample data (1 million rows), I have got a pretty good accuracy rate, so adding more observations will lead to a better or equal result. Since 5GB data will definitely have more than 1 million rows, I think this is large enough to ensure the accuracy. I used random sampling to take the sample, which ensures the sample has similar distribution as the population. Then I split the data into training dataset and test dataset: one is used to train models and one is used to check the accuracy of the model. 

For model part, I chose random forest classifier and gradient-boosted tree classifier, which are both based on trees. For random forest, I started at two variables `subreddit_type_s` and `new_no_follow_s`, because no_follow is related with collapsed (from heatmap) and subreddit_type is a variable I am interested in. Then I add one more variable `score` to the model to test whether the performance will be improved, since from previous tables, we know that the average score is very different for different status of collapse. The accuracy rate I got is about 0.95, so at this point, instead of adding another variable to the same model, I want to investigate a different model. My third model used gradient-boosted tree classifier, predicting `collapsed` by `subreddit_type_s` and `new_no_follow_s`. Through this way, I want to compare that holding the same predictors, which tree-based model has higher accuracy rate or is more efficient.


## Results
The first random forest classifier model only used two predictors `subreddit_type_s` and `new_no_follow_s`. I trained the model using training dataset and got the test model accuracy is about 0.94, which is pretty good. Then I tried to add a new variable `score` into the model to see whether the performance will be improved. Therefore, for the second random forest classifier model I used `subreddit_type_s`, `new_no_follow_s` and `score`. It turned out that by adding this new variable, the test model accuracy does improve to 0.95. 

In addition to comparing whether adding `score` will improve the performance, I also investigated whether using same variables but different models will lead to different performance. At this step, I applied gradient-boosted tree classifier and found the test model accuracy is very close to the one using random forest classifier model. Therefore, for this dataset, using random forest classifier model and gradient-boosted tree classifier does not bring significant difference.

Another important point is, from the model summary, we can see that when using random forest classifier model, there are 20 trees, while using gradient-boosted tree classifier there are 10 trees. However, the performance is similar. Therefore, maybe gradient-boosted tree classifier is more efficient.


## Future work
This project mainly focuses on classification models: predict whether a comment will collapse or not. In the future, I may try regression models like random forest regression and gradient-boosted tree regression to predict the numerical score response.


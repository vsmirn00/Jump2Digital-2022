# Jump2Digital-2022

#### Background

On the basis of the Paris Agreement, different nations agreed to tackle down climate change. One of the main goals was to reduce global warming beneath 2 degrees. Hopefully, underneath 1.5 degrees in comparison with preindustrial levels. 

In order to adress this hefty hurdle, each nation is to produce the maximum gas emition possible to reach a neutral climate by the middle of this century. In the same tandem, the European Union is investing unduly amounts of financial resources to research new technologies for the sake of adressing this issue. One of these new technologies, is the laser-based sensor which allows us to detect the air quality.

#### Problem

Based on the available data provided in 2 datasets (train and test), we have 8 different sensors to assess the quality of the air. That being so, our target variable is composed of 3 different labels: 0, 1 and 2. Corresponding thus, to good, moderate and bad air quality respectively. Hence, our goal is to implement a Random Forest Classifier to make the predictions and submit them with a json format. 

#### Results

We performed a superficial exploratory data analysis, focusing mainly on the shape that the numirc features took. This is, performind distplots, boxplots and violinplots. On the grounds of this analysis, one can realise that the data is normally distributed. Furthermore, we compared the 3 correlation matrices (Pearson, Spearman and Kendall), and noticed that except for the features 4, 7 and 8, most of the other features have a high correlation.

On the other hand, we implemented a gridsearch to overhaul our first random forest. By doing so, we found the best hyperparameters to tune our final model. In view of the fact that our model performed well enough, we used the model to make the predictions for the test dataset.

#### Analysis

With the aim of evaluating results, a Confusion matrix and a Classification report were plotted aside from the accuracy value. This final accuracy was of 93.3% and an F1-score of the same order nearing the 93%. 

#### Solution

We uploaded the solution in json and csv formats.

#### License

MIT license

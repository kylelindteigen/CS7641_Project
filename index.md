#### By: Alec Domotor, Kyrylo Kobzyev, Kyle Lindteigen, Yogesh Raparia, Jaya Sai Veepuru

## Introduction/Background

A couple of our team members are fans of the NBA and are interested in the idea of sports analytics. We as a group also wanted to create something that could potentially financially benefit us and from the combination of those two came our idea to look into different ways of beating the betting odds for NBA games.

## Problem Definition

We would like to find a way to correctly predict the outcome of NBA games, and specifically we would like to create a model that will determine the odds of a team beating the betting spread on those games. More broadly we would also like to look into finding ways to group good teams and bad ones using their statistical seasonal averages.

## Data Collection

There are three sources we got our data for this project from. The first is basketball reference which will give us the seasonal statistics of each team per 100 possessions. The reason the per 100 possessions is important is because it factors out the pace that a team plays at which can skew a teams per game averages. Some factors in this dataset include field goal percentage, 3 point percentage, assists, turnovers, points, etc. We also include the data on the opponents per 100 possession stats with the same factors that are included in the teams per 100 stats. The reason we decided to include this is because the statistics involving defense are not sufficient enough to capture how good a teams defense is so including the rates against opponents should fill this gap a bit. As for data cleaning we ended up normalizing all of the teams seasonal statistics per season.

The second source we use is GoldSheet which has per game plus minus odds going back all the way to 1994 which gives us a lot of data to work with. The important factor in this data set is the plus minus but it also includes the results of the games which is what we would be looking to predict. The third data set comes from sports book reviews online which has the betting odds for a team winning or losing. In order to clean these data sets we simply combine them to include the three factors that we are looking to use and then we also want to include the season statistics that we mentioned in our first dataset.

For the feature selection, since this is the midterm report and we have not quite finished tinkering with what works well with our project we ended up using most of the factors from the first source mentioned in our unsupervised learning model. Some factors that are included in our data sets that we don’t think we will need are games played, minutes played, stadium, age of players, and similar data that doesn’t have much to do with on the court performance. When we get to our supervised learning we will need to think more about the data we include because some of the factors are not independent, for example field goal percentage and true shooting percentage are not independent of one another.


## Methods

### Unsupervised learning:

For unsupervised learning, we would want to cluster the teams or players that are likely to perform better and win similar to this article [James]. For this we would like to use GMM, as it will provide probability for each team, which is likely to go to playoffs or likely to perform better. We could also use K-means for unsupervised learning. But as K-means is a hard classifier, it might not be helpful while placing bets or looking for odds. Also, we would like to use hierarchical clustering to observe which teams and players come under an umbrella and how the model performs differently than K-means.

### Supervised Learning:

We noticed that Neural Networks is the most commonly used algorithm for sports betting and we would like to go in the same direction as was used in this article [Hubáček & Sourek & Železný]. As NN is the last topic in the class course, we want to explore other methods in the meantime. As our problem statement revolves around the odds or probability of a team winning (classification type), we would like to use SVM, Logistic Regression, or Naive Bayes approaches. Out of these approaches, we would like to use Logistic Regression as it will use the probability or Odds while classifying the output. Also, Naive Bayes approaches uses the probability of a team winning under the circumstances of events (x1, x2, x3...). So, we would like to use these two methods and compare the outputs. SVM is a hard classifier, so it might be very helpful for our case.

## Potential results

Our primary focus is on beating betting spreads, so we would like results relating to the odds of teams beating that spread. Our unsupervised model has a different task and that is to group teams and players based on their statistical averages. The results for this will hopefully show groups of teams and players that end up performing better.

## Discussion

This project has a number of potential difficulties the first of which would be dealing with the data and the uncertain nature of sports betting as is talked about in the Charles Malafosse article. Sports are an area that is difficult to predict due to dealing with all of the variables that contribute to the success of a team and there are certain unpredictable aspects such as players getting injured. Not only is it hard to predict but the people creating these betting spreads usually use all the tools at their disposal to create even odds for both sides leaving it more difficult for us to predict one side or the other. Even with this all being the case this project could be very profitable for us should we succeed in creating a good model. 

#### [Link to download video](https://raw.githubusercontent.com/kylelindteigen/CS7641_Project/gh-pages/ProjectProposalVid.mp4)

## References:

Malafosse, Charles. “Machine Learning for Sports Betting: Not a Basic Classification Problem.” Medium, Towards Data Science, 11 Oct. 2019, [https://towardsdatascience.com/machine-learning-for-sports-betting-not-a-basic-classification-problem-b42ae4900782](https://towardsdatascience.com/machine-learning-for-sports-betting-not-a-basic-classification-problem-b42ae4900782). 

Hubáček, Ondřej & Sourek, Gustav & Železný, Filip. (2019). Exploiting sports-betting market using machine learning. International Journal of Forecasting. 35. 10.1016/j.ijforecast.2019.01.001. 

James. "Clustering NBA Playstyles Using Machine Learning" Medium, Towards Data Science, 26 Oct. 2019,
[https://towardsdatascience.com/clustering-nba-playstyles-using-machine-learning-8c7e8e23c90c](https://towardsdatascience.com/clustering-nba-playstyles-using-machine-learning-8c7e8e23c90c).








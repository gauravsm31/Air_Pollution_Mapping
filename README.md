# CSE547-Air-Pollution-Mapping
Air pollution monitoring is done by analyzing particulate matter (PM) in air. PM analysis is important in assessing an individual’s exposure to potentially harmful particles, such as aeroallergens, toxins, and emissions from combustion sources.

Currently, PM is recorded at sparse locations in a geographical area, however, the PM level can vary dramatically over small distances. 

The sparsity of air quality measurement sensors makes assessing PM at specific locations quite difficult. In this project, we evaluated the accuracy of assessing PM levels at specific locations in the city of Krakow in Poland from spatio-temporal data of PM levels by applying
different models. 

We apply three approaches for mapping and prediction: 

1. Using Bellkor recommendation system, and achieved an overall R2 = 0.928 for measuring the PM level trend (Su Ye)

2. PM levels are mapped and predicted onto 1km X 1km grid points over the city using semi-supervised classification. L1-regularized logistic regression along with expectation maximization results in 69.5% accuracyfor mapping and 61.5% - 52% accuracy for prediction from 1 hr - 4 hrs respectively. (Gaurav Mahamuni)

3. A measurement of the PM level trend using radial basis function interpolation achieved an overall R2 = 0.873. (Mingyu Wang)

![alt text](https://github.com/gauravsm31/CSE547-Air-Pollution-Mapping/blob/master/150_og.png)
           PM2.5 concentration labels for 7th March 6:00 AM at 29 sensor location
 
 ![alt text](https://github.com/gauravsm31/CSE547-Air-Pollution-Mapping/blob/master/150_mapped.png)
           PM2.5 concentration labels for 7th March 6:00 AM mapped by semi-supervised L1-regularized logistic regression model.
 

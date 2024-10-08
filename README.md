# 2024 Infodengue Sprint: Dengue Fever Predictive Modeling in Brazil

## Background
The year 2024 has seen an exceptional number of reported dengue fever cases in various parts of the world. In Brazil, as of august, 5, were reported 6.4 million cases and 5080 confirmed deaths. 

the disease has spread to new areas in the south and at altitudes where epidemics were not previously recorded, or or reemerging in areas whose control programs had already eradicated them, with an incidence rate has far exceeding that of previous years. The objective of this sprint, therefore, is to promote, in a standardized way, the training of predictive models with the aim of developing an ensemble forecast for dengue in Brazil. Our premisse is that predicting where the next epidemic will hit is important for allocating resources to reduce diseases burden, enabling timely response.

## Dengue Forecast Ensemble Chalenge
The Infodengue project a works with early warning system to help manage arbovirus diseases response and surveillance, working closely with the Brazilian MOH. he Mosqlimate project’s mission is facilitate modeling the impact of climate on the dynamics of Arboviruses

The dengue  forecast ensemble was designed to work with models from the 2024 Infodengue sprint.This initiative is led by the Mosqlimate project, in collaboration with the Harmonize and IDExtremes projects. The nvited teams are collaborators of the Infodengue team with stronger experience with dengue modelling.

In this chalenge, for constructing the ensemble forecast model, we combined the best methods submitted that were selected considering their ability to combine the strengths of different types of models. The ensemble forecast model will be built as a Stacking Regression model. 

For this feature, were used Scikit-learn’s Stacking implementation. A meta-estimator was trained on the same data as the individual models and forecasts from individual models were used as a look-up table.

## The Sprint goals
Main goal: provide forecasts for 2025, at state level, by:
1. Organizing a community of modellers with unified goal and methods - done
2. Together generate a set of independent models tested using data from previous seasons - done
3. Train ensemble model with all submissions 
4. Produce forecasts for 2025 using the best models, either single or combined
5. Report the results as a technical report to the Ministry of Health
6. Update and monitor the performance of the models in 2025
7. Publish a scientific paper on this experience
8. Organize a larger Sprint initiative in 2025.

## Methodology 
The challenge had two test goals and a forecast goal, described below. 

Validation test 1. Predict the weekly number of dengue cases by state (UF) in the 2022-2023 season [EW 41 2022- EW40 2023], using data covering the period from EW 01 2010 to EW 25 2022;

Validation test 2. Predict the weekly number of dengue cases by state (UF) in the 2023-2024 season [EW 41 2023- EW40 2024], using data covering the period from EW 01 2010 to EW 25 2023;

Forecast target. Predict the weekly number of dengue cases in Brazil, and by state (UF), in the 2024-2025 season [EW 41 2024- EW40 2025], using data covering the period from EW 01 2010 to EW 25 2024;

The date range of interest comprised a period between epidemiological week (EW) 41 of one year and EW 40 of the following year, which corresponds to a typical dengue season in Brazil. Participants were encouraged to generate results for all 27 federative units in Brazil or, optionally, they could also generate results for a minimum set of states that represented all regions of Brazil, namely: North: Amazonas (AM), Northeast: Ceará (CE ), Midwest: Goiás (GO), Southeast: Minas Gerais (MG), South: Paraná (PR)

The following outcomes should be provided by the models, both with point estimates and predictive intervals of 90%: (i) Curve of dengue cases by EW for the 2022-2023 and 2023-2024 seasons and (ii) accumulated number of cases probable cases of dengue fever for UF in the 2022-2023 and 2023-2024 seasons

The training data sets and their respective variable dictionary were made available on the Mosqlimate platform. In generalthe dataset contained epidemiological, demographic, climate, miscellaneous and environmental data. Other data sources indicated by the participants themselves were shared on the same platform since they followed the same characteristics of the platform data: open access, updatable and available to all Brazilian states.

All information to make the developed models available was published in the repository https://github.com/Mosqlimate-project/sprint-template

## Results
There are seven teams participating in the Dengue 2024 Sprint that were released several models to dengue prediction presented in a workshop on August, 16.  
1. D-fense - 
2. Dobby Data - LTSH model 
3. GeoHealth - Prophet model with PCA and vaiance threshold and LSTM model with PCA and vaiance threshold Models 	 
4. Global Health Resilience - Temp-SPI Interaction Model
5. PET - BB-M Model
6. Ki-Dengu Peppa -  Weekly and yearly (iid) components and Weekly and yearly (rw1) components Models	 
7. DS_OKSTATE - Info dengue CNN LSTM Ensemble Model	 

### Models description

### Predictions



## Forecast Evaluation
The Mosqlimate group evaluated the performance of each model using a set of scores. The logarithmic score, CRPS and the interval score were computed using the 'ScoringRules Python package'. Other metrics were calculated as additional feedback for the teams, without affecting the classification of the models, such as (i) average scores in these regions of interest in the prediction window, considering epidemic onset (weeks between growth start and the peak) and epidemic peak (3 week window centered on the peak) and (ii) the time lag, maximizing cross-correlation between forecasts and data

## Ranking:

Individual scores were calculated for each state and each year, corresponding to test 1 and test 2. Based on these scores, the concordance models were classified with different challenges. For each year and state, the models were assessed according the score and the predicted epidemiological week, for each year and state. At the end, a global ranking was calculated using a similar method. 

For the emsemble, the models werw added to the set incrementally, following the raking order until there is no further improvement in performance.

The dettailed results are available in this repository.

# My Master Thesis

This repository contains all the work I did on my master thesis, titled _Predicting the Momentum Flux-Profile Relationship from Macro Weather Parameters_. You will also find my minor thesis (not really proud of it) and the plan for this project (even though we completely changed it after a few weeks)

## Abstract
The study of climate heavily relies on simulations. For efficiency reasons, many important phenomena cannot be simulated, and have to be parametrized, i.e. their effect must be described based on macro parameters. The turbulence resulting from the interaction between the atmosphere and the surface is an example of such phenomena. One of the quantities of interest arising from turbulence is the momentum flux-profile relationship, which relates the transport of momentum (flux) and the change of wind speed with altitude (profile). This quantity can be computed following the Monin-Obukhov Similarity Theory (Obukhov, 1971). However, this theory requires parameters that are hard to measure, both in real life and in the simulations, is only applicable in a restricted range of conditions, and produces predictions that are accurate only up to 20-30\% (Foken, 2006).

The goal of this thesis is to compute the momentum flux-profile relationship using only macro weather parameters, which are readily available in climate simulations; this is done using Data Mining techniques on 17 years of weather data collected from the Cabauw meteorological tower in the Netherlands. Moreover, we asses the impact of different sets of features on the prediction error.

Results show that even the simplest linear models are able to compete with the similarity theory, and complex methods such as gradient boosted trees can reduce the mean squared error by almost 50\%. Furthermore, these methods are applicable to a much wider range of conditions compared to the similarity theory, while providing roughly the same predictive performance achieved by this theory _in its validity range_. These results are obtained using wind speed and air temperature at different levels, the soil temperature, and the net radiation at the surface; the improvement offered by the heat fluxes is significant, but of low magnitude. The soil heat flux, the dew point, and the hourly trend of the features do not have a tangible impact on the performance.

Read the full thesis [here](manuscript/build/thesis.pdf) (download it).

## Description of the notebooks
 - `create_dataset.ipynb`: combines the individual datasets released by the [Cesar consortium](http://www.cesar-observatory.nl/) into a Pandas dataframe, used by all the other notebooks. After you register, go to the [database web portal](http://www.cesar-database.nl/Welcome.do) and download the dataset with the surface fluxes, surface mete, tower meteo and soil heat.
 - `exploratory_data_analysis.ipynb` where I look around and make sure everythin is correct. You will find the summaries, tSNE plots, the gradient comparison, and tons of scatter plots.
 - `fit_estimators.ipynb` contains the pyspark code to run the cross validations and the ensemble of linear models. Made to be run on [hopsworks](https://www.hops.site/hopsworks/).
 - `neural_networks.ipynb` contains experiments with neural networks, both fully connected and recurrent;
 - `hyperopt.ipynb` contains the code to optimize the hyper-parametes of xgboost with GPyOpt and [HyperBand](https://github.com/zygmuntz/hyperband), as well as the charts for the prediction error during a cyclone, density plots, and error as a function of the instability parameter.
 - `process_results.ipynb` code to read the result files dumped by `fit_estimators.ipynb` and create charts and summary tables of the results.
 - `neural_architecture_search_impl.ipynb` contains my own implementation of neural architecture search with reinforcement learning, roughly following [this paper](https://arxiv.org/abs/1611.01578). Not really validated.
 - `cabauw_nas.ipynb` is where I tried to apply neural architecture search to my dataset. Utter failure, do not bother.

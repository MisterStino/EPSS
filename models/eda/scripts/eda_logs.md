## Stijn observations
I think that if we plot 97th percentile epss scores we see that many of these jump on 7th of march 2023 -- when : EPSS v3 (v2023.03.01) started publishing on 2023-03-07, you will see a shift in scores on that day.

Question here is: is this because EPSS actually improved and started picking up actually more risky cve. Or is it artificial. Or both? And what should we do for the model to learn properly. If it is artificial this might mess with the models ability to correctly learn from features.

After further inspection we also see that some cve go from very high to very low, and gradually move upwards after that. So this is artificial flip as well. 

## I think we should include both calendar date in the features of the cve and the version of epss that is active on that date. Give model some info to attribute this to.


## We still have to introduce missing values and handle these.
So currently there might be non continuity in a cve's time series. E.g. a missing date. There is no missing value, because we simply were not able to fetch it. So we should for each cve check if time line is continous in calendar date. If not: introduce missing value there. Next: find the best way to handle this. 
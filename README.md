# ** BAYESIAN LOGISTIC REGRESSION ANALYSIS IN R TO PREDICT THE SURVIVAL OF A BREAST CANCER PATIENT **

## INTRODUCTION
In this project **Bayesian logistic regression** is applied to find the **linear decision boundary** that can predict the **survival of patients who had undergone surgery for breast cancer**. The dataset contains cases from a study that was conducted between **1958 and 1970 at the University of Chicago’s Billings Hospital** on the survival of patients who had undergone surgery for breast cancer.

## OVERVIEW 
**Breast cancer** is cancer that forms in the cells of the breasts.
After skin cancer, breast cancer is the most common cancer diagnosed in women in the United States. Breast cancer can occur in both men and women, but it's far more common in women.
Substantial support for breast cancer awareness and research funding has helped create advances in the diagnosis and treatment of breast cancer. Breast cancer survival rates have increased, and the number of deaths associated with this disease is steadily declining, largely due to factors such as earlier detection, a new personalized approach to treatment and a better understanding of the disease.

### SOURCE 
Data is collected from https://archive.ics.uci.edu/ml/datasets/Haberman%27s+Survival. 
It contains **306 observations** each having 3 predictors of numeric type and one dependent binary variable.
It contains the following attributes:
1. Age - Age of patient at time of operation - predictor
2. Patient’s year of operation. - Predictor
3. Number of positive axillary nodes detected - Predictor
4. Survival status (1 is survived 5 year or longer 2 otherwise) - Dependent Variable

## DATA CLEANING AND PRE-PROCESSING
Patient’s year of operation may not provide any meaningful information to the model hence can be removed from the feature set. Also, Survival encoding has been changed i.e.. for non-survival coding is replaced from 2 to 0.
```
dat <- read.csv("haberman.data") dat$X64 <-NULL
names(dat) <- c("Age","PosAuxNodes","Survived") dat$Age <-(dat $Age -mean (dat$Age))/sd(dat$Age)
dat$PosAuxNodes <-(dat $PosAuxNodes -mean (dat$PosAuxNodes))/sd(dat$PosAuxNodes) dat$Survived[dat$Survived == 2] <-0
```
## BAYERSIAN LOGISTIC MODEL
The linear combination of predictors can form a linear model which is given as the input to the logistic function which gives probability between 0-1. Bernoulli function takes this probability and decides the label for output variable.
This logistic model is represented by the following equation:
![logistic model](https://github.com/ksuraj93/Bayesian-Logistic-regression_Breast-Cancer-Patient-Survival-Prediction-in-R/blob/master/bayer_!.JPG)
![logistic model 2](https://github.com/ksuraj93/Bayesian-Logistic-regression_Breast-Cancer-Patient-Survival-Prediction-in-R/blob/master/22.JPG)
where,
Each coefficient of prediction and including intercept term, the prior distribution of these are assumed to follow normal distribution with mean 0 and standard deviation of 1. As it is assumed none of the predictor is important for this model. Once Bayesian model is fitted it gives the posterior distribution which can be used in determining the importance of corresponding predictor.

Dependent variable Y follows Bernoulli distribution based on the probability return by logit function. Logit function takes linear combinations of coefficient and predictive variables along with intercept term.
## JAG’s MODEL
As the prior distribution are defined, now to calculate the posterior distribution. It is determined using MCMC chain simulation by applying JAG’s model.
```
cat("model {
for ( i in 1:Ntotal ) {
# In JAGS, ilogit is logistic:
y[i] ~ dbern( ilogit( zbeta0 + sum( zbeta[1:Nx] * zx[i,1:Nx] ) ) )
}
# Priors vague on standardized scale: zbeta0 ~ dnorm( 0 , 1/2^2 )
for ( j in 1:Nx ) {
zbeta[j] ~ dnorm( 0 , 1/2^2 )
}
# Transform to original scale: beta[1:Nx] <- zbeta[1:Nx] / xsd[1:Nx]
beta0 <- zbeta0 - sum( zbeta[1:Nx] * xm[1:Nx] / xsd[1:Nx] )
}",file="jag.model.txt")
x =dat[,1 :2]
xm = apply(x,2, mean) xsd = apply(x,2, sd)
df <- list("Ntotal"= nrow(dat),
"y"=dat $Survived, "zx"=x,
"Nx"=2,
"xm"=xm, "xsd"=xsd)
bayes.mod.params <- c("beta0","beta[1]","beta[2]") bayes.mod.fit <- jags(data =df,
parameters.to.save =bayes.mod.params, n.chains =3,model.file = 'jag.model.txt')
```
![JAGS](https://github.com/ksuraj93/Bayesian-Logistic-regression_Breast-Cancer-Patient-Survival-Prediction-in-R/blob/master/33.JPG)
```
bayes.mod.fit.mcmc <- as.mcmc(bayes.mod.fit)
densityplot(bayes.mod.fit.mcmc,layout= c(2,2),aspect="fill")
```
![Jags_1](https://github.com/ksuraj93/Bayesian-Logistic-regression_Breast-Cancer-Patient-Survival-Prediction-in-R/blob/master/44.JPG)
From the posterior distribution it is observed that each coefficient mean is not close to 0. It means each predictor is important in determining the survival of the patient. 
 ```
 v=posterior.mode(bayes.mod.fit.mcmc)
 ```


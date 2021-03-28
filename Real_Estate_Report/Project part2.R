## Project part 2

## Question 1:

real_estate <- read.csv(file = file.choose(), header = TRUE)
names(real_estate)[1] <- "Price"

library(tidyverse)
library(car)

subset <- real_estate[, c(1, 8:10)]
pairs(subset, pch = 21, bg = c("blue", "red", "green"))
cor(real_estate[, c(1, 8:10)])
vif(lm_subset)
## Yes, there is a significant evidence of multicolinearity: in the subplot(2, 3) 
## we notice that there is a strong correlation between Living.Area and Bedrooms, a phenomenon 
## where we can conclude collinearity between these two predictor variables. Same kind of evidence can 
## be provided from correlation matrix of these four variables: we can find that the value of correlation 
## between Living.Area and Bedrooms is 0.7044476, much larger than any other values of different pairs in 
## this correlation matrix.

## Question 2:

lm_subset <- lm(Price ~ ., data = subset)
summary(lm_subset)

## LS regression function is 28357.351(beta 0) + 131.440*X1(beta 1, Living.Area) - 15873.874*X2(
## beta 2, Bedrooms) + 1138.275*X3(beta 3, Fireplaces)

## beta 2(the coefficient of the number of bedrooms) indicates that, given other condition constant,  
## as beta 2 increases 1 unit, E{Price} also tends to decrease 15873.874 units.

## Question 3:

anova(lm(Price ~ 1, data = real_estate), lm_subset)
## The alternatives of F test in this model are:
## H0: beta1 = beta2 = beta3 = 0 (no linear relationship between X variables and the target variable)
## H1: at least one of variable has linear relartionship with the target
## The value of F test is 122.8, and the P value of F test is less than 2.2e-16, which is close to zero.
## The conclusion is that we accept the Ha and reject H0.

## Question 4:

library(car)
anova(lm(Price ~ Living.Area, data = real_estate), 
      lm(Price ~ Living.Area + Bedrooms, data = real_estate), 
      lm(Price ~ Living.Area + Bedrooms + Fireplaces, data = real_estate))

Anova(lm_subset,type = 3)
## we notice that the value of extra SSR of X3 is 7.6232e+07.
library(rsq)
lm_subset_withoutx3 <- update(lm_subset, ~.-Fireplaces)
rsq.partial(lm_subset, lm_subset_withoutx3)
## we notice that the value of partial determination is 7.891829e-05.

## Question 5:

anova(lm_subset_withoutx3, lm_subset)
## The alternatives of F test in this model are:
## H0: b3 = 0
## Ha: b3 != 0
## The value of F test is 0.0204, and the P value of F test is less than 0.8864, which is close to 1.
## The conclusion is that we accept the H0 and reject Ha.

## Question 6:
pairs(real_estate)
subset2 <- real_estate[ , c(1,7:8)]
head(subset2)

ggplot(subset2, aes(x = Living.Area, y = Price, col = factor(Central.Air))) + 
  geom_point() + geom_smooth(method = "lm", se = FALSE) + 
  labs(x = "Living Area", y = "Price") + 
  ggtitle("ScatterpLot of The Selling Price Against The Living Area") + 
  theme(plot.title = element_text(hjust = 0.5))

## From this scatterplot, we can notice that these two lines are not coincide. 
## The interaction term actually exists because these two lines are not paralleled and 
## indeed intersect within the sample space.

## Question 7:
x1x2 = subset2$Central.Air*subset2$Living.Area
subset2$Central.Air <- factor(subset2$Central.Air)
lm_subset2 <- lm(Price ~ Living.Area * Central.Air, data = subset2)
summary(lm_subset2)
## the LS Regression function is "Price = 28420.7+98.245*Living.Area - 45769.9*Central.Air + 35.6 * 
## Living.Area:Central.Air
## beta 2 (The coefficient of Central Air) indicates how much lower Price for having central air system 
## is than that for not having central air system, holding X1 = 0; However, there is no practical meaning.
## beta 12(The coefficient of the interaction term between Living Area and Central Air) indicates how much 
## larger/less the magnitude of changes of selling price for a house with a central air system than that of 
## changes of selling price for a house without a central air system.


## Question 8:

## H0: beta 2 = beta 3 = 0
## Ha: at least one beta i (i = 2, 3) != 0
lm_subset2_reduced <- lm(Price ~ Living.Area, data = subset2)
anova(lm_subset2_reduced, lm_subset2)
## the value of F-test statistic is 6.6487, the p-value is 0.001528 < 0.05. So we conclude Ha.

## Question 9:
lm_mod_best <- step(object = lm(Price ~ ., data = real_estate), data = real_estate, direction = "backward")
summary(lm_mod_best)
## from the last result we see that adding Rooms and Bedrooms increases AIC value, so we remove them.
## Therefore the subset of predicted variables that should be included is: waterfront, 
## Land.Value, New.Construct, Central.Air and Living.Area. 


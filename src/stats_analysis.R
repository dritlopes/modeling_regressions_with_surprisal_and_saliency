# 0. SETUP					 		                                                        

# 0.1 - Clear existing workspace objects 
rm(list = ls())

# 0.2 - Set working directory to where the data file is located & results should be saved
setwd("/Users/adriellilopes/Documents/Students/Regression/data/MECO")

# 0.3 - Install packages
# install.packages("BiocManager")
# BiocManager::install("graph")
# install.packages("arules",dependencies = TRUE)
# install.packages("pracma",dependencies = TRUE)
# install.packages("pastecs",dependencies = TRUE)
# install.packages("QuantPsyc",dependencies = TRUE)
# install.packages("lmtest",dependencies = TRUE)
# install.packages("car",dependencies = TRUE)
# install.packages("boot",dependencies = TRUE)
# install.packages("MASS",dependencies = TRUE)
# install.packages("leaps",dependencies = TRUE)
# install.packages("performance", dependencies = TRUE)
# install.packages("sjmisc", dependencies = TRUE)
# install.packages("sjPlot", dependencies = TRUE)
# install.packages("tidyverse")
# install.packages("glue")

# 0.4 - Load packages
library(ggplot2)
library(Hmisc)
library(ggm)
library(polycor)
library(gridExtra)
library(grid)
library(arules)
library(pracma)
library(pastecs)
library(QuantPsyc)
library(lmtest)
library(car)
library(boot)
library(gmodels)
library(MASS)
library(psych)
library(nlme)
library(reshape)
library(lme4)
library(plyr)
library(performance)
library(stats)
library(tidyverse)
library(gamm4)
library(glue)

# documentation: glmer: Fitting Generalized Linear Mixed-Effects Models
# https://www.rdocumentation.org/packages/lme4/versions/1.1-34/topics/glmer

# 1. SURPRISAL

model<-'gpt2'
surprisal_df<-read.csv(glue("surprisal_{model}_fixation_df.csv"),header=T)
head(surprisal_df)
tail(surprisal_df)
dim(surprisal_df)
names(surprisal_df)
summary(surprisal_df)
View(surprisal_df)
psych::describe(surprisal_df)
# psych::describe(nword$reg.dist)
# table(nword$reg.dist)
# table(nword$line.change)

# 1.1. OUTGOING REGRESSIONS

# 1.1.1. Intercept-only model and Random comparison
# Fixed intercept
IntOnly <- glm(reg.out ~ 1, data=surprisal_df, family = "binomial") 
summary(IntOnly)
# Random intercept (1|uniform_id)
RandomIntOnly <- glmer(reg.out~1 + (1|uniform_id), data=surprisal_df, family = "binomial") 
summary(RandomIntOnly)
# Intraclass Correlation Coeficcient
icc(RandomIntOnly)

# 1.1.2. N

# Surprisal, length and frequency
glmerSurprisalLenFreqRegOut <- glmer(reg.out ~ surprisal + length.log + frequency + (1|uniform_id), data=surprisal_df, family = "binomial")
summary(glmerSurprisalLenFreqRegOut)

# Surprisal, length, frequency, distance and interaction
glmerSurprisalLenFreqDistIntRegOut <- glmer(reg.out ~ surprisal + length.log + frequency + reg.dist.log + surprisal*reg.dist.log + (1|uniform_id), data=surprisal_df, family = "binomial")
summary(glmerSurprisalLenFreqDistIntRegOut)

# 1.1.3. N+1

# Surprisal, length and frequency
glmerSurprisalLenFreqRegOutPlusOne <- glmer(reg.out.plus.one ~ surprisal + length.log + frequency + (1|uniform_id), data=surprisal_df, family = "binomial")
summary(glmerSurprisalLenFreqRegOutPlusOne)

# Predict with model SurprisalRegOutN+1
surprisal_df_copy <- surprisal_df[!is.na(surprisal_df$surprisal),]
surprisal_df_copy <- surprisal_df_copy[!is.na(surprisal_df_copy$length.log),]
surprisal_df_copy <- surprisal_df_copy[!is.na(surprisal_df_copy$frequency),]
surprisal_df_copy <- surprisal_df_copy[!is.na(surprisal_df_copy$uniform_id),]
surprisal_df_copy <- surprisal_df_copy[!is.na(surprisal_df_copy$reg.out.plus.one),]
predicted_regression_likelihood <- predict(
  glmerSurprisalLenFreqRegOutPlusOne, 
  newdata = surprisal_df_copy,
  type = "response")
head(predicted_regression_likelihood)
length(predicted_regression_likelihood)
surprisal_df_copy$predicted_regression_likelihood = predicted_regression_likelihood
write.csv(surprisal_df_copy, "glmerSurprisalLenFreqRegOutPlusOne_predicted_regressions_gpt2large.csv")

# Surprisal, length, frequency, distance and interaction
glmerSurprisalLenFreqDistIntRegOutPlusOne <- glmer(reg.out.plus.one ~ surprisal + length.log + frequency + reg.dist.log.plus.one + surprisal*reg.dist.log.plus.one + (1|uniform_id), data=surprisal_df, family = "binomial")
summary(glmerSurprisalLenFreqDistIntRegOutPlusOne)
# Predict with model SurprisalRegOutN+1
surprisal_df_copy <- surprisal_df[!is.na(surprisal_df$surprisal),]
surprisal_df_copy <- surprisal_df_copy[!is.na(surprisal_df_copy$length.log),]
surprisal_df_copy <- surprisal_df_copy[!is.na(surprisal_df_copy$frequency),]
surprisal_df_copy <- surprisal_df_copy[!is.na(surprisal_df_copy$uniform_id),]
surprisal_df_copy <- surprisal_df_copy[!is.na(surprisal_df_copy$reg.out.plus.one),]
surprisal_df_copy <- surprisal_df_copy[!is.na(surprisal_df_copy$reg.dist.log.plus.one),]
predicted_regression_likelihood <- predict(
  glmerSurprisalLenFreqDistIntRegOutPlusOne, 
  newdata = surprisal_df_copy,
  type = "response")
head(predicted_regression_likelihood)
length(predicted_regression_likelihood)
surprisal_df_copy$predicted_regression_likelihood = predicted_regression_likelihood
write.csv(surprisal_df_copy, "glmerSurprisalLenFreqDistIntRegOutPlusOne_predicted_regressions.csv")

# Models per distance
for (distance in unique (surprisal_df$dist)){
  nword <- subset(surprisal_df, reg.dist.binned.plus.one == 'NoDistance' | reg.dist.binned.plus.one == distance)
  model <- glmer(reg.out.plus.one ~ surprisal + length.log + frequency + (1|uniform_id), data=nword, family = "binomial")
  summary(model)
  predicted_regression_likelihood <- predict(
    model, 
    newdata = surprisal_df_copy,
    type = "response")
  surprisal_df_copy$predicted_regression_likelihood = predicted_regression_likelihood
  write.csv(surprisal_df_copy, glue("glmerSurprisal{distance}_predicted_regressions.csv"))
}

# 1.2. INCOMING REGRESSIONS

# Surprisal, length and frequency
glmerSurprisalLenFreqRegIn <- glmer(reg.in ~ surprisal + length.log + frequency + (1|uniform_id), data=surprisal_df, family = "binomial")
summary(glmerSurprisalLenFreqRegIn)
# Surprisal, length, frequency, distance and interaction
glmerSurprisalLenFreqDistIntRegIn <- glmer(reg.in ~ surprisal + length.log + frequency + reg.dist.in.log + surprisal*reg.dist.in.log + (1|uniform_id), data=surprisal_df, family = "binomial")
summary(glmerSurprisalLenFreqDistIntRegIn)
# Modes per distance
for (distance in unique (surprisal_df$dist)){
  nword <- subset(surprisal_df, reg.dist.in.binned == 'NoDistance' | reg.dist.in.binned == distance)
  model <- glmer(reg.in ~ surprisal + length.log + frequency + (1|uniform_id), data=nword, family = "binomial")
  summary(model)
}


# 2. SALIENCY

model<-'gpt2'
saliency_df<-read.csv(glue("saliency_{model}_regIn.csv"),header=T)
head(saliency_df)
tail(saliency_df)
dim(saliency_df)
names(saliency_df)
summary(saliency_df)
View(saliency_df)
psych::describe(saliency_df)

saliency_df_copy <- saliency_df[!is.na(saliency_df$uniform_id),]
saliency_df_copy <- saliency_df_copy[!is.na(saliency_df_copy$ianum),]
saliency_df_copy <- saliency_df_copy[!is.na(saliency_df_copy$frequency),]
saliency_df_copy <- saliency_df_copy[!is.na(saliency_df_copy$length),]

# 2.1. N

saliency_df <- saliency_df_copy[!is.na(saliency_df_copy$saliency),]
# Saliency, length, frequency & participants, regression origin
glmerSaliencyLenFreqRegIn <- glmer(reg.in ~ saliency + length + frequency + (1|uniform_id/ianum), data=saliency_df, family = "binomial")
print(summary(glmerSaliencyLenFreqRegIn))
# Predict with model
predicted_regressionIn_likelihood <- predict(
  glmerSaliencyLenFreqRegIn,
  newdata = saliency_df,
  type = "response")
saliency_df$predicted_regressionIn_likelihood = predicted_regressionIn_likelihood
write.csv(saliency_df, glue("glmerSaliencyLenFreqRegIn_predicted_regressions_{model}.csv"))
# Models per distance
for (distance in unique (saliency_df$dist)) {
  saliency_df_dist<-subset(saliency_df, dist == distance)
  n_instances<-nrow(saliency_df_dist)
  n_reg_in<-length(which(saliency_df_dist$reg.in==1))
  if (n_reg_in > 1){
    if (n_reg_in > n_instances * .01) {
      print(distance)
      print(n_instances)
      print(n_reg_in)
      glmerSaliencyRegInDist <- glmer(reg.in ~ saliency + length + frequency + (1|uniform_id/ianum), data=saliency_df_dist, family = "binomial")
      print(summary(glmerSaliencyRegInDist))
      # Predict with model
      predicted_regressionIn_likelihood <- predict(
        glmerSaliencyRegInDist,
        newdata = saliency_df_dist,
        type = "response")
      saliency_df_dist$predicted_regressionIn_likelihood = predicted_regressionIn_likelihood
      write.csv(saliency_df_dist, glue("glmerSaliencyRegIn{distance}_predicted_regressions_{model}.csv"))
    }
  }
}

# 2.2. N - 1

saliency_df <- saliency_df_copy[!is.na(saliency_df_copy$saliency.minus.one),]
glmerSaliencyLenFreqRegIn <- glmer(reg.in ~ saliency.minus.one + length + frequency + (1|uniform_id/ianum), data=saliency_df, family = "binomial")
print(summary(glmerSaliencyLenFreqRegIn))
# Predict with model
predicted_regressionIn_likelihood <- predict(
  glmerSaliencyLenFreqRegIn,
  newdata = saliency_df,
  type = "response")
saliency_df$predicted_regressionIn_likelihood = predicted_regressionIn_likelihood
write.csv(saliency_df, glue("glmerSaliencyMinusOneLenFreqRegIn_predicted_regressions_{model}.csv"))
# Models per distance
for (distance in unique (saliency_df$dist)) {
  saliency_df_dist<-subset(saliency_df, dist == distance)
  n_instances<-nrow(saliency_df_dist)
  n_reg_in<-length(which(saliency_df_dist$reg.in==1))
  if (n_reg_in > 1){
    if (n_reg_in > n_instances * .01) {
      print(distance)
      print(n_instances)
      print(n_reg_in)
      glmerSaliencyRegInDist <- glmer(reg.in ~ saliency.minus.one + length + frequency + (1|uniform_id/ianum), data=saliency_df_dist, family = "binomial")
      print(summary(glmerSaliencyRegInDist))
      # Predict with model
      predicted_regressionIn_likelihood <- predict(
        glmerSaliencyRegInDist,
        newdata = saliency_df_dist,
        type = "response")
      saliency_df_dist$predicted_regressionIn_likelihood = predicted_regressionIn_likelihood
      write.csv(saliency_df_dist, glue("glmerSaliencyMinusOneRegIn{distance}_predicted_regressions_{model}.csv"))
    }
  }
}


# 2.2. N + 2

saliency_df <- saliency_df_copy[!is.na(saliency_df_copy$saliency.plus.two),]
glmerSaliencyLenFreqRegIn <- glmer(reg.in ~ saliency.plus.two + length + frequency + (1|uniform_id/ianum), data=saliency_df, family = "binomial")
print(summary(glmerSaliencyLenFreqRegIn))
# Predict with model
predicted_regressionIn_likelihood <- predict(
  glmerSaliencyLenFreqRegIn,
  newdata = saliency_df,
  type = "response")
saliency_df$predicted_regressionIn_likelihood = predicted_regressionIn_likelihood
write.csv(saliency_df, glue("glmerSaliencyPlusTwoLenFreqRegIn_predicted_regressions_{model}.csv"))
# Models per distance
for (distance in unique (saliency_df$dist)) {
  saliency_df_dist<-subset(saliency_df, dist == distance)
  n_instances<-nrow(saliency_df_dist)
  n_reg_in<-length(which(saliency_df_dist$reg.in==1))
  if (n_reg_in > 1){
    if (n_reg_in > n_instances * .01) {
      print(distance)
      print(n_instances)
      print(n_reg_in)
      glmerSaliencyRegInDist <- glmer(reg.in ~ saliency.plus.two + length + frequency + (1|uniform_id/ianum), data=saliency_df_dist, family = "binomial")
      print(summary(glmerSaliencyRegInDist))
      # Predict with model
      predicted_regressionIn_likelihood <- predict(
        glmerSaliencyRegInDist,
        newdata = saliency_df_dist,
        type = "response")
      saliency_df_dist$predicted_regressionIn_likelihood = predicted_regressionIn_likelihood
      write.csv(saliency_df_dist, glue("glmerSaliencyPlusTwoRegIn{distance}_predicted_regressions_{model}.csv"))
    }
  }
}
# Correlation between saliency N and saliency N+2
saliency_df <- saliency_df[!is.na(saliency_df$saliency),]
saliency_df <- saliency_df[!is.na(saliency_df$saliency.plus.two),]
correlation <- cor.test(saliency_df$saliency, saliency_df$saliency.plus.two, method='pearson')
correlation
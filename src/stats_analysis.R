# Author: A.T. Lopes Rego
# Date: 19/03/25


# 0. SETUP					 		                                                        

# 0.1 - Clear existing workspace objects 
rm(list = ls())

# install.packages("itsadug")

# 0.2 Load packages

library(lme4)
library(glue)
library(mgcv)
library(dplyr)
library(itsadug)
library(tidyr)

# 0.2 - Set working directory to where the data file is located & results should be saved
corpus <-'Provo' # switch between corpora to get models for each
setwd(glue("/Users/adriellilopes/PycharmProjects/modeling_regressions_with_surprisal_and_saliency/data/{corpus}/"))


# 1. OUTGOING REGRESSION ANALYSIS (When do we regress?)

model <-'gpt2'
df<-read.csv(glue("processed/surprisal_{model}_fixation_1.csv"),header=T)
head(df)
dim(df)

# Add distance of saccades
# df[c('sac.out.raw')] <- lapply(df[c('sac.out')], function(x) abs(x))
# df[c('sac.out.dist.raw')] <- lapply(df[c('sac.out.dist')], function(x) abs(x)) # distance in words in MECO
# head(df)

# Remove outliers based on z-scores
# df <- subset(df,  z_score > -3 & z_score < 5)
# dim(df)

# correlations
df <- df %>% drop_na('reg.out')
cor(df_no_nan[, c('surprisal', 'frequency', 'length')])
cor(df_no_nan[,c('surprisal', 'ianum')])
cor(df[,c('reg.out', 'ianum')])

# 1.1. Baselines

# length and frequency
gamLenFreq <- gam(reg.out ~ s(log(frequency), k=3) + s(log(length), k=3) + s(participant_id_int, bs='re'), data=df, family='binomial', method='REML')
summary(gamLenFreq)
plot(gamLenFreq, se=TRUE, rug=TRUE, shade = TRUE, shade.col = "lightblue", pages=1)

# 1.2. Surprisal

# Surprisal, length and frequency

df$frequency = log(df$frequency) # needed for plotting
df$length = log(df$length)

gamSurprisalProvo <- gam(reg.out ~ s(surprisal, k=3) + s(frequency, k=3) + s(length, k=3) + s(participant_id_int, bs='re'), data=df, family='binomial', method='REML')
summary(gamSurprisalProvo)
BIC(gamLenFreq) - BIC(gamSurprisalProvo)
plot(gamSurprisalProvo, se=TRUE, rug=TRUE, shade = TRUE, shade.col = "lightblue", pages=1, main=corpus)
gamSurprisalMECO <- gam(reg.out ~ s(surprisal, k=3) + s(frequency, k=3) + s(length, k=3) + s(participant_id_int, bs='re'), data=df, family='binomial', method='REML')
summary(gamSurprisalMECO)
BIC(gamLenFreq) - BIC(gamSurprisalMECO)
plot(gamSurprisalMECO, se=TRUE, rug=TRUE, shade = TRUE, shade.col = "lightblue", pages=1, main=corpus)

# N-1

df$ia.minus.one.frequency = log(df$ia.minus.one.frequency)
df$ia.minus.one.length = log(df$ia.minus.one.length)
df$frequency = log(df$frequency)
df$length = log(df$length)

gamSurprisalMinusOneProvo <- gam(reg.out ~ s(surprisal, k=3) + s(ia.minus.one.surprisal, k=3) + s(frequency, k=3) + s(ia.minus.one.frequency, k=3) + s(length, k=3) + s(ia.minus.one.length, k=3) + s(participant_id_int, bs='re'), data=df, family='binomial', method='REML')
summary(gamSurprisalMinusOneProvo)
# plot(gamSurprisalMinusOneProvo, se=TRUE, rug=TRUE, shade = TRUE, shade.col = "lightblue", pages=1)
BIC(gamSurprisalProvo) - BIC(gamSurprisalMinusOneProvo)

gamSurprisalMinusOneMECO <- gam(reg.out ~ s(surprisal, k=3) + s(ia.minus.one.surprisal, k=3) + s(frequency, k=3) + s(ia.minus.one.frequency, k=3) + s(length, k=3) + s(ia.minus.one.length, k=3) + s(participant_id_int, bs='re'), data=df, family='binomial', method='REML')
summary(gamSurprisalMinusOneMECO)
# plot(gamSurprisalMinusOneMECO, se=TRUE, rug=TRUE, shade = TRUE, shade.col = "lightblue", pages=1)
BIC(gamSurprisalMECO) - BIC(gamSurprisalMinusOneMECO)

# Surprisal n-1 with median split on fixation duration

df_short <- subset(df, dur.bin == 'short')

gamSurprisalShortMinusOneProvo <- gam(reg.out ~ s(surprisal, k=3) + s(ia.minus.one.surprisal, k=3) + s(frequency, k=3) + s(ia.minus.one.frequency, k=3) + s(length, k=3) + s(ia.minus.one.length, k=3) + s(participant_id_int, bs='re'), data=df_short, family='binomial', method='REML')
summary(gamSurprisalShortMinusOneProvo)
# plot(gamSurprisalShortMinusOneProvo, se=TRUE, rug=TRUE, shade = TRUE, shade.col = "lightblue", pages=1)

gamSurprisalShortMinusOneMECO <- gam(reg.out ~ s(surprisal, k=3) + s(ia.minus.one.surprisal, k=3) + s(frequency, k=3) + s(ia.minus.one.frequency, k=3) + s(length, k=3) + s(ia.minus.one.length, k=3) + s(participant_id_int, bs='re'), data=df_short, family='binomial', method='REML')
summary(gamSurprisalShortMinusOneMECO)
# plot(gamSurprisalShortMinusOne, se=TRUE, rug=TRUE, shade = TRUE, shade.col = "lightblue", pages=1)

df_long <- subset(df, dur.bin == 'long')
df_bin <- subset(df_long, sac.out != -1)

gamSurprisalLongMinusOneProvo <- gam(reg.out ~ s(surprisal, k=3) + s(ia.minus.one.surprisal, k=3) + s(frequency, k=3) + s(ia.minus.one.frequency, k=3) + s(length, k=3) + s(ia.minus.one.length, k=3) + s(participant_id_int, bs='re'), data=df_bin, family='binomial', method='REML')
summary(gamSurprisalLongMinusOneProvo)
# plot(gamSurprisalLongMinusOneProvo, se=TRUE, rug=TRUE, shade = TRUE, shade.col = "lightblue", pages=1)

gamSurprisalLongMinusOneMECO <- gam(reg.out ~ s(surprisal, k=3) + s(ia.minus.one.surprisal, k=3) + s(frequency, k=3) + s(ia.minus.one.frequency, k=3) + s(length, k=3) + s(ia.minus.one.length, k=3) + s(participant_id_int, bs='re'), data=df_bin, family='binomial', method='REML')
summary(gamSurprisalLongMinusOneMECO)
# plot(gamSurprisalLongMinusOneMECO, se=TRUE, rug=TRUE, shade = TRUE, shade.col = "lightblue", pages=1)

# Plot

par(mfrow=c(1,1), cex=1.1)
# Get model term data:
st1 <- get_modelterm(gamSurprisalProvo, select=1)
st2 <- get_modelterm(gamSurprisalMECO, select=1)
# plot model terms:
emptyPlot(25, c(-1,+1), h=0,
          main='Surprisal', 
          xmark = TRUE, ymark = TRUE, las=1)
plot_error(st1$surprisal, st1$fit, st1$se.fit, shade=TRUE, col='red')
plot_error(st2$surprisal, st2$fit, st2$se.fit, shade=TRUE, col='blue', lty=4, lwd=2)
# add legend:
legend('bottomleft',
       legend=c('Provo', 'MECO'),
       fill=c(alpha('red'), alpha('blue')),
       bty='n')


# 2. INCOMING REGRESSION ANALYSIS (Where do we regress to?)

model <-'gpt2'
df<-read.csv(glue("processed/saliency_{model}_fixation.csv"),header=T)
head(df)
dim(df)

# needed for plotting
df$context.ia.frequency = log(df$context.ia.frequency) 
df$context.ia.length = log(df$context.ia.length)
df$dist = log(df$dist)

# correlations
df_no_nan <- df %>% drop_na('context.ia.surprisal', 'context.ia.frequency', 'saliency', 'context.ia.length')
cor(df_no_nan[, c('saliency', 'context.ia.surprisal', 'context.ia.frequency', 'context.ia.length')])

# 2.1. Baseline

# Length, frequency, and surprisal

gamLenFreqSurpProvo <- gam(reg.in ~ s(context.ia.surprisal, k=3) + s(dist, k=3) + s(context.ia.frequency, k=3) + s(context.ia.length, k=3) + s(participant_id_int, bs='re'), data=df, family='binomial', method='REML')
summary(gamLenFreqSurpProvo)
# plot(gamLenFreqSurpProvo, se=TRUE, rug=TRUE, shade = TRUE, shade.col = "lightblue", pages=1)
gamLenFreqSurpMECO <- gam(reg.in ~ s(context.ia.surprisal, k=3) + s(dist, k=3) + s(context.ia.frequency, k=3) + s(context.ia.length, k=3) + s(participant_id_int, bs='re'), data=df, family='binomial', method='REML')
summary(gamLenFreqSurpMECO)

# 2.2. Saliency

# Saliency

gamSaliencyProvo <- gam(reg.in ~ s(saliency, k=3) + s(context.ia.surprisal, k=3) + s(dist, k=3) + s(context.ia.frequency, k=3) + s(context.ia.length, k=3) + s(participant_id_int, bs='re'), data=df, family='binomial', method='REML')
summary(gamSaliencyProvo)
BIC(gamSaliencyProvo) - BIC(gamLenFreqSurpProvo)
# plot(gamSaliencyProvo, se=TRUE, rug=TRUE, shade = TRUE, shade.col = "lightblue", pages=1)
gamSaliencyMECO <- gam(reg.in ~ s(saliency,k=3) + s(context.ia.surprisal, k=3) + s(dist, k=3) + s(context.ia.frequency, k=3) + s(context.ia.length, k=3) + s(participant_id_int, bs='re'), data=df, family='binomial', method='REML')
summary(gamSaliencyMECO)
BIC(gamSaliencyMECO) - BIC(gamLenFreqSurpMECO)

# Plot

par(mfrow=c(1,1), cex=1.1)
# Get model term data:
st1 <- get_modelterm(gamSaliencyProvo, select=5)
st2 <- get_modelterm(gamSaliencyMECO, select=5)
# plot model terms:
emptyPlot(3, c(-1,+1), h=0,
          main='Length', 
          xmark = TRUE, ymark = TRUE, las=1)
plot_error(st1$context.ia.length, st1$fit, st1$se.fit, shade=TRUE, col='red')
plot_error(st2$context.ia.length, st2$fit, st2$se.fit, shade=TRUE, col='blue', lty=4, lwd=2)
# add legend:
legend('bottomleft',
       legend=c('Provo', 'MECO'),
       fill=c(alpha('red'), alpha('blue')),
       bty='n')

# Saliency ranking

gamSaliencyRankProvo <- gam(reg.in ~ s(saliency.rank, k=3) + s(context.ia.surprisal, k=3) + s(dist, k=3) + s(context.ia.frequency, k=3) + s(context.ia.length, k=3) + s(participant_id_int, bs='re'), data=df, family='binomial', method='REML')
summary(gamSaliencyRankProvo)
# plot(gamSaliencyRankProvo, se=TRUE, rug=TRUE, shade = TRUE, shade.col = "lightblue", pages=1)
gamSaliencyRankMECO <- gam(reg.in ~ s(saliency.rank, k=3) + s(context.ia.surprisal, k=3) + s(dist, k=3) + s(context.ia.frequency, k=3) + s(context.ia.length, k=3) + s(participant_id_int, bs='re'), data=df, family='binomial', method='REML')
summary(gamSaliencyRankMECO)

# Saliency and Distance Bin
df_sent <- subset(df, sent.change == 1)
# dist_bins <- unique(df_sent$dist.bin)
# dist_bins
# Provo:  "3.0-4.0"   "2.0-3.0"   "1.0-2.0"   "4.0-14.0"  "14.0-19.0" "19.0-59.0"
# MECO:  "6.0-13.0"   "3.0-6.0"    "2.0-3.0"    "1.0-2.0"    "17.0-24.0"  "13.0-17.0"  "24.0-184.0"
df_bin <- subset(df_sent, dist.bin == "24.0-184.0")
# table(df_bin$reg.in)
glmerSaliencyDistBin <- glmer(reg.in ~ saliency + context.ia.surprisal + context.ia.frequency + context.ia.length + (1|participant_id), data=df_bin, family='binomial')
summary(glmerSaliencyDistBin)

gamSaliencyDistBin <- gam(reg.in ~ s(saliency,k=3) + s(context.ia.surprisal, k=3) + s(context.ia.frequency, k=3) + s(context.ia.length, k=3) + s(participant_id_int, bs='re'), data=df_bin, family='binomial', method='REML')
summary(gamSaliencyDistBin)
plot(gamSaliencyDistBin, se=TRUE, rug=TRUE, shade = TRUE, shade.col = "lightblue", pages=1)


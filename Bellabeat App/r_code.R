library(tidyverse)
library(lubridate) 
library(dplyr)
library(ggplot2)
library(tidyr)
library(janitor)

activity_DB <- read.csv("../input//dailyActivity_merged.csv")
calories_DB <- read.csv("../input/hourlyCalories_merged.csv")
intensities_DB <- read.csv("../input/hourlyIntensities_merged.csv")
sleep_DB <- read.csv("../input/sleepDay_merged.csv")
weight_DB <- read.csv("../input/weightLogInfo_merged.csv")

head(activity_DB)
head(sleep_DB)

# activity
activity$ActivityDate=as.POSIXct(activity_DB$ActivityDate, format="%m/%d/%Y", tz=Sys.timezone())
activity$date <- format(activity_DB$ActivityDate, format = "%m/%d/%y")
# sleep
sleep$SleepDay=as.POSIXct(sleep_DB$SleepDay, format="%m/%d/%Y %I:%M:%S %p", tz=Sys.timezone())
sleep$date <- format(sleep_DB$SleepDay, format = "%m/%d/%y")
# intensities
intensities$ActivityHour=as.POSIXct(intensities_DB$ActivityHour, format="%m/%d/%Y %I:%M:%S %p", tz=Sys.timezone())
intensities$time <- format(intensities_DB$ActivityHour, format = "%H:%M:%S")
intensities$date <- format(intensities_DB$ActivityHour, format = "%m/%d/%y")
# calories
calories$ActivityHour=as.POSIXct(calories_DB$ActivityHour, format="%m/%d/%Y %I:%M:%S %p", tz=Sys.timezone())
calories$time <- format(calories_DB$ActivityHour, format = "%H:%M:%S")
calories$date <- format(calories_DB$ActivityHour, format = "%m/%d/%y")

n_distinct(activity_DB$Id)
n_distinct(calories_DB$Id)
n_distinct(intensities_DB$Id)
n_distinct(sleep_DB$Id)
n_distinct(weight_DB$Id)

# activity
activity_DB %>%  
  select(TotalSteps,
         TotalDistance,
         SedentaryMinutes, Calories) %>%
  summary()

# explore num of active minutes per category
activity_DB %>%
  select(VeryActiveMinutes, FairlyActiveMinutes, LightlyActiveMinutes) %>%
  summary()
# sleep
sleep_DB %>%
  select(TotalSleepRecords, TotalMinutesAsleep, TotalTimeInBed) %>%
  summary()
# calories
calories_DB %>%
  select(Calories) %>%
  summary()
# weight
weight_DB %>%
  select(WeightKg, BMI) %>%
  summary()


merged_DB <- merge(sleep_DB, activity_DB, by=c('Id', 'date'))
head(merged_DB)

# aggregate data by day of week to summarize averages 
mutated_DB <- mutate(merged_DB, 
                      day = wday(SleepDay, label = TRUE))
summarized_activity_sleep <- mutated_DB %>% 
  group_by(day) %>% 
  summarise(AvgDailySteps = mean(TotalSteps),
            AvgAsleepMinutes = mean(TotalMinutesAsleep),
            AvgAwakeTimeInBed = mean(TotalTimeInBed), 
            AvgSedentaryMinutes = mean(SedentaryMinutes),
            AvgLightlyActiveMinutes = mean(LightlyActiveMinutes),
            AvgFairlyActiveMinutes = mean(FairlyActiveMinutes),
            AvgVeryActiveMinutes = mean(VeryActiveMinutes), 
            AvgCalories = mean(Calories))
head(summarized_activity_sleep)

# checking for significant change in weight
weight%>%
  group_by(Id)%>%
  summarise(min(WeightKg),max(WeightKg))


#visualization

ggplot(data=activity_DB, aes(x=TotalSteps, y=Calories)) + 
  geom_point() + geom_smooth() + labs(title="Total Steps vs. Calories")

ggplot(data=sleep_DB, aes(x=TotalMinutesAsleep, y=TotalTimeInBed)) + 
  geom_point()+ labs(title="Total Minutes Asleep vs. Total Time in Bed")

ggplot(data=activity_DB, aes(x=Calories, y=TotalSteps)) + 
  geom_point() + geom_smooth() + labs(title="Calories burned for every step taken")

ggplot(data=activity_DB, aes(x=Calories, y=VeryActiveMinutes)) + 
  geom_point() + geom_smooth() + labs(title="Calories burned for every Very Active minutes")


intensities_DB_new <- intensities_DB %>%
  group_by(time) %>%
  drop_na() %>%
  summarise(mean_total_int = mean(TotalIntensity))

ggplot(data=intensities_DB_new, aes(x=time, y=mean_total_int)) + geom_histogram(stat = "identity", fill='blue') +
  theme(axis.text.x = element_text(angle = 90)) +
  labs(title="Average Total Intensity vs. Time")

ggplot(data=merged_DB, aes(x=TotalMinutesAsleep, y=SedentaryMinutes)) + 
  geom_point(color='blue') + geom_smooth() +
  labs(title="Minutes Asleep vs. Sedentary Minutes")

#slicing segments
VeryActiveMin <- sum(activity_DB$VeryActiveMinutes)
FairlyActiveMin <- sum(activity_DB$FairlyActiveMinutes)
LightlyActiveMin <- sum(activity_DB$LightlyActiveMinutes)
SedentaryMin <- sum(activity_DB$SedentaryMinutes)
TotalMin <- VeryActiveMin + FairlyActiveMin + LightlyActiveMin + SedentaryMin

# plotting the chart
slices <- c(VeryActiveMin,FairlyActiveMin,LightlyActiveMin,SedentaryMin)
lbls <- c("VeryActive","FairlyActive","LightlyActive","Sedentary")
pct <- round(slices/sum(slices)*100)
lbls <- paste(lbls, pct)
lbls <- paste(lbls, "%", sep="")
pie(slices, labels = lbls, col = rainbow(length(lbls)), main = "Percentage of Activity in Minutes")


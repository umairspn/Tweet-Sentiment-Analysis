library(tidyverse)

dir.createIfNot <- function(new.dir) {
  if(!dir.exists(new.dir)) {
    dir.create(new.dir)
  }
}

makeSKLearnStructure <- function(datasplit,splitname,content,label,id) {
  dir.createIfNot(splitname)
  labels <- unique(datasplit[[label]])
  
  for(i in labels) {
    dir.createIfNot(paste0(splitname,"/",i))
  }
  
  for(i in 1:nrow(datasplit)) {
    row.i <- datasplit[i,]
    outfn <- paste(splitname,row.i[[label]],row.i[[id]],sep="/")
    write_lines(row.i[[content]],outfn)
  }
}



airsent <- read_csv("Airline-Sentiment-2-w-AA.csv")

airsent.r <- airsent %>% select(`_unit_id`,airline_sentiment,
                                `airline_sentiment:confidence`,
                              airline,text)

## Clean
for(i in 1:nrow(airsent.r)) {
  x <- airsent.r[i,"text"]
  x2 <- str_remove_all(x,"\\\\")
  x3 <- str_remove_all(x2,"http[:a-z.0-9/A-Z]+")
  x4 <- str_extract_all(x3,"[#$a-zA-Z0-9@&;'\"]+")[[1]] %>% paste0(collapse = " ")
  airsent.r[i,"text"] <- str_replace_all(x4,"\n"," ")
}

airsent.c <- airsent.r %>% 
  filter(`airline_sentiment:confidence` == 1) %>% 
  select(-`airline_sentiment:confidence`)

##Split
set.seed(22)
train <- bind_rows(
  filter(airsent.c,airline_sentiment == "negative") %>% sample_n(5000),
  filter(airsent.c,airline_sentiment == "neutral") %>% sample_n(1000),
  filter(airsent.c,airline_sentiment == "positive") %>% sample_n(1000)
)

other <- filter(airsent.c,!text %in% train$text)

test <- bind_rows(
  filter(other,airline_sentiment == "negative") %>% sample_n(1000),
  filter(other,airline_sentiment == "neutral") %>% sample_n(250),
  filter(other,airline_sentiment == "positive") %>% sample_n(250)
)

dev <- filter(other,!text %in% test$text)


write_csv(train,"airline_train.csv")

write_csv(test,"airline_test.csv")

write_csv(dev.posineg,"airline_dev.csv")

makeSKLearnStructure(train,"AirSent_Train",
                     "text","airline_sentiment","_unit_id")

makeSKLearnStructure(test,"AirSent_Test",
                     "text","airline_sentiment","_unit_id")

makeSKLearnStructure(dev,"AirSent_Dev",
                     "text","airline_sentiment","_unit_id")

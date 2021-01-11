
dir_create <- function(newdir) {
  if(dir.exists(newdir)) return(newdir)
  dir.create(newdir)
}

library(readr)

setwd("C:\Users\Umair\Desktop\Semester 1\Intro to NLP\Project\Final Project\NLP-Tweets-Sentiment-master\feature_extraction_example")
trfn <- "lyrics.csv"
tr <- read_csv(trfn)
tr$rn <- rownames(tr)
if(stringr::str_detect(trfn,"test")){
  condir <- "baz_con"
  posdir <- "baz_pos"
  refdir <- "baz_ref"
} else {
condir <- "foo_con"
posdir <- "foo_pos"
refdir <- "foo_ref"
}
dir_create(condir)
dir_create(posdir)
dir_create(refdir)


for(i in unique(tr$label)) {
  dir_create(paste0(condir,"/",i))
  dir_create(paste0(posdir,"/",i))
  dir_create(paste0(refdir,"/",i))
}

for(i in 1:nrow(tr)) {
  labdir <- paste0(tr$label[i],"/")
  confn <- paste0(condir,"/",labdir,tr$rn[i],".txt")
  posfn <- paste0(posdir,"/",labdir,tr$rn[i],".txt")
  reffn <- paste0(refdir,"/",labdir,tr$rn[i],".txt")
  write_lines(tr$content[i],confn)
  write_lines(tr$pos_tags[i],posfn)
  write_lines(tr$ref_tags[i],reffn)
}

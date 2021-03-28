## Data Analysis of Amazon Cellphone review


#1. library prep:

library(tidyverse) # general utility & workflow functions
library(tidytext) # tidy implimentation of NLP methods
library(topicmodels) # for LDA topic modelling 
library(tm) # general text mining functions, making document term matrixes
library(SnowballC) # for stemming

#2. import csv.file:

items <- read.csv(file = file.choose(), header = TRUE)
reviews <- read.csv(file = file.choose(), header = TRUE)

View(items)
View(reviews)

#3. pre-processing raw data:

items <- items %>% mutate(class = case_when(originalPrice == 0 & price == 0 ~ "used",
                                            originalPrice != 0 & price != 0 ~ "promotion",
                                            originalPrice == 0 & price != 0 ~ "new"))

reviews$helpfulVotes <- reviews$helpfulVotes %>% replace_na(0)

reviews <- na.omit(reviews)

View(items)
View(reviews)

#4. brief statistical analysis

##4.1: the number of brands of cellphones:

items %>% distinct(asin) %>% nrow()

##4.2: the number of different classes of cellphones:

items$class %>% factor() %>% summary()

##4.3: the number of people involved:

reviews$name %>% unique() %>% length()

#5 NLP Top Topic Analysis

##5.1: pretest by using LDA model without 2nd pre-processing data: 

top_terms_by_topic_LDA <- function(input_text, plot = TRUE, number_of_topics = 4){
  corpus <- Corpus(VectorSource(input_text))
  DTM <- DocumentTermMatrix(corpus)
  
  unique_indexes <- unique(DTM$i)
  DTM <- DTM[unique_indexes, ]
  
  lda <- LDA(DTM, k = number_of_topics, control = list(seed = 1234))
  topics <- tidy(lda, matrix = "beta")
  
  top_terms <- topics %>% 
    filter(term != "phone.,") %>%
    group_by(topic) %>%
    top_n(10, beta) %>%
    ungroup() %>%
    arrange(topic, desc(beta))
  
  if(plot == TRUE){
    top_terms %>% 
      mutate(term = reorder(term, beta)) %>%  
      ggplot(aes(term, beta, fill = factor(topic))) + 
      geom_col(show.legend = FALSE) + 
      facet_wrap(~ topic, scales = "free") + 
      labs(x = NULL, y = "Beta") + 
      coord_flip() 
  }else{ 
    return(top_terms)
  }
}

top_terms_by_topic_LDA(reviews$body, number_of_topics = 2)

##5.2: 2nd pre-processing data for NLP LDA model:

usable_reviews <- str_replace_all(reviews$body,"[^[:graph:]]", " ") 

# because we need to remove non-graphical characters to use tolower()

reviewsCorpus <- Corpus(VectorSource(usable_reviews))
reviewsDTM <- DocumentTermMatrix(reviewsCorpus)

reviewsDTM_tidy <- tidy(reviewsDTM)

ntlk_stop_words <- tibble(word = c("i", "me", "my", "myself", "we", "our",
                                   "ours", "ourselves", "you", "your", "yours",
                                   "yourself", "yourselves", "he", "him", "his",
                                   "himself", "she", "her", "hers", "herself", 
                                   "it", "its", "itself", "they", "them", "their",
                                   "theirs", "themselves", "what", "which", "who",
                                   "whom", "this", "that", "these", "those", "am",
                                   "is", "are", "was", "were", "be", "been", "being",
                                   "have", "has", "had", "having", "do", "does", "did",
                                   "doing", "a", "an", "the", "and", "but", "if", "or",
                                   "because", "as", "until", "while", "of", "at", 
                                   "by", "for", "with", "about", "against", "between",
                                   "into", "through", "during", "before", "after", 
                                   "above", "below", "to", "from", "up", "down", "in",
                                   "out", "on", "off", "over", "under", "again", 
                                   "further", "then", "once", "here", "there", "when",
                                   "where", "why", "how", "all", "any", "both", "each", 
                                   "few", "more", "most", "other", "some", "such", "no",
                                   "nor", "not", "only", "own", "same", "so", "than",
                                   "too", "very", "s", "t", "can", "will", "just", "don",
                                   "should", "now"))
ntlk_stop_words$word <- paste(ntlk_stop_words$word, ",", sep = "")
ntlk_stop_words2 <- ntlk_stop_words
ntlk_stop_words2$word <- paste(ntlk_stop_words2$word, ",", sep = "")
ntlk_stop_words3 <- ntlk_stop_words
ntlk_stop_words3$word <- paste(ntlk_stop_words3$word, ".", sep = "")
ntlk_stop_words_total <- rbind(ntlk_stop_words, ntlk_stop_words2, ntlk_stop_words3)

custom_stop_words <- tibble(word = c("phone", "phone,", "phone.,", "===>", "amazon", "it.,"))

reviewsDTM_tidy_cleaned <- reviewsDTM_tidy %>%
  anti_join(stop_words, by = c("term" = "word")) %>%
  anti_join(ntlk_stop_words_total, by = c("term" = "word")) %>%
  anti_join(custom_stop_words, by = c("term" = "word"))

cleaned_documents <- reviewsDTM_tidy_cleaned %>%
  group_by(document) %>% 
  mutate(terms = toString(rep(term, count))) %>%
  select(document, terms) %>% 
  unique()


head(cleaned_documents) # to have a quick look at cleaned_documents
View(cleaned_documents) # to view the whole picture of cleaned_documents

top_terms_by_topic_LDA(cleaned_documents$terms, number_of_topics = 2)

reviewsDTM_tidy_cleaned <- reviewsDTM_tidy_cleaned %>%
  mutate(stem = wordStem(term))

cleaned_documents <- reviewsDTM_tidy_cleaned %>%
  group_by(document) %>% 
  mutate(terms = toString(rep(stem, count))) %>%
  select(document, terms) %>%
  unique()

top_terms_by_topic_LDA(cleaned_documents$terms, number_of_topics = 4)

## from the right subplot of the result, we can see that the hottest topic about cellphone
## in customer's review on Amazon are:
## screen, battery, buy, app, android(operation system), day, camera, time, call

top_terms_by_topic_tfidf <- function(text_df, text_column, group_column, plot = TRUE){
  
  group_column <- enquo(group_column)
  text_column <- enquo(text_column)
  
  words <- text_df %>%
    unnest_tokens(word, !!text_column) %>%
    count(!!group_column, word) %>%
    ungroup()
  
  total_words <- words %>% 
    group_by(!!group_column) %>% 
    summarize(total = sum(n))
  
  words <- left_join(words, total_words)
  
  tf_idf <- words %>%
    bind_tf_idf(word, !!group_column, n) %>%
    select(-total) %>%
    arrange(desc(tf_idf)) %>%
    mutate(word = factor(word, levels = rev(unique(word))))
  
  if(plot == TRUE){

    group_name <- quo_name(group_column)
    
    tf_idf %>% 
      group_by(!!group_column) %>% 
      top_n(10) %>% 
      ungroup %>%
      ggplot(aes(word, tf_idf, fill = as.factor(group_name))) +
      geom_col(show.legend = FALSE) +
      labs(x = NULL, y = "tf-idf") +
      facet_wrap(reformulate(group_name), scales = "free") +
      coord_flip()
  }else{

    return(tf_idf)
  }
  
}

reviews <- as_tibble(reviews)
reviews <- mutate(reviews, body = as.character(body)) 
# because tokenizer function doesn't recognize factor data type,
# we need to convert from factor into normal character.

usable_reviews2 <- reviews

usable_reviews2$body <- gsub("[^[:alnum:]]", " ", usable_reviews2$body)

top_terms_by_topic_tfidf(text_df = usable_reviews2, 
                         text_column = body, 
                         group_column = verified, 
                         plot = TRUE) 

## then we find a more interesting things: when verified is TRUE, we can see that the hottest topics are described by Spanish.
## after translation, we know that through tf-idf model, the customers are concerned about:
## battery, recommendation, load, great, quick(speed), past experience (nunca = never or ever), fascination and user's gender (sus = his) 
#!/usr/bin/env /c/Apps/Anaconda3/Scripts/Rscript

# [source] https://gist.github.com/primaryobjects/8038d345aae48ae48988906b0525d175


# has_devel()  # has error
# find_rtools()
# has_devel()  # now it works

# set_config(
#   use_proxy(url = "proxy.bloomberg.com", port = 80)
# )
# set_config(config(ssl_verifypeer = 0L))

# Setup RTools path (optional).
#Sys.setenv(PATH = paste("C:/Apps/Rtools/bin", Sys.getenv("PATH"), sep=";"))
#Sys.setenv(BINPREF = "C:/Apps/Rtools/mingw_$(WIN)/bin/")
# See tutorial: https://github.com/bmschmidt/wordVectors/blob/master/vignettes/introduction.Rmd

# install.packages("devtools")
library(devtools)
# devtools::install_github("bmschmidt/wordVectors")
library(wordVectors)
library(httr)
library(tm)

word2vec = function(fileName) {
    if (grepl('.txt', fileName, fixed = T)) {
        # Convert test.txt to test.bin.
        binaryFileName = gsub('.txt', '.bin', fileName, fixed = T)
    }
    else {
        binaryFileName = paste0(fileName, '.bin')
    }

    # Train word2vec model.
    if (!file.exists(binaryFileName)) {
        # Lowercase and setup ngrams.
        prepFileName = 'temp.prep'
        prep_word2vec(origin = fileName, destination = prepFileName, lowercase = T, bundle_ngrams = 2)

        # Train word2vec model.
        model = train_word2vec(prepFileName, binaryFileName, vectors = 200, threads = 4, window = 12, iter = 5, negative_samples = 0)

        # Cleanup.
        unlink(prepFileName)
    } else {
        model = read.vectors(binaryFileName)
    }

    model
}

###
### Example 1: Simple text file.
###

# Read text file.
doc = readChar('C:/Home/tfR/data/article2.txt', file.info('C:/Home/tfR/data/article2.txt')$size)

# Remove stop-words.
stopwords_regex = paste(stopwords('en'), collapse = '\\b|\\b')
stopwords_regex = paste0('\\b', stopwords_regex, '\\b')
doc = stringr::str_replace_all(doc, stopwords_regex, '')

# Write text file with stop-words removed.
cat(doc, file = "article2.txt", sep = "\n", append = TRUE)

# Train word2vec model and explore.
model = word2vec('C:/Home/tfR/data/article2.txt')

model %>% closest_to("president")
model %>% closest_to("trump")
model %>% closest_to("mcmillon")

# Cleanup.
unlink('article2.txt')



# Plot similar terms to 'president' and 'trump'.
w2v = model[[c("president", "trump"), average = F]]
cosine_similarity = model[1:14,] %>% cosineSimilarity(w2v)
comparision_two_words = cosine_similarity[
    rank(-cosine_similarity[, 1]) < 20 |
    rank(-cosine_similarity[, 2]) < 20,
    ]
plot(comparision_two_words, type = 'n')
text(comparision_two_words, labels = rownames(comparision_two_words))


w2v = model[[c("mcmillon", "trump"), average = F]]
cosine_similarity = model[1:14,] %>% cosineSimilarity(w2v)
comparision_two_words = cosine_similarity[
    rank(-cosine_similarity[, 1]) < 20 |
    rank(-cosine_similarity[, 2]) < 20,
    ]
plot(comparision_two_words, type = 'n')
text(comparision_two_words, labels = rownames(comparision_two_words))



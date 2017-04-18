# --------------------------------Load the dataset----------------------------------------#
# Download datset from https://www.kaggle.com/uciml/sms-spam-collection-dataset

spamsms <- read.csv("D:/TextClassificationR/ClassifyTextR/spamsms.csv", header = TRUE)

# Check the structure of the data.
str(spamsms)

# There are 3 extra columns, remove them
spamsms <- spamsms[,c(1,2)]

# Set the sms column as 'message' and spam text as 'label', which we are going to predict whether spam or not.
colnames(spamsms)[1] <- 'label'
colnames(spamsms)[2] <- 'message'

# Convert the trget column to factor as it is a categorical variable.
spamsms$label <- as.factor(spamsms$label)

# Now, Verify the structure of column
str(spamsms)

#---------------------------------Clean text-------------------------------------#

# Load the package tm
library(tm)

# -----------Create bag of words------------------#
# Create a corpus from message column
bag <- Corpus(VectorSource(spamsms$message))

bag <- tm_map(bag, tolower, lazy = T) 

# Check if it is working fine. Check the 15th document.
writeLines(as.character(bag[[15]]))

bag <- tm_map(bag, PlainTextDocument)

bag <- tm_map(bag, removePunctuation)

bag <- tm_map(bag, removeWords, c(stopwords("english")))

bag <- tm_map(bag, stripWhitespace)

bag <- tm_map(bag, stemDocument, lazy = T)

writeLines(as.character(bag[[15]]))

# Convert bag of words to data frame

frequencies <- DocumentTermMatrix(bag)

# look at words that appear atleast 200 times
findFreqTerms(frequencies, lowfreq = 200)

sparseWords <- removeSparseTerms(frequencies, 0.995)

# convert the matrix of sparse words to data frame
sparseWords <- as.data.frame(as.matrix(sparseWords))

# rename column names to proper format in order to be used by R
colnames(sparseWords) <- make.names(colnames(sparseWords))

str(sparseWords)

sparseWords$label <- spamsms$label

#----------------------------------Predicting whether SMS is spam/non-spam-----------------#

# split data into 75:25 and assign to train and test.
set.seed(987)
library(caTools)
split <- sample.split(sparseWords$label, SplitRatio = 0.75)
train <- subset(sparseWords, split == T)
test <- subset(sparseWords, split == F)

# Support Vector Machine Model
svm.model <- svm(label ~ ., data = train, kernel = "linear", cost = 0.1, gamma = 0.1)
svm.predict <- predict(svm.model, test)
table(test$label, svm.predict) # Confusion matrix

svm.accuracy.table <- as.data.frame(table(test$label, svm.predict))
print(paste("SVM accuracy:",
            100*round(((svm.accuracy.table$Freq[1]+svm.accuracy.table$Freq[4])/nrow(test)), 4),
            "%"))

# Got "SVM accuracy: 96.55 %"

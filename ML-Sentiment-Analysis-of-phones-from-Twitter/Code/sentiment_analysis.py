

import numpy as np
import re
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier
import argparse
import os


vNegative = []
Negative = []
Positive = []
vPositive = []
data_X = ""
data_Y = ""


def generateStopWordList():


    stopWords_dataset = "/Users/siri123/Desktop/ML-Sentiment-Analysis-of-phones-from-Twitter/Data/stopwords.txt"

    stopWords = []


    try:
        fp = open(stopWords_dataset, 'r')
        line = fp.readline()
        while line:
            word = line.strip()
            stopWords.append(word)
            line = fp.readline()
        fp.close()
    except:
        print("ERROR: Opening File")

    return stopWords

def generateAffinityList(datasetLink):

    affin_dataset = datasetLink
    try:
        affin_list = open(affin_dataset).readlines()
    except:
        print("ERROR: Opening File", affin_dataset)
        exit(0)

    return affin_list


def createDictionaryFromPolarity(affin_list):

    words = []
    score = []

    for word in affin_list:
        words.append(word.split("\t")[0].lower())
        score.append(int(word.split("\t")[1].split("\n")[0]))

    for elem in range(len(words)):
        if score[elem] == -4 or score[elem] == -5:
            vNegative.append(words[elem])
        elif score[elem] == -3 or score[elem] == -2 or score[elem] == -1:
            Negative.append(words[elem])
        elif score[elem] == 3 or score[elem] == 2 or score[elem] == 1:
            Positive.append(words[elem])
        elif score[elem] == 4 or score[elem] == 5:
            vPositive.append(words[elem])


def preprocessing(dataSet):

    processed_data = []

    stopWords = generateStopWordList()


    for tweet in dataSet:

        temp_tweet = tweet

        #Convert @username to USER_MENTION
        tweet = re.sub('@[^\s]+','USER_MENTION',tweet).lower()
        tweet.replace(temp_tweet, tweet)

        #Remove the unnecessary white spaces
        tweet = re.sub('[\s]+',' ', tweet)
        tweet.replace(temp_tweet,tweet)

        #Replace #HASTAG with only the word by removing the HASH (#) symbol
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet)

        #Replace all the numeric terms
        tweet = re.sub('[0-9]+', "",tweet)
        tweet.replace(temp_tweet,tweet)

        #Remove all the STOP WORDS
        for sw in stopWords:
            if sw in tweet:
                tweet = re.sub(r'\b' + sw + r'\b'+" ","",tweet)

        tweet.replace(temp_tweet, tweet)

        #Replace all Punctuations
        tweet = re.sub('[^a-zA-z ]',"",tweet)
        tweet.replace(temp_tweet,tweet)

        #Remove additional white spaces
        tweet = re.sub('[\s]+',' ', tweet)
        tweet.replace(temp_tweet,tweet)

        #Save the Processed Tweet after data cleansing
        processed_data.append(tweet)

    return processed_data


def FeaturizeTrainingData(dataset, type_class):

    neutral_list = []
    i=0


    data = [tweet.strip().split(" ") for tweet in dataset]


    feature_vector = []

    for sentence in data:
        vNegative_count = 0
        Negative_count = 0
        Positive_count = 0
        vPositive_count = 0

        for word in sentence:
            if word in vPositive:
                vPositive_count = vPositive_count + 1
            elif word in Positive:
                Positive_count = Positive_count + 1
            elif word in vNegative:
                vNegative_count = vNegative_count + 1
            elif word in Negative:
                Negative_count = Negative_count + 1
        i+=1

        if vPositive_count == vNegative_count == Positive_count == Negative_count:
            feature_vector.append([vPositive_count, Positive_count, Negative_count, vNegative_count, "neutral"])
            neutral_list.append(i)
        else:
            feature_vector.append([vPositive_count, Positive_count, Negative_count, vNegative_count, type_class])

    return feature_vector


def FeatureizeTestData(dataset):

    data = [tweet.strip().split(" ") for tweet in dataset]
    count_Matrix = []
    feature_vector = []

    for sentence in data:
        vNegative_count = 0
        Negative_count = 0
        Positive_count = 0
        vPositive_count = 0

        for word in sentence:
            if word in vPositive:
                vPositive_count = vPositive_count + 1
            elif word in Positive:
                Positive_count = Positive_count + 1
            elif word in vNegative:
                vNegative_count = vNegative_count + 1
            elif word in Negative:
                Negative_count = Negative_count + 1

        if (vPositive_count + Positive_count) > (vNegative_count + Negative_count):
            feature_vector.append([vPositive_count, Positive_count, Negative_count, vNegative_count, "positive"])
        elif (vPositive_count + Positive_count) < (vNegative_count + Negative_count):
            feature_vector.append([vPositive_count, Positive_count, Negative_count, vNegative_count, "negative"])
        else:
            feature_vector.append([vPositive_count, Positive_count, Negative_count, vNegative_count, "neutral"])


    return feature_vector


def classify_naive_bayes(train_X, train_Y, test_X):

    print("Classifying using Gaussian Naive Bayes ...")

    gnb = GaussianNB()
    yHat = gnb.fit(train_X,train_Y).predict(test_X)

    return yHat

def classify_XGBoost(train_X, train_Y, test_X):

    print("Classifying using XGBoost ...")

    xgb = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500,min_samples_leaf=50,max_depth=8,max_features='sqrt',subsample=0.8,random_state=10)
    yHat = xgb.fit(train_X, train_Y).predict(test_X)

    return yHat


def classify_svm(train_X, train_Y, test_X):

    print("Classifying using Support Vector Machine ...")

    clf = SVC()
    clf.fit(train_X,train_Y)
    yHat = clf.predict(test_X)

    return yHat


def classify_maxEnt(train_X, train_Y, test_X):

    print("Classifying using Maximum Entropy ...")
    maxEnt = LogisticRegressionCV()
    maxEnt.fit(train_X, train_Y)
    yHat = maxEnt.predict(test_X)

    return yHat

def classify_naive_bayes_twitter(train_X, train_Y, test_X, test_Y):

    print("Classifying using Gaussian Naive Bayes ...")
    gnb = GaussianNB()
    yHat = gnb.fit(train_X,train_Y).predict(test_X)

    conf_mat = confusion_matrix(test_Y,yHat)
    print(conf_mat)
    Accuracy = (sum(conf_mat.diagonal())) / np.sum(conf_mat)
    print("Accuray: ", Accuracy)
    evaluate_classifier(conf_mat)

def classify_XGBoost_twitter(train_X, train_Y, test_X, test_Y):

    print("Classifying using XGBoost ...")
    xgb = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500,min_samples_leaf=50,max_depth=8,max_features='sqrt',subsample=0.8,random_state=10)
    yHat = xgb.fit(train_X,train_Y).predict(test_X)

    conf_mat = confusion_matrix(test_Y,yHat)
    print(conf_mat)
    Accuracy = (sum(conf_mat.diagonal())) / np.sum(conf_mat)
    print("Accuray: ", Accuracy)
    evaluate_classifier(conf_mat)


def classify_svm_twitter(train_X, train_Y, test_X, test_Y):

    print("Classifying using Support Vector Machine ...")
    clf = SVC()
    clf.fit(train_X,train_Y)
    yHat = clf.predict(test_X)
    conf_mat = confusion_matrix(test_Y,yHat)
    print(conf_mat)
    Accuracy = (sum(conf_mat.diagonal())) / np.sum(conf_mat)
    print("Accuracy: ", Accuracy)
    evaluate_classifier(conf_mat)

def classify_maxEnt_twitter(train_X, train_Y, test_X, test_Y):

    print("Classifying using Maximum Entropy ...")
    maxEnt = LogisticRegressionCV()
    maxEnt.fit(train_X, train_Y)
    yHat = maxEnt.predict(test_X)
    conf_mat = confusion_matrix(test_Y,yHat)
    print(conf_mat)
    Accuracy = (sum(conf_mat.diagonal())) / np.sum(conf_mat)
    print("Accuracy: ", Accuracy)
    evaluate_classifier(conf_mat)


def classify_twitter_data(file_name):

    test_data = open("/Users/siri123/Desktop/ML-Sentiment-Analysis-of-phones-from-Twitter/Data/oneplus.txt", encoding="utf8").readlines()
    test_data = preprocessing(test_data)
    test_data = FeatureizeTestData(test_data)
    test_data = np.reshape(np.asarray(test_data),newshape=(len(test_data),5))

    data_X_test = test_data[:,:4].astype(int)
    data_Y_test = test_data[:,4]

    print("Classifying", args.DataSetName)
    if args.Algorithm == "all":
        classify_naive_bayes_twitter(data_X, data_Y, data_X_test, data_Y_test)
        classify_svm_twitter(data_X, data_Y, data_X_test, data_Y_test)
        classify_maxEnt_twitter(data_X, data_Y, data_X_test, data_Y_test)
        classify_XGBoost_twitter(data_X, data_Y, data_X_test, data_Y_test)
    elif args.Algorithm == "gnb":
        classify_naive_bayes_twitter(data_X, data_Y, data_X_test, data_Y_test)
    elif args.Algorithm == "svm":
        classify_svm_twitter(data_X, data_Y, data_X_test, data_Y_test)
    elif args.Algorithm == "maxEnt":
        classify_maxEnt_twitter(data_X, data_Y, data_X_test, data_Y_test)
    elif args.Algorithm == "xgb":
        classify_XGBoost_twitter(data_X, data_Y, data_X_test, data_Y_test)



def evaluate_classifier(conf_mat):
    Precision = conf_mat[0,0]/(sum(conf_mat[0]))
    Recall = conf_mat[0,0] / (sum(conf_mat[:,0]))
    F_Measure = (2 * (Precision * Recall))/ (Precision + Recall)

    print("Precision: ",Precision)
    print("Recall: ", Recall)
    print("F-Measure: ", F_Measure)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Sentimental Analysis of phones")
    parser.add_argument("DataSetName", help="Dataset to Classify (rottom oneplus iphone googlepixel redmi)", metavar='dataset')
    parser.add_argument("Algorithm", help="Classification Algorithm to be used (all gnb svm maxEnt xgb)", metavar='algo')
    parser.add_argument("Crossvalidation", help="Using Cross validation (yes/no)", metavar='CV')
    args = parser.parse_args()

    os.chdir('../')
    dirPath = os.getcwd()



    print("Please wait while we Classify your data ...")
    affin_list = generateAffinityList("/Users/siri123/Desktop/ML-Sentiment-Analysis-of-phones-from-Twitter/Data/Affin_Data.txt")


    createDictionaryFromPolarity(affin_list)


    print("Reading your data ...")
    positive_data = open("/Users/siri123/Desktop/ML-Sentiment-Analysis-of-phones-from-Twitter/Data/rt-polarity-pos.txt").readlines()
    print("Preprocessing in progress ...")
    positive_data = preprocessing(positive_data)


    negative_data = open("/Users/siri123/Desktop/ML-Sentiment-Analysis-of-phones-from-Twitter/Data/rt-polarity-neg.txt").readlines()
    negative_data = preprocessing(negative_data)

    print("Generating the Feature Vectors ...")
    positive_sentiment = FeaturizeTrainingData(positive_data, "positive")
    negative_sentiment = FeaturizeTrainingData(negative_data,"negative")
    final_data = positive_sentiment + negative_sentiment
    final_data = np.reshape(np.asarray(final_data),newshape=(len(final_data),5))


    data_X = final_data[:,:4].astype(int)
    data_Y = final_data[:,4]


    print("Training the Classifer according to the data provided ...")
    print("Classifying the Test Data ...")
    print("Evaluation Results will be displayed Shortly ...")

    if args.Crossvalidation == "no" or args.Crossvalidation == "No":
        if args.DataSetName == 'rottom':
            if args.Algorithm == "all":
                yHat = classify_naive_bayes(data_X, data_Y, data_X)
                conf_mat = confusion_matrix(data_Y, yHat)
                print(conf_mat)
                Accuracy = (sum(conf_mat.diagonal())) / np.sum(conf_mat)
                print("Accuracy: ", Accuracy)
                evaluate_classifier(conf_mat)

                yHat = classify_svm(data_X, data_Y, data_X)
                conf_mat = confusion_matrix(data_Y, yHat)
                print(conf_mat)
                Accuracy = (sum(conf_mat.diagonal())) / np.sum(conf_mat)
                print("Accuracy: ", Accuracy)
                evaluate_classifier(conf_mat)

                yHat = classify_maxEnt(data_X, data_Y, data_X)
                conf_mat = confusion_matrix(data_Y, yHat)
                print(conf_mat)
                Accuracy = (sum(conf_mat.diagonal())) / np.sum(conf_mat)
                print("Accuracy: ", Accuracy)
                evaluate_classifier(conf_mat)

                yHat = classify_XGBoost(data_X, data_Y, data_X)
                conf_mat = confusion_matrix(data_Y, yHat)
                print(conf_mat)
                Accuracy = (sum(conf_mat.diagonal())) / np.sum(conf_mat)
                print("Accuracy: ", Accuracy)
                evaluate_classifier(conf_mat)

            elif args.Algorithm == "gnb":
                yHat = classify_naive_bayes(data_X, data_Y, data_X)
                conf_mat = confusion_matrix(data_Y, yHat)
                print(conf_mat)
                Accuracy = (sum(conf_mat.diagonal())) / np.sum(conf_mat)
                print("Accuracy: ", Accuracy)
                evaluate_classifier(conf_mat)
            elif args.Algorithm == "svm":
                yHat = classify_svm(data_X, data_Y, data_X)
                conf_mat = confusion_matrix(data_Y, yHat)
                print(conf_mat)
                Accuracy = (sum(conf_mat.diagonal())) / np.sum(conf_mat)
                print("Accuracy: ", Accuracy)
                evaluate_classifier(conf_mat)
            elif args.Algorithm == "maxEnt":
                yHat = classify_maxEnt(data_X, data_Y, data_X)
                conf_mat = confusion_matrix(data_Y, yHat)
                print(conf_mat)
                Accuracy = (sum(conf_mat.diagonal())) / np.sum(conf_mat)
                print("Accuracy: ", Accuracy)
                evaluate_classifier(conf_mat)
            elif args.Algorithm == "xgb":
                yHat = classify_XGBoost(data_X, data_Y, data_X)
                conf_mat = confusion_matrix(data_Y, yHat)
                print(conf_mat)
                Accuracy = (sum(conf_mat.diagonal())) / np.sum(conf_mat)
                print("Accuracy: ", Accuracy)
                evaluate_classifier(conf_mat)


        elif args.DataSetName == "oneplus":
            classify_twitter_data(file_name="redmi.txt")
        elif args.DataSetName == "iphone":
            classify_twitter_data(file_name="iphone.txt")
        elif args.DataSetName == "googlepixel":
            classify_twitter_data(file_name="googlepixel.txt")
        elif args.DataSetName == "redmi":
            classify_twitter_data(file_name="oneplus.txt")
        else:
            print("ERROR while specifying phone Tweets File, please check the name again")

    if args.Crossvalidation == "yes" or args.Crossvalidation == "Yes":
        cv_kFold = KFold(n=len(data_X), n_folds=10, shuffle=True, random_state=5)
        i = 0
        print("Starting "+str(cv_kFold.n_folds)+" Crossvalidation")
        for train_idx, test_idx in cv_kFold:
            X_train, X_test = np.array([data_X[ele] for ele in train_idx]), np.array([data_X[ele] for ele in test_idx])
            Y_train, Y_test = np.array([data_Y[ele] for ele in train_idx]), np.array([data_Y[ele] for ele in test_idx])

            i+=1
            print("Fold: ",i)
            if args.DataSetName == 'rottom':
                if args.Algorithm == "all":
                    yHat = classify_naive_bayes(X_train, Y_train, X_test)
                    conf_mat = confusion_matrix(Y_test, yHat)
                    print(conf_mat)
                    Accuracy = (sum(conf_mat.diagonal())) / np.sum(conf_mat)
                    print("Accuracy: ", Accuracy)
                    evaluate_classifier(conf_mat)
                    yHat = classify_svm(X_train, Y_train, X_test)
                    conf_mat = confusion_matrix(Y_test, yHat)
                    print(conf_mat)
                    Accuracy = (sum(conf_mat.diagonal())) / np.sum(conf_mat)
                    print("Accuracy: ", Accuracy)
                    evaluate_classifier(conf_mat)
                    yHat = classify_maxEnt(X_train, Y_train, X_test)
                    conf_mat = confusion_matrix(Y_test, yHat)
                    print(conf_mat)
                    Accuracy = (sum(conf_mat.diagonal())) / np.sum(conf_mat)
                    print("Accuracy: ", Accuracy)
                    evaluate_classifier(conf_mat)
                    yHat = classify_XGBoost(X_train, Y_train, X_test)
                    conf_mat = confusion_matrix(Y_test, yHat)
                    print(conf_mat)
                    Accuracy = (sum(conf_mat.diagonal())) / np.sum(conf_mat)
                    print("Accuracy: ", Accuracy)
                    evaluate_classifier(conf_mat)
                elif args.Algorithm == "gnb":
                    yHat = classify_naive_bayes(X_train, Y_train, X_test)
                    conf_mat = confusion_matrix(Y_test, yHat)
                    print(conf_mat)
                    Accuracy = (sum(conf_mat.diagonal())) / np.sum(conf_mat)
                    print("Accuracy: ", Accuracy)
                    evaluate_classifier(conf_mat)
                elif args.Algorithm == "svm":
                    yHat = classify_svm(X_train, Y_train, X_test)
                    conf_mat = confusion_matrix(Y_test, yHat)
                    print(conf_mat)
                    Accuracy = (sum(conf_mat.diagonal())) / np.sum(conf_mat)
                    print("Accuracy: ", Accuracy)
                    evaluate_classifier(conf_mat)
                elif args.Algorithm == "maxEnt":
                    yHat = classify_maxEnt(X_train, Y_train, X_test)
                    conf_mat = confusion_matrix(Y_test, yHat)
                    print(conf_mat)
                    Accuracy = (sum(conf_mat.diagonal())) / np.sum(conf_mat)
                    print("Accuracy: ", Accuracy)
                    evaluate_classifier(conf_mat)
                elif args.Algorithm == "xgb":
                    yHat = classify_XGBoost(X_train, Y_train, X_test)
                    conf_mat = confusion_matrix(Y_test, yHat)
                    print(conf_mat)
                    Accuracy = (sum(conf_mat.diagonal())) / np.sum(conf_mat)
                    print("Accuracy: ", Accuracy)
                    evaluate_classifier(conf_mat)

            elif args.DataSetName == "oneplus":
                classify_twitter_data(file_name="redmi.txt")
            elif args.DataSetName == "iphone":
                classify_twitter_data(file_name="iphone.txt")
            elif args.DataSetName == "googlepixel":
                classify_twitter_data(file_name="googlepixel.txt")
            elif args.DataSetName == "redmi":
                classify_twitter_data(file_name="oneplus.txt")
            else:
                print("ERROR while specifying phone Tweets File, please check the name again")
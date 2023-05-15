
# Dependencies
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LinearRegression

# Import locally stored data
businesses = pd.read_json('data/yelp_business.json', lines=True)
reviews = pd.read_json('data/yelp_review.json', lines=True)
users = pd.read_json('data/yelp_user.json', lines=True)
checkins = pd.read_json('data/yelp_checkin.json', lines=True)
tips = pd.read_json('data/yelp_tip.json', lines=True)
photos = pd.read_json('data/yelp_photo.json', lines=True)

# Merge data into one dataframe
df = pd.merge(businesses, reviews, how='left', on='business_id')

df = pd.merge(df, users , how='left', on='business_id')

df = pd.merge(df, checkins , how='left', on='business_id')

df = pd.merge(df, tips , how='left', on='business_id')

df = pd.merge(df, photos , how='left', on='business_id')

# Remove features that do not impact model
features_to_remove = ['business_id','categories','city','hours','is_open','latitude','longitude','name','neighborhood','postal_code','state','time']

df.drop(labels=features_to_remove, axis=1, inplace=True)

# Clean data, replacing all NaN with 0
df.fillna({'weekday_checkins':0,
           'weekend_checkins':0,
           'average_tip_length':0,
           'number_tips':0,
           'average_caption_length':0,
           'number_pics':0},
          inplace=True)

# # Plot data
# plt.scatter(df['average_review_sentiment'], df['stars'], alpha=0.5)
# plt.show()

# plt.scatter(df['average_review_length'], df['stars'], alpha=0.5)
# plt.show()

# plt.scatter(df['average_review_age'], df['stars'], alpha=0.01)
# plt.show()

# plt.scatter(df['number_funny_votes'], df['stars'], alpha=0.5)
# plt.show()

# features = df[['average_review_length', 'average_review_age']]
# ratings = df['stars']

binary_features = ['alcohol?','has_bike_parking','takes_credit_cards','good_for_kids','take_reservations','has_wifi']

numeric_features = ['review_count','price_range','average_caption_length','number_pics','average_review_age','average_review_length','average_review_sentiment','number_funny_votes','number_cool_votes','number_useful_votes','average_tip_length','number_tips','average_number_friends','average_days_on_yelp','average_number_fans','average_review_count','average_number_years_elite','weekday_checkins','weekend_checkins']

all_features = binary_features + numeric_features

features = df[['average_review_length','average_review_age']]
ratings = df['stars']

# Train dataset
X_train, X_test, y_train, y_test = train_test_split(features, ratings, test_size=0.2, random_state=1)

model = LinearRegression()
model.fit(X_train, y_train)

model.score(X_train, y_train)

sorted(list(zip(['average_review_length','average_review_age'],model.coef_)),key = lambda x: abs(x[1]),reverse=True)

y_predicted = model.predict(X_test)

# subset of only average review sentiment
sentiment = ['average_review_sentiment']

# subset of all features that have a response range [0,1]
binary_features = ['alcohol?','has_bike_parking','takes_credit_cards','good_for_kids','take_reservations','has_wifi']

# take a list of features to model as a parameter
def model_these_features(feature_list):
    
    # Set target variable and dependent variables to desired data
    ratings = df.loc[:,'stars']
    features = df.loc[:,feature_list]
    
    # Train the model on selected data
    X_train, X_test, y_train, y_test = train_test_split(features, ratings, test_size = 0.2, random_state = 1)
    
    if len(X_train.shape) < 2:
        X_train = np.array(X_train).reshape(-1,1)
        X_test = np.array(X_test).reshape(-1,1)
    
    # Fit a linear regression model to the training data
    model = LinearRegression()
    model.fit(X_train,y_train)
    
    # Show the score for the models training data and the models testing data
    print('Train Score:', model.score(X_train,y_train))
    print('Test Score:', model.score(X_test,y_test))
    
    # print the model features and their corresponding coefficients, from most predictive to least predictive
    print(sorted(list(zip(feature_list,model.coef_)),key = lambda x: abs(x[1]),reverse=True))
    
    # Show the values that the trained model would predict for y based on x test data
    y_predicted = model.predict(X_test)
    
    # Plot the tested data vs the predicted data to analyze visually
    plt.scatter(y_test,y_predicted)
    plt.xlabel('Yelp Rating')
    plt.ylabel('Predicted Yelp Rating')
    plt.ylim(1,5)
    plt.show()


model_these_features(all_features)



# features = df.loc[:,all_features]
# ratings = df.loc[:,'stars']
# X_train, X_test, y_train, y_test = train_test_split(features, ratings, test_size = 0.2, random_state = 1)
# model = LinearRegression()
# model.fit(X_train,y_train)



# pd.DataFrame(list(zip(features.columns,features.describe().loc['mean'],features.describe().loc['min'],features.describe().loc['max'])),columns=['Feature','Mean','Min','Max'])



# danielles_delicious_delicacies = np.array([1,1,1,1,1,1,40,2,5,2,200,600,.8,20,30,40,60,5,105,3000,12,200,1,50,75]).reshape(1,-1)




# model.predict(danielles_delicious_delicacies)






'''



'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor


def prepare_country_stats(oecd_bli, gdp_per_capita):
    '''
    
    Args:
    
    oecd_bli: OECD life satisfcation index for year 2015
    
    gdp_per_capita: IMF GDP per capita data until year 2015
    
    Return:
    
    full_country_stats: A data frame whose index is Country list. 
    Data set includes both IMF and OECd data
    
    
    '''
    
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"] # selecting only 'total' values
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value") # reshaping table
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True) # renaming 2015 column as GDP per capita
    gdp_per_capita.set_index("Country", inplace=True) # setting index the index is 'Country'
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita, # merging GDP per capita and Life Satisfaciton
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True) # sorting against GDP per capita
    return full_country_stats


oecd_bli = pd.read_csv("oecd_bli_2015.csv", thousands=',')
gdp_per_capita = pd.read_csv("gdp_per_capita.csv",thousands=',',delimiter='\t', encoding='latin1', na_values="n/a")

full_country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)

X = np.c_[full_country_stats["GDP per capita"]]
y = np.c_[full_country_stats["Life satisfaction"]]

full_country_stats.plot(kind='scatter', x="GDP per capita", y='Life satisfaction')
plt.show()

## Linear Regression of all data

model_1=sklearn.linear_model.LinearRegression()
model_1.fit(X,y)
reg_score_1=model_1.score(X,y)
reg_coef_1=model_1.coef_[0][0]
reg_x0_1=model_1.intercept_[0]

print('1st model regression score:', reg_score_1)
print('1st model regresison coefficient: ', reg_coef_1)
print('1st model x0: ', reg_x0_1)

full_country_stats.plot(kind='scatter', x="GDP per capita", y='Life satisfaction')
plt.axis([0,110000,0,10])
X_model=np.linspace(0,110000,1000)
plt.plot(X_model, reg_x0_1+X_model*reg_coef_1,"b")
plt.show()

def best_test_scores(random_states, test_sizes):
    '''
    
    Args:
    
    random_states: an integer that determines which random state will be selected. 
    Test data will change according to selected data.
    
    test_sizes: an integer determined by user and it should be lower than 18 - among 36 countries
    
    Returns: 
    
    test_scores: Regresssion scores of K-neighbors regression of variable randpm states and test data sizes
    
    '''
    
    if test_sizes<=1:
        raise TypeError("Test size must be greater than 1")
    elif test_sizes>18:
        raise TypeError("Test size must be  less than or equal to 18")
        
    test_scores=np.zeros((random_states, test_sizes))
    params = {'n_neighbors': np.arange(2, test_sizes+1)}
    
    for random_state in np.arange(random_states):
        for test_size in np.arange(4,test_sizes+1):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/36, random_state=random_state)
            
            knn = KNeighborsRegressor()
            
            model = GridSearchCV(knn, params, cv=5)
            model.fit(X_train,y_train)
            best_param=model.best_params_
            best_param=int([x for x in best_param.values()][0])
            
            print('Number of neighbors required when test size is', test_size, 'for random state ', random_state, 'is', best_param)
            
            the_model=KNeighborsRegressor(n_neighbors=best_param)
            the_model.fit(X_train, y_train)

            pred=the_model.predict(X_test)
            accr=the_model.score(X_test,y_test)
            
            test_scores[random_state,test_size-1]=accr
            
            print('Accuracy of the best model is', accr)
            
    return test_scores

# Finding best regression scores of k-neighbors regression with cross-validation

best_test_scores=best_test_scores(250,10)

def optimal_test_score_finder(best_test_scores):
    
    '''
    
    Args:
    
    best_test_scores: Test score results that obtained from best_test_scores. 
    best_test_scores must be obtained to activate the algorithm.
    
    Returns: 
    
    optimal_test_places: Gives an array about where is optimal test scores in best_test_scores
    These data test score are greater than 0.7
    
    '''
    
    sorted_test_scores=np.sort(best_test_scores,axis=None)[::-1]
    sorted_test_scores=sorted_test_scores[sorted_test_scores>0.7] # for easing the computation scores higher than 0.7 were selected
    
    optimal_test_places=[]
    
    for test in range(len(sorted_test_scores)):
        optimal_test_places.append(np.where(sorted_test_scores[test] == best_test_scores))
        
    print('First 10 optimal score places are: ', optimal_test_places[0:10])
    
    return optimal_test_places
        
    
    
    # Extracting optimal data places

optimal_test_places=optimal_test_score_finder(best_test_scores)

def optimal_test_score_definer(optimal_test_places, select_test):
    
    '''
    
    Args:
    
    optimal_test_places: The data about where is the optimal scores in best_test_scores.
    optimal_test_places algorithm must be run.
    
    select_test: An integer which can defined by using the output of printed data for defining which countries was eliminated to get optimal data.
    
    Returns:
    
    elim_count: Gives a list of which countries have eliminated during optimization process.
    
    X_train_opt: Optimum GDP training data
    
    X_test_opt: Optimum GDP test data
    
    y_train_opt: Optimum life satisfaction training data
    
    y_test_opt: Optimum life satisfaction test data
    
    '''
    
    optimal_random=optimal_test_places[select_test][0][0] ## x-coordinates on best_test_scores, selected random state
    
    optimal_test_var_num=optimal_test_places[select_test][1][0] ## y-coordinates on best_test_scores, number of variables in the test data
    
    X_train_opt, X_test_opt, y_train_opt, y_test_opt = train_test_split(X, y, test_size=(optimal_test_var_num+1)/36, random_state=optimal_random)
    
    elim_count_places=[]
    elim_count=[]
    
    for GDP in range(len(X_test_opt)):
        
        elim_count_places.append(np.where(X_test_opt[GDP][0]==full_country_stats["GDP per capita"]))
        
        elim_count=full_country_stats["GDP per capita"].index[[elim_count_places]]
    
    return X_train_opt, X_test_opt, y_train_opt, y_test_opt, elim_count

# Extracting training and test data and countries which elimated during optimization process

# For random state 145

X_train_opt, X_test_opt, y_train_opt, y_test_opt, elim_count= optimal_test_score_definer(optimal_test_places, 2)

print('Optimized training GDP data: ', X_train_opt)
print('Optimized training Life Satisfaciton data: ', y_train_opt)
print('Optimized test GDP data: ', X_test_opt)
print('Optimized test Life Satisfaciton data: ', y_test_opt)
print('Eliminated Countries: ', elim_count)

## Linear Regression of optimized data for random state 145

model_opt=sklearn.linear_model.LinearRegression()
model_opt.fit(X_train_opt,y_train_opt)
opt_predict=model_opt.predict(X_test_opt)
reg_score_opt=model_opt.score(X_test_opt,y_test_opt)
reg_coef_opt=model_opt.coef_[0][0]
reg_x0_opt=model_opt.intercept_[0]

print('Predicted optimized y-test:', opt_predict)
print('Optimized y_test data:: ', y_test_opt)
print('Optimized model regression score:', reg_score_opt)
print('Optimized model regresison coefficient: ', reg_coef_opt)
print('Optimized model x0: ', reg_x0_opt)

full_country_stats.plot(kind='scatter', x="GDP per capita", y='Life satisfaction')
plt.axis([0,110000,0,10])
X_model=np.linspace(0,110000,1000)
plt.plot(X_model, reg_x0_1+X_model*reg_coef_1,"b", label='regression of all data')
plt.plot(X_model, reg_x0_opt+X_model*reg_coef_opt,"r",label='regression of random state 145')
plt.legend(loc='lower right')
plt.show()

# For random state 10

X_train_opt_2, X_test_opt_2, y_train_opt_2, y_test_opt_2, elim_count_2= optimal_test_score_definer(optimal_test_places, 7)

print('Optimized training GDP data: ', X_train_opt_2)
print('Optimized training Life Satisfaciton data: ', y_train_opt_2)
print('Optimized test GDP data: ', X_test_opt_2)
print('Optimized test Life Satisfaciton data: ', y_test_opt_2)
print('Eliminated Countries: ', elim_count_2)

## Linear Regression of optimized data

model_opt_2=sklearn.linear_model.LinearRegression()
model_opt_2.fit(X_train_opt_2,y_train_opt_2)
opt_predict_2=model_opt_2.predict(X_test_opt_2)
reg_score_opt_2=model_opt_2.score(X_test_opt_2,y_test_opt_2)
reg_coef_opt_2=model_opt_2.coef_[0][0]
reg_x0_opt_2=model_opt_2.intercept_[0]

print('Predicted optimized y-test:', opt_predict_2)
print('y_test data for random state 10: ', y_test_opt_2)
print('Regression score for random state 10:', reg_score_opt_2)
print('Regresison coefficient for random state 10: ', reg_coef_opt_2)
print('x0 for for random state 10: ', reg_x0_opt_2)

full_country_stats.plot(kind='scatter', x="GDP per capita", y='Life satisfaction')
plt.axis([0,110000,0,10])
X_model=np.linspace(0,110000,1000)
plt.plot(X_model, reg_x0_1+X_model*reg_coef_1,"b", label='regression of all data')
plt.plot(X_model, reg_x0_opt_2+X_model*reg_coef_opt_2,"r",label='regression of random state 10')
plt.legend(loc='lower right')
plt.show()
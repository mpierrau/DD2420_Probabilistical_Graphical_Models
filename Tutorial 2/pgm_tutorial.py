'''

This code template belongs to
"
PGM-TUTORIAL: EVALUATION OF THE 
PGMPY MODULE FOR PYTHON ON BAYESIAN PGMS
"

Created: Summer 2017
@author: miker@kth.se

Refer to https://github.com/pgmpy/pgmpy
for the installation of the pgmpy module

See http://pgmpy.org/models.html#module-pgmpy.models.BayesianModel
for examples on fitting data

See http://pgmpy.org/inference.html
for examples on inference

'''

def separator():
    input('Enter to continue')
    print('-'*70, '\n')
    
# Generally used stuff from pgmpy and others:
import math
import random
import numpy as np
import pandas as pd
from pgmpy.inference import VariableElimination
from pgmpy.inference import BeliefPropagation
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator, K2Score

# Specific imports for the tutorial
import pgm_tutorial_data
from pgm_tutorial_data import ratio, get_random_partition

RAW_DATA = pgm_tutorial_data.RAW_DATA
FEATURES = [f for f in RAW_DATA]

separator()

'''
# End of Task 1

'''


# Task 2 ------------ Probability queries (inference)

data = pd.DataFrame(data=RAW_DATA)
model = BayesianModel([('delay', 'age'),
                       ('delay', 'gender'),
                       ('delay', 'avg_mat'),
                       ('delay', 'avg_cs')])
model.fit(data) # Uses the default ML-estimation

STATE_NAMES = model.cpds[0].state_names
print('State names:')
for s in STATE_NAMES:
    print(s, STATE_NAMES[s])

ve = VariableElimination(model)

print("Results using ve.query(delay given age) \n")

q = ve.query(variables = ['age'], evidence = {'delay': '0'})
print(q)

print("Results using ratio function")

for age in STATE_NAMES['age']:
    print('age : ' + age + '\n')
    for delay in STATE_NAMES['delay']:
        print('delay : ' + delay + '\n')
        print(ratio(data, lambda t: t['age']==age, lambda t: t['delay']==delay))


separator()

q=ve.map_query(variables=None,evidence=None)
print("MAP-query\n")
print(q)

mm=ve.max_marginal(variables=None,evidence=None)
print("Max-Marginal query\n")
print(mm)

mm2=ve.map_query(variables=['age'],evidence={'delay':'0'})

# End of Task 2


# Task 3 ------------ Reversed PGM

data = pd.DataFrame(data=RAW_DATA)
model = BayesianModel([('age', 'delay'),
                       ('gender', 'delay'),
                       ('avg_mat', 'delay'),
                       ('avg_cs', 'delay')])
model.fit(data) # Uses the default ML-estimation

print('Data size : ' + (str) (np.shape(data)))

STATE_NAMES = model.cpds[3].state_names

print('State names:')
for s in STATE_NAMES:
    print(s, STATE_NAMES[s])

vals = model.cpds[3].values

print('np.shape(vals)' + (str) (np.shape(vals)))
zero_count = 0
tot_count = 0

for i in range(np.shape(vals)[0]):
    print("i = %d" % i)
    for j in range(np.shape(vals)[1]):
        for k in range(np.shape(vals)[2]):
            for l in range(np.shape(vals)[3]):
                for m in range(np.shape(vals)[4]):
                    tot_count += 1
                    no = len(data[(data.age == STATE_NAMES['age'][j])][(data.avg_cs == STATE_NAMES['avg_cs'][k])][(data.avg_mat == STATE_NAMES['avg_mat'][l])][(data.delay == STATE_NAMES['delay'][i])][(data.gender == STATE_NAMES['gender'][m])])
                    
                    if(no == 0):
                        print(no)
                        zero_count+=1

print(zero_count)
print(tot_count)

separator()

#Task 3.5

ve = VariableElimination(model)
marg = ve.query(variables=['delay'])
print(marg)

relfreq = np.zeros(np.size(STATE_NAMES['delay']))
for i in range(len(STATE_NAMES['delay'])):
    relfreq[i] = ratio(data, lambda t: t['delay']==STATE_NAMES['delay'][i])
    print('delay : ' + STATE_NAMES['delay'][i] + '\n')
    print(relfreq[i])



# End of Task 3


# Task 4 ------------ Comparing accuracy of PGM models

from scipy.stats import entropy

data = pd.DataFrame(data=RAW_DATA)

model1 = BayesianModel([('delay', 'age'),
                       ('delay', 'gender'),
                       ('delay', 'avg_mat'),
                       ('delay', 'avg_cs')])

model2 = BayesianModel([('age', 'delay'),
                        ('gender', 'delay'),
                        ('avg_mat', 'delay'),
                        ('avg_cs', 'delay')])

models = [model1, model2]

[m.fit(data) for m in models] # ML-fit

STATE_NAMES = model2.cpds[3].state_names
print('\nState names:')
for s in STATE_NAMES:
    print(s, STATE_NAMES[s])

S = STATE_NAMES

VARIABLES = list(S.keys())

def random_query(variables, target):
    # Helper function, generates random evidence query
    n = random.randrange(1, len(variables)+1)
    evidence = {v: random.choice(S[v]) for v in random.sample(variables, n)}
    if target in evidence: del evidence[target]
    return (target, evidence)

queries = []
for target in ['delay']:
    variables = [v for v in VARIABLES if v != target]
    queries.extend([random_query(variables, target) for _ in range(1000)])

divs = []
# divs will be filled with lists on the form
# [query, distr. in data, distr. model 1, div. model 1, distr. model 2, div. model 2]
for v, e in queries:
        # Relative frequencies, compared below
        rf = [ratio(RAW_DATA, lambda t: t[v]==s,
                    lambda t:all(t[k] == w for k,w in e.items())) \
              for s in S[v]]
        # Special treatment for missing samples
        #### if sum(rf) == 0: rf = [1/len(rf)]*len(rf) # Commented out on purpose

        #print(len(divs), '-'*20)
        #print('Query:', v, 'given', e)
        #print('rf: ', rf)
         
        div = [(v, e), rf]
        for m in models:
            #print('\nModel:', m.edges())
            ve = VariableElimination(m)
            q = ve.query(variables = [v], evidence = e, show_progress=False)
            div.extend([q.values, entropy(q.values,rf)])
            #print('PGM:', q.values, ', Divergence:', div[-1])
        divs.append(div)

divs2 = [r for r in divs if math.isfinite(r[3]) and math.isfinite(r[5])]

# Task 4.2 - 4.3
# What is the meaning of what is printed below?

for n in range(1,5):
    print([
        len([r for r in divs2 if len(r[0][1])==n]),
        len([r for r in divs2 if len(r[0][1])==n and r[3] < r[5]]),
        len([r for r in divs2 if len(r[0][1])==n and r[3] < r[5]])/len([r for r in divs2 if len(r[0][1])==n]),
        len([r for r in divs2 if len(r[0][1])==n and r[3] > r[5]]),
        len([r for r in divs2 if len(r[0][1])==n and r[3] > r[5]])/len([r for r in divs2 if len(r[0][1])==n]),
        sum(r[3] for r in divs2 if len(r[0][1])==n),
        sum(r[5] for r in divs2 if len(r[0][1])==n),
        len([r for r in divs if len(r[0][1])==n and \
                not(math.isfinite(r[3]) and math.isfinite(r[5]))])])

# Task 4.4

queries = []
for target in ['age']:
    variables = [v for v in VARIABLES if v != target]
    queries.extend([random_query(variables, target) for _ in range(1000)])

divs = []
# divs will be filled with lists on the form
# [query, distr. in data, distr. model 1, div. model 1, distr. model 2, div. model 2]
for v, e in queries:
        # Relative frequencies, compared below
        rf = [ratio(RAW_DATA, lambda t: t[v]==s,
                    lambda t:all(t[k] == w for k,w in e.items())) \
              for s in S[v]]
        # Special treatment for missing samples
        #### if sum(rf) == 0: rf = [1/len(rf)]*len(rf) # Commented out on purpose

        #print(len(divs), '-'*20)
        #print('Query:', v, 'given', e)
        #print('rf: ', rf)
         
        div = [(v, e), rf]
        for m in models:
            #print('\nModel:', m.edges())
            ve = VariableElimination(m)
            q = ve.query(variables = [v], evidence = e, show_progress=False)
            div.extend([q.values, entropy(q.values,rf)])
            #print('PGM:', q.values, ', Divergence:', div[-1])
        divs.append(div)

divs2 = [r for r in divs if math.isfinite(r[3]) and math.isfinite(r[5])]


for n in range(1,5):
    print([
        #len([r for r in divs2 if len(r[0][1])==n]),
        #len([r for r in divs2 if len(r[0][1])==n and r[3] < r[5]]),
        len([r for r in divs2 if len(r[0][1])==n and r[3] < r[5]])/len([r for r in divs2 if len(r[0][1])==n]),
        #len([r for r in divs2 if len(r[0][1])==n and r[3] > r[5]]),
        len([r for r in divs2 if len(r[0][1])==n and r[3] > r[5]])/len([r for r in divs2 if len(r[0][1])==n]),
        sum(r[3] for r in divs2 if len(r[0][1])==n),
        sum(r[5] for r in divs2 if len(r[0][1])==n),
        len([r for r in divs if len(r[0][1])==n and \
                not(math.isfinite(r[3]) and math.isfinite(r[5]))])])

# first row gives number of queries with 2 evidence variables given (after sorting out inf and nans)
# second row gives number of queries with 2 evidence variables given and for which div(model1) < div(model2) (after sorting out inf and nans)
# third row gives number of queries with 2 evidence variables given and for which div(model1) > div(model2) (after sorting out inf and nans)
# fourth row gives number of queries that had 2 evidence variables given and non-finite divergences for either of the models
# fifth row gives the sum of all divergences between distribution from model1 and rf
# sixth row gives the sum of all divergences between distribution from model2 and rf

# Task 4.5
# KEEP GOING HERE

queries = []
for target in ['delay', 'age']:
    variables = [v for v in VARIABLES if v != target]
    queries.extend([random_query(variables, target) for _ in range(500)])

divs3 = []
# divs will be filled with lists on the form
# [query, distr. in data, distr. model 1, div. model 1, distr. model 2, div. model 2]
for v, e in queries:
        # Relative frequencies, compared below
        rf = [ratio(RAW_DATA, lambda t: t[v]==s,
                    lambda t:all(t[k] == w for k,w in e.items())) \
              for s in S[v]]
        # Special treatment for missing samples
        #### if sum(rf) == 0: rf = [1/len(rf)]*len(rf) # Commented out on purpose

        #print(len(divs), '-'*20)
        #print('Query:', v, 'given', e)
        #print('rf: ', rf)
         
        div = [(v, e), rf]
        for m in models:
            #print('\nModel:', m.edges())
            ve = VariableElimination(m)
            q = ve.query(variables = [v], evidence = e, show_progress=False)
            div.extend([q.values, entropy(q.values,rf)])
            #print('PGM:', q.values, ', Divergence:', div[-1])
        divs3.append(div)

divs4 = [r for r in divs3 if math.isfinite(r[3]) and math.isfinite(r[5])]

# Task 4.2
# What is the meaning of what is printed below?

for n in range(1,5):
    print([len([r for r in divs4 if len(r[0][1])==n]),
    len([r for r in divs4 if len(r[0][1])==n and r[3] < r[5]])/len([r for r in divs4 if len(r[0][1])==n]),
    len([r for r in divs4 if len(r[0][1])==n and r[3] > r[5]])/len([r for r in divs4 if len(r[0][1])==n]),
    sum(r[3] for r in divs4 if len(r[0][1])==n),
    sum(r[5] for r in divs4 if len(r[0][1])==n),
    len([r for r in divs3 if len(r[0][1])==n and \
            not(math.isfinite(r[3]) and math.isfinite(r[5]))])])

# The following is required for working with same data in next task:
import pickle
f = open('data.pickle', 'wb')
pickle.dump(divs4, f)
f.close()

separator()

# End of Task 4



# Task 5 ------------ Checking for overfitting

from scipy.stats import entropy

data = pd.DataFrame(data=RAW_DATA)

model1 = BayesianModel([('delay', 'age'),
                       ('delay', 'gender'),
                       ('delay', 'avg_mat'),
                       ('delay', 'avg_cs')])

model2 = BayesianModel([('age', 'delay'),
                        ('gender', 'delay'),
                        ('avg_mat', 'delay'),
                        ('avg_cs', 'delay')])

models = [model1, model2]

[m.fit(data) for m in models] # ML-fit

STATE_NAMES = model2.cpds[3].state_names
print('\nState names:')
for s in STATE_NAMES:
    print(s, STATE_NAMES[s])

S = STATE_NAMES

# Assumes you pickled data from previous task
import pickle
divs_in = pickle.load(open('data.pickle', 'rb'))

divs = []
k_fold = 5
for k in range(k_fold):
    # Dividing data into 75% training, 25% validation.
    # Change the seed to something of your choice:
    seed = '19910806' + str(k)
    raw_data1, raw_data2 = get_random_partition(0.75, seed)
    training_data = pd.DataFrame(data=raw_data1)
    # refit with training data
    [m.remove_cpds(*m.cpds) for m in models] # Gets rid of warnings
    [m.fit(training_data) for m in models]
    for i, div in enumerate(divs_in):
        print(len(divs_in)*k + i,'/', len(divs_in)*k_fold)
        div = div[:] # Make a copy for technical reasons
        try:
            v, e = div[0]
            # Relative frequencies from validation data, compared below
            rf = [ratio(raw_data2, lambda t: t[v]==s,
                        lambda t:all(t[key] == w for key,w in e.items())) \
                  for s in S[v]]
            for m in models:
                #print('\nModel:', m.edges())
                ve = VariableElimination(m)
                q = ve.query(variables = [v], evidence = e, show_progress=False)
                div.append(entropy(q.values, rf))
                #print('PGM:', q[v].values, ', Divergence:', div[-1])
            divs.append(div)
        except IndexError:
            print('fail')

# Filter out the failures
divs2 = [d for d in divs if len(d) == 8]

# Modify the following lines according to your needs.
# Perhaps turn it into a loop as well.

for n in range(1,5):
    print([len([r for r in divs2 if len(r[0][1])==n]),
    len([r for r in divs2 if len(r[0][1])==n and r[3] < r[5]]),
        len([r for r in divs2 if len(r[0][1])==n and r[3] < r[5]])/len([r for r in divs2 if len(r[0][1])==n]),
        len([r for r in divs2 if len(r[0][1])==n and r[3] > r[5]]),
        len([r for r in divs2 if len(r[0][1])==n and r[3] > r[5]])/len([r for r in divs2 if len(r[0][1])==n]),
        len([r for r in divs2 if len(r[0][1])==n and r[-2] < r[-1]]),
        len([r for r in divs2 if len(r[0][1])==n and r[-2] < r[-1]])/len([r for r in divs2 if len(r[0][1])==n and \
            (math.isfinite(r[-2]) and math.isfinite(r[-1]))]),
        len([r for r in divs2 if len(r[0][1])==n and r[-2] > r[-1]]),
        len([r for r in divs2 if len(r[0][1])==n and r[-2] > r[-1]])/len([r for r in divs2 if len(r[0][1])==n and \
            (math.isfinite(r[-2]) and math.isfinite(r[-1]))])])


separator()

# Row 1: How many cases with n number of evidence variables observed
# Row 2: How many cases where model1 performs better than model2 (trained on full data)
# Row 3:  How many cases where model2 performs better than model1 (trained on full data)
# Row 4: How many cases where model1 performs better than model2 (trained on 75% and validated on 25% of data)
# Row 5: How many cases where model1 performs better than model2 (trained on 75% and validated on 25% of data)
# End of Task 5
'''



'''

# Task 6 ------------ Finding a better structure

data = pd.DataFrame(data=RAW_DATA)

model1 = BayesianModel([('delay', 'age'),
                       ('delay', 'gender'),
                       ('delay', 'avg_mat'),
                       ('delay', 'avg_cs')])

model2 = BayesianModel([('age', 'delay'),
                        ('gender', 'delay'),
                        ('avg_mat', 'delay'),
                        ('avg_cs', 'delay')])

models = [model1, model2]

[m.fit(data) for m in models] # ML-fit

STATE_NAMES = model1.cpds[0].state_names
print('\nState names:')
for s in STATE_NAMES:
    print(s, STATE_NAMES[s])

# Information for the curious:
# Structure-scores: http://pgmpy.org/estimators.html#structure-score
# K2-score: for instance http://www.lx.it.pt/~asmc/pub/talks/09-TA/ta_pres.pdf
# Additive smoothing and pseudocount: https://en.wikipedia.org/wiki/Additive_smoothing
# Scoring functions: https://www.cs.helsinki.fi/u/bmmalone/probabilistic-models-spring-2014/ScoringFunctions.pdf
k2 = K2Score(data)
print('Structure scores:', [k2.score(m) for m in models])

separator()

print('\n\nExhaustive structure search based on structure scores:')

from pgmpy.estimators import ExhaustiveSearch
from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import BicScore

# Warning: Doing exhaustive search on a PGM with all 5 variables
# takes more time than you should have to wait. Hence
# re-fit the models to data where some variable(s) has been removed
# for this assignement.
raw_data2 = {'age': data['age'],
             'avg_cs': data['avg_cs'],
             'avg_mat': data['avg_mat'],
             'delay': data['delay'], # Don't comment out this one
             'gender': data['gender'],
             }

data2 = pd.DataFrame(data=raw_data2)

import time
t0 = time.time()
# Uncomment below to perform exhaustive search
searcher = ExhaustiveSearch(data2, scoring_method=K2Score(data2))
search = searcher.all_scores()
print('time:', time.time() - t0)

# Uncomment for printout:
#for score, model in search:
#    print("{0}        {1}".format(score, model.edges()))

separator()

hcs = HillClimbSearch(data2,scoring_method=K2Score(data))
model = hcs.estimate()

hcs2 = HillClimbSearch(data2,scoring_method=K2Score(data2))
model2 = hcs2.estimate()

hcs_bic = HillClimbSearch(data,scoring_method=BicScore(data))
model_bic = hcs_bic.estimate()

hcs_bic2 = HillClimbSearch(data2,scoring_method=BicScore(data2))
model_bic2 = hcs_bic2.estimate()


# End of Task 6

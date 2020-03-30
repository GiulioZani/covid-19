# coding: utf-8

# In[60]:


import pandas as pd
import numpy as np
from datetime import datetime, timedelta
# from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
# from scipy.optimize import fsolve
# import matplotlib.pyplot as plt
# import pdb
from scipy.interpolate import UnivariateSpline
import json
from tco import *


# In[49]:


df = pd.read_csv("https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv")
import pdb
#pdb.set_trace()
df = df.loc[:, ['data', 'totale_casi']]
FMT = '%Y-%m-%dT%H:%M:%S'
date = df['data']
df['data'] = date.map(lambda x : (datetime.strptime(x, FMT ) - datetime.strptime("2020-01-01T00:00:00", FMT ) ).days  )
#df['data'] = date.map(lambda x : (datetime.strptime(x) - datetime.strptime("2020-01-01 00:00:00", FMT ) ).days  )


xs = list(df.iloc[:, 0])
ys = list(df.iloc[:, 1])


# In[50]:


def logistic_model(x, a, b, c):
    return c/(1 + np.exp(-a*(x - b)))


def inv_logistic_model(x, a, b, c):
    return (a*b - np.log(c/x - 1))/a


def exponential_model(x, a, b, c):
    return a*np.exp(b*(x-c))


def inv_exponential_model(x, a, b, c):
    return (np.log(x/a)/b + c)


# In[51]:


logi_fit = curve_fit(logistic_model, xs, ys, p0=[1/2, 100, 20000])
errors = [np.sqrt(logi_fit[1][i][i]) for i in [0, 1, 2]]
logistic = lambda x: logistic_model(x, *logi_fit[0])
# zero_cases_x = int(fsolve(lambda x: logistic(x) - int(logi_fit[0][-1]), logi_fit[0][1]))


@with_continuations()
def get_first_element(i=1, self=None):
    return self(i+1) if (logi_fit[0][-1] - logistic(i)) >= 1 else i


zero_cases_x = get_first_element()
# pred_xs = list(range(max(xs), sol))
print(zero_cases_x)


# In[52]:


exp_fit = curve_fit(exponential_model, xs, ys, p0=[1, 0.6, 0])
exponential = lambda x: exponential_model(x, *exp_fit[0])


# In[53]:


xs_range = range(55, zero_cases_x)

logistic_ys = [logistic(x) for x in xs_range]
exponential_ys = [exponential(x) for x in xs_range]


# In[54]:


from_0_xs = list(range(list(xs_range)[-1]))
from_0_logistic_ys = [logistic(x) for x in from_0_xs]
logistic_spline = UnivariateSpline(from_0_xs, from_0_logistic_ys, s=0, k=4)
first_derivative = logistic_spline.derivative(n=1)(xs_range)
second_derivative = logistic_spline.derivative(n=2)(xs_range)
zeros = np.zeros((55, 110))
second_derivative_spline = UnivariateSpline(range(300), logistic_spline.derivative()(range(300)), s=0, k=4).derivative()
root = float(second_derivative_spline.roots())
root_val = logistic_spline([root])[0]
inflection_point = [root, root_val]


# In[55]:


inv_logistic = lambda x: inv_logistic_model(x, *logi_fit[0])
inv_exponential = lambda x: inv_exponential_model(x, *exp_fit[0])
inv_ys_logistic = [inv_logistic(y) for y in ys]
inv_ys_exponential = [inv_exponential(y) for y in ys]


# In[56]:


r_logistic = np.corrcoef(xs, inv_ys_logistic)[0][1]
r_exponential = np.corrcoef(xs, inv_ys_exponential)[0][1]


# In[57]:

'''
plt.figure(2)
plt.scatter(xs, inv_ys_logistic, label=f'Logistic inverse r={round(r_logistic, 4)}')
plt.scatter(xs, inv_ys_exponential, label=f'Exponential inverse r={round(r_exponential, 4)}')
plt.legend()
# plt.show()
plt.close()
'''

# In[58]:


begin = datetime(2020, 1, 1)
end_date = str(datetime.date(begin + timedelta(zero_cases_x)))

result_data = {
    'xs': xs,
    'ys': ys,
    'xs_range': list(xs_range),
    'logistic_ys': logistic_ys,
    'exponential_ys': exponential_ys,
    'zero_cases': {
        'end_date': end_date,
        'x': zero_cases_x
    },
    'inflection_point': inflection_point,
    'logistic_first_derivative_ys': first_derivative.tolist(),
    'logistic_r': r_logistic,
    'exponential_r': r_exponential,
    'logistic_parameters': {
        'a': logi_fit[0][0],
        'b': logi_fit[0][1],
        'c': logi_fit[0][2]
    },
    'exponential_parameters': {
        'a': exp_fit[0][0],
        'b': exp_fit[0][1],
        'c': exp_fit[0][2]
    }
}
print('about to write')
print(result_data)
json.dump(result_data, open('plotting/parameters.json', 'w'))

# In[59]:
'''
#plt.figure(1)
#plt.rc('font', size=14)
fig, ax = plt.subplots()
plt.title('COVID-19')
# Real data
#ax.scatter(xs, ys, label="Real data", color="red")
# Predicted logistic curve
plt.plot(xs, [logistic_model(i, logi_fit[0][0], logi_fit[0][1], logi_fit[0][2]) for i in xs], label="Logistic model" )
#ax.plot(xs_range, logistic_ys, label=f"Logistic Model r={round(r_logistic,3)}")
# Predicted exponential curve
#ax.plot(xs_range, exponential_ys, label=f"Exponential Model r={round(r_exponential, 3)}" )
#ax.plot(inflection_point[0], inflection_point[1], 'bx', markersize=12, markeredgewidth=3, label='Inflection Point')
#plt.plot(zero_cases_x, 0, 'gx', markersize=12, markeredgewidth=3, label=f'# New Cases < 1/day after: {end_date}')
ax.annotate(f'# New Cases < 1/day after: {end_date}', xy=(20, 1000), xytext=(80, 10000))
#ax.plot(xs_range, np.array(first_derivative), label='New Cases')
#plt.legend()
plt.xlabel("Days since 1 Jan 2020")
plt.ylabel("Total number of infected people")
plt.ylim((min(ys)*0.9,logi_fit[0][-1]*1.1))
plt.show()
'''


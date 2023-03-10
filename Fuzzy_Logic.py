import pandas as pd
df = pd.read_csv('/home/soft25/Downloads/HomeC.csv') #Read the dataset


df.info() #Information of the dataset
df.shape #Shape of the dataset

df['icon'].unique() 
df['summary'].unique()
df['cloudCover'].unique()

df['summary'].isnull().any()
df.isnull().sum()
#data[data.isnull().sum(axis=1)]
df.shape
df.drop(index = 503910,axis=0,inplace=True)
df.shape


df.columns = ['time','use','gen','House overall','Dishwasher','Furnace 1','Furnace 2','Home office',
              'Fridge','Wine cellar','Garage door','Kitchen 12','Kitchen 14','Kitchen 38',
              'Barn','Well','Microwave','Living room','Solar','temperature','icon','humidity',
              'visibility','summary','apparentTemperature','pressure','windSpeed','cloudCover',
              'windBearing','precipIntensity','dewPoint','precipProbability']
df.columns
df['Furnance'] = df[['Furnace 1','Furnace 2']].sum(axis=1)
df['Kitchen'] = df[['Kitchen 12','Kitchen 14','Kitchen 38']].sum(axis=1)

df.drop(['Furnace 1'],axis=1,inplace=True)
df.drop(['Furnace 2'],axis=1,inplace=True)
df.drop(['Kitchen 12'],axis=1,inplace=True)
df.drop(['Kitchen 38'],axis=1,inplace=True)
df.drop(['Kitchen 14'],axis=1,inplace=True)

print("Shape of the dataset : {}, row of the dataset = {} , Column of the dataset = {}".format(
    df.shape,df.shape[0],df.shape[1]))

import seaborn as sns
import matplotlib.pyplot as plt

sns.distplot(df['use'])
plt.title('Energy usage')
plt.show()

#df['use'] = round(df['use'])


#df['use'] = round(df['use'])
import numpy as np
l_low = 0.0
l_high = 0.4
lt_low =0.5
lt_high =1.2

classes = []
for i in df['use']:
    if l_low <= i <=l_high:
        classes.append(0)
    elif lt_low <= i <=lt_high:
        classes.append(1)
    else:
        classes.append(2)

classes = pd.DataFrame(classes)

df.drop(['use'], axis=1, inplace=True)

df = pd.concat([df, classes], axis=1)

df['use'] = df[0]

df.drop([0], axis=1, inplace=True)

df['temperature'] = round(df['temperature'])

Classes = []

for i in df['temperature']:
    if i in range(-20, 42):
        Classes.append(0)

    elif i in range(42, 62):
        Classes.append(1)

    else:
        Classes.append(2)


Classes = pd.DataFrame(Classes)

df = pd.concat([df, Classes], axis=1)

df.drop(['temperature'], axis=1, inplace=True)

df['temperature'] = df[0]
df.drop([0], axis=1, inplace=True)

sns.distplot(df['humidity'])
plt.title('humidity')
plt.show()

sns.distplot(df['windSpeed'])
plt.title('windSpeed')
plt.show()
df['windSpeed'] = round(df['windSpeed'])
wind = []
for i in df['windSpeed']:
    if i in range(0, 5):
        wind.append(0)
    elif i in range(6, 10):
        wind.append(1)
    else:
        wind.append(2)

wind = pd.DataFrame(wind)

df = pd.concat((df, wind), axis=1)

df['wind'] = df[0]
df.drop([0], axis=1, inplace=True)

humi = []
for i in df['humidity']:
    if i < 0.6:
        humi.append(0)
    elif i > 0.8:
        humi.append(2)
    else:
        humi.append(1)

humi = pd.DataFrame(humi)

df = pd.concat((df, humi), axis=1)
df['humi'] = df[0]
df.drop([0], axis=1, inplace=True)

df = df[['use', 'temperature', 'wind', 'humi']]

import numpy as np # linear algebra
import skfuzzy as fuzz
from skfuzzy import control as ctrl

temperature = ctrl.Antecedent(df['temperature'],'temperature')
wind = ctrl.Antecedent(df['wind'], 'wind')
humi = ctrl.Antecedent(df['humi'], "humi")

use = ctrl.Consequent(df['use'], "use")

temperature.automf(3)
wind.automf(3)
humi.automf(3)

use['low'] = fuzz.trimf(use.universe, [0, 0, 0])
use['mediam'] = fuzz.trimf(use.universe, [0, 0, 1])
use['high'] = fuzz.trimf(use.universe, [0, 1, 2])
#use['highest'] = fuzz.trimf(use.universe, [1, 2, 3])

temperature.view()
wind.view()
humi.view()
use.view()

#Low
rule1 = ctrl.Rule(temperature['poor'] | wind['poor'] | humi['poor'], use['low'])
#mediam
rule2 = ctrl.Rule(temperature['average'] | wind['average'] | humi['average'], use['lowest'])
#high
rule3 = ctrl.Rule(temperature['good'] | wind['good'] | humi['good'], use['high'])
#highest
#rule4 = ctrl.Rule(temperature['average'] | wind['poor'] | humi['good'], use['highest'])

rule1.view()
rule2.view()
rule3.view()
#rule4.view()

chance_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])

chances = ctrl.ControlSystemSimulation(chance_ctrl)

chances.input["temperature"] = 0
chances.input["wind"] = 1
chances.input['humi'] = 2

chances.compute()

print(chances.output['use'])
use.view(sim=chances)

print('--------------------------------------------------------------------------------------------')


import pandas as pd
df = pd.read_csv('/home/soft25/Downloads/HomeC.csv') #Read the dataset


df.info() #Information of the dataset
df.shape #Shape of the dataset

df['icon'].unique() 
df['summary'].unique()
df['cloudCover'].unique()

df['summary'].isnull().any()
df.isnull().sum()
#data[data.isnull().sum(axis=1)]
df.shape
df.drop(index = 503910,axis=0,inplace=True)
df.shape

df.columns = ['time','use','gen','House overall','Dishwasher','Furnace 1','Furnace 2','Home office',
              'Fridge','Wine cellar','Garage door','Kitchen 12','Kitchen 14','Kitchen 38',
              'Barn','Well','Microwave','Living room','Solar','temperature','icon','humidity',
              'visibility','summary','apparentTemperature','pressure','windSpeed','cloudCover',
              'windBearing','precipIntensity','dewPoint','precipProbability']
df.columns
df['Furnance'] = df[['Furnace 1','Furnace 2']].sum(axis=1)
df['Kitchen'] = df[['Kitchen 12','Kitchen 14','Kitchen 38']].sum(axis=1)

df.drop(['Furnace 1'],axis=1,inplace=True)
df.drop(['Furnace 2'],axis=1,inplace=True)
df.drop(['Kitchen 12'],axis=1,inplace=True)
df.drop(['Kitchen 38'],axis=1,inplace=True)
df.drop(['Kitchen 14'],axis=1,inplace=True)

print("Shape of the dataset : {}, row of the dataset = {} , Column of the dataset = {}".format(
    df.shape,df.shape[0],df.shape[1]))

df = df[['temperature','humidity','windSpeed','use']]

import seaborn as sns
import matplotlib.pyplot as plt


sns.distplot(df['temperature'])
plt.title('temperarure')
plt.show()


sns.distplot(df['humidity'])
plt.title('humidity')
plt.show()


sns.distplot(df['windSpeed'])
plt.title('windSpeed')
plt.show()


sns.distplot(df['use'])
plt.title('use')
plt.show()

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

temperature = ctrl.Antecedent(np.arange(-20, 101, 1), 'temperature')
humidity = ctrl.Antecedent(np.arange(0, 1, 0.1), 'humidity')
wind = ctrl.Antecedent(np.arange(0,25,1),'wind')

use = ctrl.Consequent(np.arange(0, 15, 0.5), 'use')

temperature['poor'] = fuzz.trimf(temperature.universe, [-20, 0, 40])
temperature['average'] = fuzz.trimf(temperature.universe, [0, 40, 100])
temperature['good'] = fuzz.trimf(temperature.universe, [40, 100, 100])

humidity['poor'] = fuzz.trimf(humidity.universe, [0, 0, 0.5])
humidity['average'] = fuzz.trimf(humidity.universe, [0, 0.5, 1])
humidity['good'] = fuzz.trimf(humidity.universe, [0.5, 1, 1])

wind['poor'] = fuzz.trimf(wind.universe, [0, 0, 15])
wind['average'] = fuzz.trimf(wind.universe, [0, 15, 25])
wind['good'] = fuzz.trimf(wind.universe, [15, 25, 25])

use['low'] = fuzz.trimf(use.universe, [0, 0, 1])
use['medium'] = fuzz.trimf(use.universe, [0, 1, 2])
use['high'] = fuzz.trimf(use.universe, [1, 2, 2])

temperature.automf(3)
wind.automf(3)
humidity.automf(3)

temperature.view()
humidity.view()
wind.view()
use.view()

rule1 = ctrl.Rule(temperature['poor'] & humidity['poor'] & wind['poor'], use['low'])
rule2 = ctrl.Rule(temperature['poor'] & humidity['poor'] & wind['average'], use['low'])
rule3 = ctrl.Rule(temperature['poor'] & humidity['poor'] & wind['good'], use['low'])

rule4 = ctrl.Rule(temperature['poor'] & humidity['average'] & wind['poor'], use['medium'])
rule5 = ctrl.Rule(temperature['poor'] & humidity['average'] & wind['average'], use['medium'])
rule6 = ctrl.Rule(temperature['poor'] & humidity['average'] & wind['good'], use['medium'])


rule7 = ctrl.Rule(temperature['poor'] & humidity['good']& wind['poor'], use['high'])
rule8 = ctrl.Rule(temperature['poor'] & humidity['good']& wind['average'], use['high'])
rule9 = ctrl.Rule(temperature['poor'] & humidity['good'] &wind['good'], use['high'])


rule10 = ctrl.Rule(temperature['average'] & humidity['poor']& wind['poor'], use['medium'])
rule11 = ctrl.Rule(temperature['average'] & humidity['poor']& wind['average'], use['medium'])
rule12 = ctrl.Rule(temperature['average'] & humidity['poor']& wind['good'], use['medium'])

rule13 = ctrl.Rule(temperature['average'] & humidity['average']& wind['poor'], use['medium'])
rule14 = ctrl.Rule(temperature['average'] & humidity['average']& wind['average'], use['medium'])
rule15 = ctrl.Rule(temperature['average'] & humidity['average']& wind['good'], use['medium'])

rule16 = ctrl.Rule(temperature['average'] & humidity['good']& wind['poor'], use['high'])
rule17 = ctrl.Rule(temperature['average'] & humidity['good']& wind['average'], use['high'])
rule18 = ctrl.Rule(temperature['average'] & humidity['good']& wind['good'], use['high'])

rule19 = ctrl.Rule(temperature['good'] & humidity['poor']& wind['poor'], use['medium'])
rule20 = ctrl.Rule(temperature['good'] & humidity['poor']& wind['average'], use['medium'])
rule21 = ctrl.Rule(temperature['good'] & humidity['poor']& wind['good'], use['medium'])

rule22 = ctrl.Rule(temperature['good'] & humidity['average']& wind['poor'], use['high'])
rule23 = ctrl.Rule(temperature['good'] & humidity['average']& wind['average'], use['high'])
rule24 = ctrl.Rule(temperature['good'] & humidity['average']& wind['good'], use['high'])

rule25 = ctrl.Rule(temperature['good'] & humidity['good']& wind['poor'], use['high'])
rule26 = ctrl.Rule(temperature['good'] & humidity['good']& wind['average'], use['high'])
rule27 = ctrl.Rule(temperature['good'] & humidity['good']& wind['good'], use['high'])

chance_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10,
                               rule11, rule12, rule13,rule14,rule15,rule16,rule17,rule18,rule19,
                               rule20,rule21,rule22,rule23,rule24,rule25,rule26,rule27])


chances = ctrl.ControlSystemSimulation(chance_ctrl)

chances.input["temperature"] = 100
chances.input['humidity'] = 1
chances.input['wind'] = 25
chances.compute()

print(chances.output['use'])
use.view(sim=chances)

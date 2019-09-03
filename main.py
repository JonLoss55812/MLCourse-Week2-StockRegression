from numpy import *

def run(): 
  points = genfromtext('stockdata.csv', delimiter=',')
  #hyperparameter
  learning_rate = 0.0001
  #y = mx+b
  inital_b = 0
  initial_m = 0
  num_iteration = 1000
  #feed into model runner, seymor
  [b,m] = gradient_decent_runner(points,inital_b,initial_m,num_iteration)
  print(b)
  print(m)
  
  
if__name__= '__main__';
  run()

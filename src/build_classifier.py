import numpy as np
import pandas as pd
import scipy.stats
import copy
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import pickle
import copy
import scipy.stats
import logging
import sys
import os
from os import path
from tweaker import Tweak

# format
log_format = logging.Formatter("%(asctime)s [%(levelname)8s] %(message)s")
# level
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# handler
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setFormatter(log_format)
logger.addHandler(stdout_handler)

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    logger.info('load iris data')
    iris = datasets.load_iris()

    x_arr = iris['data']
    mean_x = x_arr.mean(axis=0)
    std_x = x_arr.std(axis=0)
    x_arr = scipy.stats.zscore(x_arr)
    y_arr = iris['target']

    logger.info('learn RandomForestClassifier')
    rfc = RandomForestClassifier(random_state=40)
    rfc.fit(x_arr, y_arr)

    logger.info('save learned RandomForestClassifier')
    pickle.dump(rfc,open(path.join(parent_dir,'output','rfc.pickle'),"wb"))

    logger.info('Tweaking')
    tweaker = Tweak()
    target = 2
    np.random.seed(40)
    print("origin feature is {}, origin label is {}".format(x_arr[1],y_arr[1]))
    print("target label is {}".format(target))
    output = tweaker.feature_tweaking(rfc,x_arr[0],[0,1,2],target,0.3,tweaker.cos_sim)
    print("tweaked feature is {}, tweaked label is {}".format(output,rfc.predict(output.reshape(1,-1))))

if __name__ == '__main__':
    main()


# coding: utf-8

# here we run some baselines to understand the level of performance to beat
# we use nested CV and grid search to get realistic performance estimates

# In[2]:


from netCDF4 import Dataset
from collections import defaultdict, namedtuple
import datetime
import json
import numpy as np
import itertools
import base64
import seaborn as sns
sns.set()
import pandas as pd
import hashlib
import sympy as sp
import matplotlib.pyplot as plt
import math
import pickle
import base64
import time
import os
import pickle
from json import JSONEncoder
import csv
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
from sklearn.linear_model import Ridge
from sklearn.utils import shuffle
from sklearn import metrics
from scipy.optimize import fmin_cg, fmin_ncg
from scipy import stats
from IPython.display import clear_output
import xgboost as xgb


# In[3]:


def read_df(dframe_path='data/cabauw/processed-full-log.csv.gz'):
    try:
        df = pd.read_csv(dframe_path, na_values='--')
    except UnicodeDecodeError:
        df = pd.read_csv(dframe_path, na_values='--', compression='gzip')


    df = df[(df.ustar > 0.1) & (abs(df.H) > 10) & (df.wind > 1)]
    df = df[(df.ds != 201603) & (df.phi_m.notnull())]
    df = df.sort_values(['ds', 'tt'])
    #df = df.dropna()

    return df


# In[4]:


def make_index(dtimes, interval):
    # returns a tuple index_above, index_below
    # index_above[i] is the largest i
    # such that dtimes[index_above[i]] - dtimes[i] < interval
    # index_below[i] is the smallest i
    # such that dtimes[i] - dtimes[index_below[i]] < interval
    # dtimes must be already sorted!
    index_below, index_above = np.zeros(
        (2, len(dtimes)), dtype=np.int
    ) - 1

    for i, x in enumerate(dtimes):
        j = index_below[i - 1] if i > 0 else 0
        while x - dtimes[j] > interval:
            j += 1

        index_below[i] = j
        index_above[j] = i

    last_above = index_above[0]
    for i in range(len(dtimes)):
        if index_above[i] < 0:
            index_above[i] = last_above
        else:
            last_above = index_above[i]
    
    return index_above, index_below


def compute_trend(df, columns, interval=3600):
    df = df.sort_values('datetime')
    for z in df.z.unique():  
        this_level = df[df.z == z]
        index_above, index_below = make_index(this_level.datetime.values, interval)

        for col in columns:
            val_above = this_level[col].values
            val_below = this_level.iloc[index_below][col].values

            time_above = this_level.datetime.values
            time_below = this_level.iloc[index_below].datetime.values

            trend = 3600 * (val_above - val_below) / (time_above - time_below)

            df.loc[df.z == z, col + '_trend'] = trend

    return df, [col + '_trend' for col in columns]


def get_features(df, use_trend, feature_level):
    # get feature names of the corresponding level
    # adding them to df if not already there

    feature_sets = [
        [
            'z', 'wind', 'temp', 'soil_temp',
            'wind_10', 'wind_20', 'wind_40',
            'temp_10', 'temp_20', 'temp_40',
        ],
        ['soilheat'],
        ['netrad'],
        ['rain', 'dewpoint'],
        ['H', 'LE'],
    ]

    if isinstance(feature_level, int):
        features = [
            f for fset in feature_sets[:feature_level]
            for f in fset
        ]
    elif isinstance(feature_level, (list, tuple)):
        features = feature_level
    else:
        raise ValueError('pass list or int')

    if ('wind_40' not in df.columns or
            'wind_20' not in df.columns or
            'wind_10' not in df.columns):

        wind_temp_levels = df.pivot_table(
            values=['wind', 'temp'], columns='z', index=['ds', 'tt']
        ).reset_index()
        wind_temp_levels.columns = [
            '%s_%d' % (a, b) if b else a
            for a, b in wind_temp_levels.columns.values
        ]
        df = df.merge(wind_temp_levels, on=['ds', 'tt'])
        
    if use_trend:
        missing_trend = [
            f for f in features
            if f != 'z' and f + '_trend' not in df.columns
        ]

        if missing_trend:
            df, added_cols = compute_trend(df, missing_trend)
            features.extend(added_cols)

    # remove feature columns with only nulls and rows with any null
    empty_columns = df.isnull().all(axis=0)
    keep_columns = df.columns.isin(features) & ~empty_columns
    missing = df.loc[:, keep_columns].isnull().any(axis=1)
    df = df[~missing]
    features = keep_columns.index.values[keep_columns.values]

    return df, list(features)


def get_train_test(df, features, target, train_idx, test_idx, normalize):
    train_x, train_y = df.iloc[train_idx][features], df.iloc[train_idx][target]
    test_x, test_y = df.iloc[test_idx][features], df.iloc[test_idx][target]

    if normalize:
        mean_x, std_x = train_x.mean(), train_x.std()
        train_x = (train_x - mean_x) / std_x
        test_x = (test_x - mean_x) / std_x

        mean_y, std_y = train_y.mean(), train_y.std()
        train_y = (train_y - mean_y) / std_y
        test_y = (test_y - mean_y) / std_y
    else:
        mean_y, std_y = 0, 1

    return train_x, train_y, test_x, test_y, mean_y, std_y


# lets start with a baseline. data doesnt follow the functions given in the literature, so while I fix it we can change those functions to fit the data. given the definition of $\phi_m$ for $\xi>0$:
# 
# $$
# \phi_m(\xi)=a+b\xi
# $$
# 
# whose derivatives are trivial. for $\xi<0$ we have
# 
# $$
# \phi_m(\xi)=a(1-c^2\xi)^d
# $$
# 
# where we square $c$ to make sure the base of the power is always positive. its derivatives are
# 
# $$
# {\frac{\partial \phi_m}{\partial a}}\rvert_{\xi<0}=(1-c^2\xi)^d
# $$
# 
# $$
# \frac{\partial \phi_m}{\partial c}=-2acd\xi(1-c^2\xi)^{d-1}
# $$
# 
# $$
# \frac{\partial \phi_m}{\partial d}=a(1-c^2\xi)^d\ln(1-c^2\xi)
# $$
# 
# considering the usual least squares with l2 regularization we have the loss function
# 
# $$
# E=\frac{1}{N}\sum_i(\hat\phi_m(\xi_i)-\phi_m(\xi_i,p))^2+\frac{\lambda}{2}\sum_p p^2
# $$
# 
# and its derivative with respect to the parameter $p$
# 
# $$
# \frac{\partial E}{\partial p}=\frac{2}{N}\sum_i\frac{\partial}{\partial p}\phi_m(\xi_i,p)\cdot(\hat\phi_m(\xi_i)-\phi_m(\xi_i,p))+\lambda p
# $$

# In[5]:


class MOSTEstimator:
    ''' estimator for the universal functions in the monin-obukhov similarity theory
        implementing scikit's interface
        
        fitting is done by minimizing the L2 regularized squared error
        via conjugate gradient
    '''
    def __init__(self, regu=0.1, use_hessian=True):
        self.regu = regu
        self.a, self.b, self.c, self.d = (1, 4.8, np.sqrt(19.3), -0.25)
        self.use_hessian = use_hessian
        self.symbols = None

    def _lazy_init_hessian(self):
        # we initialize these functions lazily so that we can pickle
        # this object and send it around before fitting
        if self.use_hessian and self.symbols is None:
            a, b, c, d, x = sp.symbols('a b c d x')
            self.symbols = a, b, c, d

            self._neg_H_fn = self._get_hessian_functions(
                a * sp.Pow(1 - x * c**2, d), x, a, b, c, d
            )
            self._pos_H_fn = self._get_hessian_functions(
                a + b * x, x, a, b, c, d
            )

    @staticmethod
    def _get_hessian_functions(expr, x, *symbols):
        # returns functions computing second-order partial derivatives
        # of expr. keyed by differentiation variables
        return {
            (s1, s2): sp.lambdify(
                (x, *symbols),
                sp.simplify(sp.diff(sp.diff(expr, s1), s2)),
                'numpy'
            ) for s1 in symbols for s2 in symbols
        }

    def get_params(self, deep=True):
        return {'regu': self.regu}

    def set_params(self, regu):
        self.regu = regu
        return self

    @classmethod
    def _compute_phi(cls, zL, a, b, c, d):
        zL = cls._to_vec(zL)
        mask = zL >= 0
        yy = np.zeros(zL.shape)
        yy[mask] = a + b * zL[mask]
        yy[~mask] = a * np.power(1 - c**2 * zL[~mask], d)
        assert all(np.isfinite(zL))
        assert all(np.isfinite(yy)), (a, b, c, d)
        return yy

    @classmethod
    def _compute_phi_prime(cls, zL, a, b, c, d):
        zL = cls._to_vec(zL)
        dpda, dpdb, dpdc, dpdd = np.zeros((4, len(zL)))

        pos, neg = zL >= 0, zL < 0

        dpda[pos] = 1
        dpdb[pos] = zL[pos]

        inner = 1 - c**2 * zL[neg]
        dpda[neg] = np.power(inner, d)
        dpdc[neg] = -2 * zL[neg] * a * c * d * np.power(inner, d - 1)
        dpdd[neg] = a * np.log(inner) * np.power(inner, d)

        return dpda, dpdb, dpdc, dpdd

    def _fmin_hess(self, params, xx, yy, regu):
        self._lazy_init_hessian()

        preds = self._compute_phi(xx, *params)
        xpos_mask = xx >= 0
        hh1 = np.zeros((4, 4))

        for i, s1 in enumerate(self.symbols):
            for j, s2 in enumerate(self.symbols):
                # when xx >= 0 the function is linear
                # its hessian is always 0

                neg = self._neg_H_fn[s1, s2](xx[~xpos_mask], *params)
                hh1[i, j] = np.sum((preds[~xpos_mask] - yy[~xpos_mask]) * neg)

        hh2 = np.zeros((4, len(xx)))
        hh2[:, :] = self._compute_phi_prime(xx, *params)
        
        hess = 2 * ((hh2.dot(hh2.T) + hh1) / len(xx) + regu * np.eye(4))
        return hess

    @staticmethod
    def _fmin_target(params, xx, yy, regu):
        preds = MOSTEstimator._compute_phi(xx, *params)
        err = np.mean((yy - preds)**2) + regu * sum(p**2 for p in params)
        return err

    @staticmethod
    def _fmin_grad(params, xx, yy, regu):
        preds = MOSTEstimator._compute_phi(xx, *params)
        der = MOSTEstimator._compute_phi_prime(xx, *params)

        grads = [
            2 * np.mean((preds - yy) * parpr) + 2 * regu * par
            for par, parpr in zip(params, der)
        ]

        return np.array(grads)

    @staticmethod
    def _to_vec(mat):
        mat = np.array(mat)
        
        # check that multi-dimensional arrays have only one
        # dimension with more than one sample
        # e.g. 1x1x99x1 is fine, 1x2x99x is not
        assert sum(1 for n in mat.shape if n > 1) == 1
        return mat.reshape(-1)
    
    def fit(self, X, y):
        X = self._to_vec(X)
        y = self._to_vec(y)

        if self.use_hessian:
            self.a, self.b, self.c, self.d = fmin_ncg(
                self._fmin_target,
                (self.a, self.b, self.c, self.d),
                self._fmin_grad,
                fhess=self._fmin_hess,
                args=(X, y, self.regu),
                disp=False,
            )
        else:
            self.a, self.b, self.c, self.d = fmin_cg(
                self._fmin_target,
                (self.a, self.b, self.c, self.d),
                self._fmin_grad,
                args=(X, y, self.regu),
                disp=False,
            )
        
        return self

    def predict(self, X):
        return self._compute_phi(X, self.a, self.b, self.c, self.d)

    def score(self, X, y):
        preds = self.predict(X)
        return metrics.mean_squared_error(y, preds)


# In[6]:


class AttributeKFold:
    ''' k-fold cross validator splitting on a particular attribute
        so that all samples with a given value are either in the train or test set

        attribute value for each sample is given in the constructor, so that
        the attribute itself need not be in the features for the model
    '''
    def __init__(self, cv, attr):
        self.cv, self.attr = cv, attr

    def get_n_splits(self, *args, **kwargs):
        return self.cv.get_n_splits(*args, **kwargs)

    def split(self, X, y=None, groups=None):
        vals = self.attr.unique()
        for train_idx, test_idx in self.cv.split(vals):
            train_mask = self.attr.isin(vals[train_idx])
            test_mask = self.attr.isin(vals[test_idx])

            X = np.argwhere(train_mask).reshape(-1)
            y = np.argwhere(test_mask).reshape(-1)
            
            assert np.all(np.isfinite(X))
            assert np.all(np.isfinite(y))
            
            yield X, y


# In[7]:


def test_attributekfold():
    outer_cv = AttributeKFold(KFold(10, shuffle=True), df.ds)
    outer_train, outer_test = np.zeros((2, len(df)))
    for outer_train_idx, outer_test_idx in outer_cv.split(df):

        outer_train[outer_train_idx] += 1
        outer_test[outer_test_idx] += 1

        inner_train, inner_test = np.zeros((2, len(outer_train_idx)))
        inner_cv = AttributeKFold(KFold(5, shuffle=True), df.iloc[outer_train_idx].ds)
        for inner_train_idx, inner_test_idx in inner_cv.split(df.iloc[outer_train_idx]):
            inner_train[inner_train_idx] += 1
            inner_test[inner_test_idx] += 1

        assert all(inner_train == 4)
        assert all(inner_test == 1)

    assert all(outer_train == 9)
    assert all(outer_test == 1)


# In[8]:


def plot_preds(ypred, ytrue):
    minn = max(min(ypred), min(ytrue))
    maxx = min(max(ypred), max(ytrue))
    
    plt.scatter(ytrue, ypred, s=2)
    plt.plot([minn, maxx], [minn, maxx], 'r--')
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.show()


# In[9]:


CVSpec = namedtuple('CVSpec', [
    'model', 'param_distribution', 'features', 'target',
    'inner_cv', 'outer_cv', 'n_iter', 'normalize',
    'inner_seed', 'outer_seed', 'param_seed', 'meta',
    'save_to',
])

CVResult = namedtuple('CVResult', [
    'meta', 'scores', 'test_x', 'test_y', 'y_pred', 'imps'
])


# set default value to None, https://stackoverflow.com/a/18348004/521776
CVSpec.__new__.__defaults__ = (None,) * len(CVSpec._fields)
CVResult.__new__.__defaults__ = (None,) * len(CVResult._fields)


def get_cv_fold(fold, cv_k, seed, attr):
    cv = AttributeKFold(
        KFold(cv_k, shuffle=True, random_state=seed),
        attr
    ).split(attr)
    for _ in range(fold):
        _ = next(cv)
    return next(cv)


def get_train_test(df, features, target, train_idx, test_idx, normalize):
    train_x, train_y = df.iloc[train_idx][features], df.iloc[train_idx][target]
    test_x, test_y = df.iloc[test_idx][features], df.iloc[test_idx][target]

    if normalize:
        mean_x, std_x = train_x.mean(), train_x.std()
        train_x = (train_x - mean_x) / std_x
        test_x = (test_x - mean_x) / std_x

        mean_y, std_y = train_y.mean(), train_y.std()
        train_y = (train_y - mean_y) / std_y
        test_y = (test_y - mean_y) / std_y
    else:
        mean_y, std_y = 0, 1

    return train_x, train_y, test_x, test_y, mean_y, std_y


class EncodeMoreStuff(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


class CachedResults:
    def __init__(self, **kwargs):
        try:
            from hops import hdfs
            fs = hdfs.get_fs()
            self.open_ = fs.open_file
            self.cache_dir = 'hdfs:///Projects/more_stuff/checkpoints/'
        except ImportError:
            self.open_ = open
            self.cache_dir = './dev/checkpoints/'

        self.fname = hashlib.sha256(
            json.dumps(kwargs, cls=EncodeMoreStuff).encode('utf8')
        ).hexdigest()
    
    def get_value(self):
        try:
            with self.open_(self.cache_dir + self.fname, 'rb') as f:
                return pickle.loads(f.read())
        except FileNotFoundError:
            return None

    def set_value(self, val):
        try:
            with self.open_(self.cache_dir + self.fname, 'wb') as f:
                f.write(pickle.dumps(val))
        except IOError:
            pass


def inner_train(df_bcast, model, features, target, params, outer_fold,
                inner_fold, keys, outer_seed, inner_seed, normalize):

    cache = CachedResults(
        model=model.__class__.__name__, features=features, target=target,
        params=params, inner_fold=inner_fold, outer_fold=outer_fold,
        keys=keys, outer_seed=outer_seed, inner_seed=inner_seed, normalize=normalize
    )
    saved = cache.get_value()
    if saved is not None:
        return saved
    
    df = df_bcast.value
    if any(f not in df.columns for f in features):
        print('some features are missing, reloading...')
        df, _ = get_features(read_df(), use_trend=True, feature_level=5)
    assert all(f in df.columns for f in features), (df.columns, features)
    
    outer_train_idx, outer_test_idx = get_cv_fold(
        outer_fold, 10, outer_seed, df.ds
    )
    
    inner_train_idx, inner_test_idx = get_cv_fold(
        inner_fold, 5, inner_seed, df.iloc[outer_train_idx].ds
    )

    train_idx, test_idx = outer_train_idx[inner_train_idx], outer_train_idx[inner_test_idx]
    train_x, train_y, test_x, test_y, mean_y, std_y = get_train_test(
        df, features, target, train_idx, test_idx, normalize
    )

    model = model.set_params(**dict(zip(keys, params)))
    model.fit(train_x, train_y)
    y_pred = model.predict(test_x)
    mse = metrics.mean_squared_error(test_y, y_pred)

    cache.set_value(mse)
    return mse


def outer_train(df_bcast, model, features, target, outer_fold, params,
                keys, outer_seed, normalize):
    cache = CachedResults(
        model=model.__class__.__name__, features=features,
        target=target, params=params, outer_fold=outer_fold,
        keys=keys, outer_seed=outer_seed, normalize=normalize
    )
    saved = cache.get_value()
    if saved is not None:
        return saved

    df = df_bcast.value
    if any(f not in df.columns for f in features):
        print('some features are missing, reloading...')
        df, _ = get_features(read_df(), use_trend=True, feature_level=5)
    assert all(f in df.columns for f in features), (df.columns, features)
    
    train_idx, test_idx = get_cv_fold(
        outer_fold, 10, outer_seed, df.ds
    )

    train_x, train_y, test_x, test_y, mean_y, std_y = get_train_test(
        df, features, target, train_idx, test_idx, normalize
    )

    model = model.set_params(**dict(zip(keys, params)))
    model.fit(train_x, train_y)
    y_pred = model.predict(test_x)
    y_pred = y_pred * std_y + mean_y
    test_y = test_y * std_y + mean_y

    perc_errors = 100 * np.abs((test_y - y_pred) / test_y)
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        importances = []
    
    result = (
        metrics.explained_variance_score(test_y, y_pred),
        metrics.mean_absolute_error(test_y, y_pred),
        metrics.mean_squared_error(test_y, y_pred),
        metrics.median_absolute_error(test_y, y_pred),
        metrics.r2_score(test_y, y_pred),
        np.mean(perc_errors),
        np.median(perc_errors),
    ), importances, ((test_x, test_y, y_pred) if outer_fold == 0 else None)

    cache.set_value(result)
    return result


def finalize_result(meta, results, save_to=None):
    scores, imps, preds = zip(*list(results))
    scores_df = pd.DataFrame(list(scores), columns=[
        'explained_variance_score',
        'mean_absolute_error',
        'mean_squared_error',
        'median_absolute_error',
        'r2_score',
        'mean_abs_percent_error',
        'median_abs_percent_error',
    ])

    test_x, test_y, y_pred = [ps for ps in preds if ps is not None][0]
    cvres = CVResult(
        meta=meta, scores=scores,
        test_x=test_x, test_y=test_y, y_pred=y_pred, imps=imps
    )

    if save_to:
        try:
            from hops import hdfs
            fs = hdfs.get_fs()
            open_ = fs.open_file
            base_dir = 'hdfs:///Projects/more_stuff/'
        except ImportError:
            open_ = open
            base_dir = './data/'

        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)

        fname = '%s/cabauw/results_%s.txt' % (base_dir, save_to)
        with open_(fname, 'w') as f:
            f.write(scores_df.describe().T.to_string())
            f.write('\n\n**raw scores\n\n')
            f.write(scores_df.to_string())
            f.write('\n\n**raw json results\n\n')
            f.write(base64.b64encode(pickle.dumps(cvres)).decode('utf8'))

    return cvres


def nested_cv_spark(df, *cv_specs):
    ''' nested cross-validation performing random search in the inner loop

        optimized for running several nested cv in parallel, each with
        different models and/or grids

        callers must make sure that the dataframe contains all the features
        and target they request
    '''
    df_bcast = sc.broadcast(df)

    result_rdd = None
    for spec in cv_specs:
        # default values
        inner_cv = spec.inner_cv or 10
        outer_cv = spec.outer_cv or 10
        n_iter = spec.n_iter or 10
        normalize = spec.normalize or True
        
        # need to have the same seed in all workers,
        # so that the folds are consistent
        inner_seed = spec.inner_seed or np.random.randint(2**32, dtype=np.uint)
        outer_seed = spec.outer_seed or np.random.randint(2**32, dtype=np.uint)

        # build grid with random parameter values
        rnd = np.random.RandomState(spec.param_seed)
        grid = [{
            par: distr.rvs(random_state=rnd) if hasattr(distr, 'rvs') else np.random.choice(distr)
            for par, distr in spec.param_distribution.items()
        } for _ in range(n_iter)]

        # build list of all grids to try and inner/outer combinations
        grid_vals = [tuple(p.values()) for p in grid]
        cv_vals = list(itertools.product(range(outer_cv), range(inner_cv)))

        results = (sc.parallelize(grid_vals, len(grid_vals))
             .cartesian(sc.parallelize(cv_vals, len(cv_vals)))

             # train and evaluate on inner fold for each outer fold
             # key by (outer fold, parameters)
             .map(lambda x: ((x[1][0], x[0]), inner_train(
                 df_bcast, spec.model, spec.features, spec.target,
                 x[0], x[1][0], x[1][1], tuple(spec.param_distribution.keys()),
                 outer_seed, inner_seed, normalize
             )))

             # for each outer fold/parameters, compute sum of mse
             .reduceByKey(lambda mse1, mse2: mse1 + mse2,
                          numPartitions=outer_cv)

             # for each outer fold, find parameters with best mse
             .map(lambda x: (x[0][0], (x[0][1], x[1])))
             .reduceByKey(lambda x, y: x if x[1] < y[1] else y)

             # for each outer fold, validate using best parameters
             # x is (outer fold, (parameters, mse))
             .map(lambda x: outer_train(
                 df_bcast, spec.model, spec.features, spec.target, x[0], x[1][0],
                 tuple(spec.param_distribution.keys()), outer_seed, spec.normalize
             ))

             # finalize results, optionally saving
             # 
             # we could use a coalesce(1) -> mapPartitions, but this would
             # compute all the outer folds of all specs in the last stage
             # (the one that contains collect), which means that it will only
             # use len(cv_specs) executors to run sum(s.outer_cv for s in cv_specs)
             # tasks. needlessly to say, that sucks
             #
             # by adding a keyBy/groupByKey/map, we force the previos outer cv
             # to be in its own stage, thus there will be one partition for each
             # outer fold of each spec, and we can fully utilize the available executors
             .keyBy(lambda x: 1)
             .groupByKey(outer_cv)
             .map(lambda x: finalize_result(spec.meta, x[1], spec.save_to))

             # with this, we force finalize_result to be in its own stage
             # 
             # since the default scheduler is FIFO, this means we will finalize
             # the results as soon as the outer cv finishes
             # without this, we would need to wait for all the outer cvs to finish
             # and the results will all be finalized together at the end
             # that would also suck
             .coalesce(1)
             .repartition(2))

        # we build the result rdd gradually with unions so that we can proceed
        # to the outer cvs as soon as the spec's inner cv has finished
        # if we built the whole result rdd in one go we would need to wait on
        # all inner cvs from all specs to finish, before being able to start
        # the outer cvs
        if result_rdd is None:
            result_rdd = results
        else:
            result_rdd = result_rdd.union(results)

    cv_results = result_rdd.collect()

    return cv_results


# In[10]:


def nested_cv(df, model, grid, feature_level, target, most_only, n_jobs=-2,
              random_iter=10, normalize=True, use_trend=True, outer_callback=None):
    df, features = get_features(df, use_trend, feature_level)
    if most_only:
        df = df[(df.zL > -2) & (df.zL < 1)]

    outer_cv = AttributeKFold(KFold(10, shuffle=True), df.ds)
    results = []
    for oi, (train_idx, test_idx) in enumerate(outer_cv.split(df.ds)):
        train_x, train_y, test_x, test_y, mean_y, std_y = get_train_test(
            df, features, target, train_idx, test_idx, normalize
        )

        # grid search for best params
        inner_cv = AttributeKFold(KFold(10, shuffle=True), df.iloc[train_idx].ds)
        gs = RandomizedSearchCV(
            model, grid, n_jobs=n_jobs, cv=inner_cv,
            n_iter=random_iter,
            scoring='neg_mean_squared_error',
            verbose=2,
        )
        
        assert np.all(np.isfinite(train_x))
        assert np.all(np.isfinite(train_y))
        gs.fit(train_x, train_y)

        # evaluate on test data
        y_pred = gs.best_estimator_.predict(test_x)
        y_pred = y_pred * std_y + mean_y
        test_y = test_y * std_y + mean_y
        perc_errors = 100 * np.abs((test_y - y_pred) / test_y)

        results.append((
            metrics.explained_variance_score(test_y, y_pred),
            metrics.mean_absolute_error(test_y, y_pred),
            metrics.mean_squared_error(test_y, y_pred),
            metrics.median_absolute_error(test_y, y_pred),
            metrics.r2_score(test_y, y_pred),
            np.mean(perc_errors),
            np.median(perc_errors),
        ))
        
        if outer_callback is not None:
            outer_callback(gs, results[-1], test_x, test_y, y_pred)
        
    clear_output()

    return pd.DataFrame(results, columns=[
        'explained_variance_score',
        'mean_absolute_error',
        'mean_squared_error',
        'median_absolute_error',
        'r2_score',
        'mean_abs_percent_error',
        'median_abs_percent_error',
    ]), (test_x, test_y, y_pred)


# In[11]:


def do_test(df, most_only, spark=True):
    ''' run each model on all features with and without trend
    '''
    if most_only:
        df = df[(df.zL > -2) & (df.zL < 1)]

    ridge_spec = Ridge, {
        'alpha': LogUniform(10, -6, 1)
    }
    
    knn_spec = KNeighborsRegressor, {
        'n_neighbors': IntDistribution(stats.uniform(1, 14)),
        'weights': ['uniform', 'distance'],
        'p': [1, 2],
    }
    
    gbr_spec = GradientBoostingRegressor, {
        'max_depth': IntDistribution(stats.uniform(loc=4, scale=8)),
        'subsample': stats.uniform(loc=0.25, scale=0.75),
        'max_features': stats.uniform(0.1, 0.9),
        'loss': ['lad', 'ls', 'huber'],
        'n_estimators': IntDistribution(LogUniform(10, 1, 4)),
        'learning_rate': LogUniform(10, -4, 0),
        'alpha': stats.uniform(0.01, 0.98),
    }

    specs = []
    for trend in [True, False]:
        for fset in range(1, 6):
            _, features = get_features(df, trend, fset)
            for model_cls, grid in [ridge_spec]: #[ridge_spec, knn_spec, gbr_spec]:
                dtime = datetime.datetime.utcnow().strftime('%Y%m%d-%H%M%S')
                fname = '%s_f%d_%strend_%smost-%s' % (
                    model_cls.__name__, fset,
                    '' if trend else 'no',
                    '' if most_only else 'no', dtime
                )
                
                meta = {
                    'trend': trend,
                    'fset': fset,
                    'model': model_cls,
                    'most_only': most_only
                }
                
                if spark:
                    # we use the same seed, so that we can reuse the cache
                    specs.append(CVSpec(
                        model_cls(), grid, features,
                        'phi_m', save_to=fname, meta=meta,
                        inner_seed=233522635,
                        outer_seed=773466534,
                        param_seed=634643256,
                    ))
                else:
                    results, preds = nested_cv(
                        df, model_cls, grid, features, 'phi_m',
                        most_only, use_trend=trend,
                    )

                    specs.append(finalize_result(
                        None, list(zip(
                            results.values,
                            [None] * results.values.shape[0],
                            [preds]
                        )), save_to=fname
                    ))

    if spark:
        cv_results = nested_cv_spark(df, *specs)
        return cv_results
    return specs


# In[12]:


class LogUniform:
    ''' random variable X such that log(x) is distributed uniformly
    '''
    def __init__(self, base, expmin, expmax):
        self.base, self.expmin, self.expmax = base, expmin, expmax

    def rvs(self, random_state=None, size=None):
        random_state = random_state or np.random.RandomState()
        exp = random_state.uniform(self.expmin, self.expmax, size=size)
        return np.power(self.base, exp)


class IntDistribution:
    ''' random variable taking only integer values
    '''
    def __init__(self, rv):
        self.rv = rv

    def rvs(self, *args, **kwargs):
        sample = self.rv.rvs(*args, **kwargs)
        return int(sample)


if __name__ == '__main__':
    df = read_df()
    ddf, _ = get_features(df, use_trend=True, feature_level=5)
    use_spark = True

    res, preds = nested_cv(
        ddf, MOSTEstimator(), {'regu': LogUniform(10, -6, 1)}, ['zL'], 'phi_m',
        most_only=True, n_jobs=-1, normalize=False, use_trend=False,
    )

    import pdb; pdb.set_trace()

    print(res.to_string())
    print(res.describe().T)


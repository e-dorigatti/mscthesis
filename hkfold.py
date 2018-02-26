from sklearn.model_selection import KFold
import numpy as np


class HKFold:
    ''' h-k fold cross validation
        discard training samples closer than h to any test sample
    '''
    def __init__(self, cv, h, warn=True):
        self.h, self.cv, self.warn = h, cv, warn
        if self.h < 0:
            raise ValueError('h must be non-negative')

    def _get_mask(self, train_idx, test_idx):
        # assumes both train_idx and test_idx are sorted

        if self.h < 1:
            hh = int((len(train_idx) + len(test_idx)) * self.h)
        else:
            hh = int(self.h)

        valid, j, mask = False, 0, np.zeros(len(train_idx), dtype=np.bool)
        for i, ti in enumerate(train_idx):
            # loop invariant: test_idx[j - 1] < ti < test_idx[j]
            while j < len(test_idx) - 1 and ti > test_idx[j]:
                j += 1

            # check that test_idx[j - 1] + h < ti < test_idx[j] - h
            ok_below = j == 0 or abs(ti - test_idx[j - 1]) > hh
            ok_above = abs(test_idx[j] - ti) > hh

            mask[i] = ok_above and ok_below
            valid |= mask[i]

        return valid, mask

    def split(self, *args, **kwargs):
        for train_idx, test_idx in self.cv.split(*args, **kwargs):
            valid, mask = self._get_mask(train_idx, test_idx)
            if valid:
                yield train_idx[mask], test_idx
            elif self.warn:
                print('%s - WARNING: empty training partition, skipping... '
                      '// you might want to set a lower h and/or bigger folds' % self.__class__.__name__)

    def get_n_splits(self, *args, **kwargs):
        return self.cv.get_n_splits(*args, **kwargs)


def train_test_split(*arrays, h, test_size=0.1):
    assert test_size > 0
    assert len(set(len(a) for a in arrays)) == 1

    if test_size > 1:
        test_size = test_size / len(arrays[0])

    if test_size < 1:
        k = int(1 / test_size)

    hkf = HKFold(KFold(k, shuffle=True), h=h, warn=False)
    train_idx, test_idx = next(hkf.split(arrays[0]))

    res = []
    for a in arrays:
        res.append(a[train_idx])
        res.append(a[test_idx])
    return res


def test_ch(args):
    c, h, cnt = args
    ok = True
    for i in range(cnt):
        hkcv = HKFold(KFold(c, shuffle=True), h=h, warn=False)
        for train_idx, test_idx in hkcv.split(range(100)):
            ok &= all(abs(tri - tei) > h for tei in test_idx for tri in train_idx)
    return c, h, ok


if __name__ == '__main__':
    import dask.bag as db
    b = (db.from_sequence((c, h, 1000) for c in range(3, 11) for h in range(5))
         .map(test_ch)
         .filter(lambda x: not x[2])
         .compute())
    assert not b, 'some test failed'

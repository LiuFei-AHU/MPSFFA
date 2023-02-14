from scipy.ndimage import rotate
import numpy as np
import random
from skimage.exposure.exposure import adjust_gamma, equalize_hist

class DataAug():
    def __init__(self):
        super().__init__()

class xy_rotate(DataAug):
    def __init__(self, mn=0, mx=0, axis=(0,1), rate=0.5):
        super().__init__()
        self.mn = mn
        self.mx = mx
        self.axis = axis
        self.rate = rate
    
    def __call__(self, data):
        # axes = [(1, 0), (2, 1), (2, 0)]
        # axis = axes[np.random.randint(len(axes))]
        if random.random() >= self.rate:
            data = rotate(data, angle=np.random.randint(self.mn, self.mx), axes=self.axis, reshape=False)
            data[data < 0.] = 0.
        return data


class xyz_rotate(DataAug):
    def __init__(self, mn=0, mx=0, rate=0.5):
        super().__init__()
        self.mn = mn
        self.mx = mx
        self.rate = rate
        self.axis = [(0,1),(0,2),(1,2)]
    
    def __call__(self, data):
        if random.random() >= self.rate:
            axis = self.axis[np.random.randint(3)]
            data = rotate(data, angle=np.random.randint(self.mn, self.mx), axes=axis, reshape=False)
            data[data < 0.] = 0.
        return data

class gamma_adjust(DataAug):
    def __init__(self,mn=0, mx=0, rate=0.5):
        super().__init__()
        self.mn = mn
        self.mx = mx
        self.rate = rate
    
    def __call__(self, data):
        if random.random() >= self.rate:
            data = adjust_gamma(data, gamma=round(random.uniform(self.mn, self.mx), 1))
        return data

class contrast(DataAug):
    def __init__(self):
        super().__init__()
    
    def __call__(self, data):
        return np.clip(random.uniform(0.8, 1.3) * data, -1, 1)

class equa_hist(DataAug):
    def __init__(self):
        super().__init__()
    
    def __call__(self, data):
        return equalize_hist(data)

class flip(DataAug):
    def __init__(self,rate=0.5):
        super().__init__()
        self.rate = rate
    
    def __call__(self, data):
        if random.random() >= self.rate:
            return np.fliplr(data)
        else:
            return data

class sample(DataAug):
    def __init__(self):
        super().__init__()
    
    def __call__(self, data):
        return data[0:-1:2,0:-1:2,0:-1:2]

class mask(DataAug):
    def __init__(self, mask_nums=1,size=[10,15,10],intersect=True):
        super().__init__()
        self.size = size
        self.mask_nums = mask_nums
        self.intersect = intersect
    
    def __call__(self, data):
        # dhw
        x,y,z = data.shape
        d,h,w = self.size
        i_list = []
        m=0
        while m<self.mask_nums:
            xi =random.randint(0, x-d)
            yi =random.randint(0, y-h)
            zi =random.randint(0, z-w)
            if not self.intersect:
                flag = False
                for _,(ai,bi,ci) in enumerate(i_list):
                    if abs(ai-xi)<d and abs(bi-yi)<h and abs(ci-zi)<w:
                        flag = True
                        break
                if flag:
                    continue
            m += 1
            i_list.append((xi,yi,zi))
            mask = np.ones(data.shape)
            mask[xi:xi+d,yi:yi+h,zi:zi+w] = 0
            data = data*mask
        return data
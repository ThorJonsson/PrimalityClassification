import numpy as np
import pandas as pd
import pdb
'''
a function to check if a positive integer is prime
prime numbers are only divisible by unity and themselves
integers less than 2 and even numbers other than 2 are not prime
tested with Python27 and Python33  by  vegaseat  30aug2013
'''
def is_prime(n):
    '''
    check if integer n is a prime, return True or False
    '''
    # 2 is the only even prime
    if n == 2:
        return 1
    # integers less than 2 and even numbers other than 2 are not prime
    elif n < 2 or not n & 1:
        return 0
    # loop looks at odd numbers 3, 5, 7, ... to sqrt(n)
    for i in range(3, int(n**0.5)+1, 2):
        if n % i == 0:
            return 0
    return 1


def create_bin_vector(n):
    length = len(bin(n)[2:])
    bin_vec = np.zeros(shape=(15,))
    for i,x_i in enumerate(bin(n)[2:]):
        bin_vec[i-length] = x_i
    return bin_vec


def make_df(size = 10000):
    data_points = np.random.randint(low = 0, high=2**(14), size=10000)
    df = pd.DataFrame(data = data_points)
    df['binary_representation'] = df[0].apply(create_bin_vector)
    df['primality'] = df[0].apply(is_prime)
    return df


class BinaryPrimalityRandomIterator(object):
    def __init__(self, dataframe, batch_size = 64):
        self.batch_size = batch_size
        self.df = dataframe
        self._set_modalities()

        self.size = len(self.data_points)

        # Cursor to maintain dataset throughout an epoch
        self.cursor = 0
        self.shuffle()
        self.epoch = 0

        # Due to being a binary classification task
        self.num_classes = 1 


    def _set_modalities(self):
        self.data_points = self.df[0].tolist()
        self.bin_rep = self.df['binary_representation'].tolist()
        self.primality = self.df['primality'].tolist()


    def shuffle(self):
        # drop=True prevents .reset_index from creating old index column
        self.df.sample(frac=1).reset_index(drop=True)
        self._set_modalities()


    def next_batch(self, increment=False):
        if self.cursor + self.batch_size > self.size:
            self.cursor = 0
            self.epoch += 1
            self.shuffle()

        batch_data_points = self.data_points[self.cursor:self.cursor+self.batch_size]
        batch_bin_rep = self.bin_rep[self.cursor:self.cursor+self.batch_size]
        batch_primality = np.array(self.primality[self.cursor:self.cursor+self.batch_size])
        batch_primality = np.reshape(batch_primality, (self.batch_size,1))
        if increment:
            self.cursor += self.batch_size

        return batch_data_points, batch_bin_rep, batch_primality

def test():
  df = make_df()
  print(df.head())
  print(is_prime(19))
  iterator = BinaryPrimalityRandomIterator(df)
  print(iterator.next_batch(increment=True))

# CSCI 323/700
# Summer 2022
# Assignment 4 - Empirical Performance of Search Structures
# Andrew Zheng
# Note: This code is based on hash tables for 4 different functions in terms of build and searching
# For bit_hash and Cuckoo Hashing class/function I used the code from https://github.com/ettirapp/cuckoo-hash
# For double hashing class, I improvised/revised of the code from
# https://gist.github.com/cs-fedy/c7d6deffff21442fabe7e4253e04032e
# For the creating of a second hash table which would be used for resolving collision
# by chaining through the use of two array/tables.
# Any same position element is placed to second table
# and the code is gotten from https://github.com/DeborahHier/CuckooHash/blob/master/CuckooHash.py


import time
import pandas as pd
import matplotlib.pyplot as plt
import random


class Node(object):
    def __init__(self, k, d):
        self.key = k
        self.data = d

    def __str__(self):
        return "(" + str(self.key) + ", " + str(self.data) + ")"


chs = {}


def build_chaining_hash(data):
    hash = {}
    size = len(data)
    bucket = size // 3
    for i in range(0, size // 3):
        hash[i] = []
    for i in data:
        key = i % bucket
        hash[key].append(data)

    return hash


def build_quad_hash(data):
    capacity = len(data)
    arr = data
    N = len(data)
    table = {}
    for i in range(capacity):
        table[i] = -1
    for i in range(N):
        m = arr[i] % capacity
        if table[m] == -1:
            table[m] = arr[i]
        else:
            for j in range(capacity):
                t = (m + j * j) % capacity
                if table[t] == -1:
                    table[t] = arr[i]
                    break
    return arr


def search_quad_hash(data, item):
    capacity = len(data)
    key = item % capacity

    h = 1
    while data[key] != -1 and h < capacity:
        if data[key] == item:
            return True
        key = (key + h * h) % capacity
        h += 1
    return False


def build_double_hash(data):
    table1 = doublehash_table(len(data))
    for d in data:
        table1.insert(d)
    return table1.table


def build_cuckoo_hash(data):
    ch = cuckoo_hash(len(data))
    for i in data:
        ch.insert(str(i % len(data)), i)
    chs[len(data)] = ch
    return data


def pseudo_random_list(n):
    data = [0]
    for i in range(1, n):
        data.append(data[i - 1] + random.randint(1, 10))
    random.shuffle(data)
    return data


# get random subset of size items from data list
def get_random_sublist(data, size):
    return [data[random.randint(0, len(data) - 1)] for i in range(size)]


def search_chaining_hash(data, item):
    size = len(data.keys())
    key = item % size
    if item in data[key]:
        return True
    else:
        return False


def search_double_hash(data, item):
    table1 = doublehash_table(len(data))
    table1.table = data
    table1.size = len(data)
    found = False
    position = table1.h1(item)
    table1.comparisons += 1
    if table1.table[position] == item:
        return position
    else:
        limit = 4
        i = 2
        new_position = position
        # start a loop to find the position
        while i <= limit:
            # calculate new position by double Hashing
            position = (i * table1.h1(item) + table1.h2(item)) % table1.size
            table1.comparisons += 1
            if table1.table[position] == item:
                found = True
                break
            elif table1.table[position] == 0:
                found = False
                break
            else:
                # as the position is not empty increase i
                i += 1
        if found:
            return position
        else:
            return found


def search_cuckoo_hash(data, item):
    ch = chs[len(data)]

    if ch.find(str(item)) is None:
        return False
    else:
        return True


class HashTab(object):
    def __init__(self, size):
        self.__hash_array1 = [None] * (size // 2)
        self.__hash_array2 = [None] * (size // 2)
        self.__numRecords = 0
        self.__size = size

    def __len__(self):
        return self.__numRecords

    def hashFunc(self, s):
        x = bit_hash(s)
        y = bit_hash(s, x)

        size = self.__size // 2

        return x % size, y % size

    def insert(self, k, d):
        if self.find(k) is not None:  return False

        n = Node(k, d)

        if self.__numRecords >= (self.__size // 2):
            self.__growHash()

        position1, position2 = self.hashFunc(n.key)

        pos = position1
        table = self.__hash_array1

        for i in range(5):

            if table[pos] is None:
                table[pos] = n
                self.__numRecords += 1
                return True

            n, table[pos] = table[pos], n

            if pos == position1:
                position1, position2 = self.hashFunc(n.key)
                pos = position2
                table = self.__hash_array2
            else:
                position1, position2 = self.hashFunc(n.key)
                pos == position1
                table = self.__hash_array1

        self.__growHash()
        self.rehash(self.__size)
        self.insert(n.key, n.data)
        return True

    def __str__(self):
        str1 = "Table 1: [ " + str(self.__hash_array1[0])
        str2 = " Table 2: [ " + str(self.__hash_array2[0])
        for i in range(1, self.__size):
            str1 += ", " + str(self.__hash_array1[i])
        str1 += "]"

        for i in range(1, self.__size):
            str2 += ", " + str(self.__hash_array2[i])
        str2 += "]"

        return str1 + str2

        # get new hash functions and reinsert everything

    def rehash(self, size):
        reset_bit_hash()

        temp = HashTab(size)  # create new hash tables

        # re-hash each item and insert it into the correct position in the new tables
        for i in range(self.__size // 2):
            x = self.__hash_array1[i]
            y = self.__hash_array2[i]
            if x is not None:
                temp.insert(x.key, x.data)
            if y is not None:
                temp.insert(y.key, y.data)

        # save new tables
        self.__hash_array1 = temp.__hash_array1
        self.__hash_array2 = temp.__hash_array2
        self.__numRecords = temp.__numRecords
        self.__size = temp.__size

    # Increase the hash table's size x 2
    def __growHash(self):
        newSize = self.__size * 2
        # re-hash each item and insert it into the
        # correct position in the new table
        self.rehash(newSize)

    # Return data if there, otherwise return None
    def find(self, k):
        pos1, pos2 = self.hashFunc(k)  # check both positions the key/data
        x = self.__hash_array1[pos1]  # could be in. return data if found.
        y = self.__hash_array2[pos2]
        if x is not None and x.key == k: return x.data
        if y is not None and y.key == k: return y.data

        # return None if the key can't be found
        return None


class doublehash_table:
    # initialize hash Table
    def __init__(self, size):
        self.size = size * 2
        self.num = 5
        # initialize table with all elements 0
        self.table = list(0 for i in range(self.size))
        self.elementCount = 0
        self.comparisons = 0

    # method that checks if the hash table is full or not
    def isFull(self):
        if self.elementCount == self.size:
            return True
        else:
            return False

    # method that returns position for a given element
    # replace with your own hash function
    def h1(self, element):
        return element % self.size

    # method that returns position for a given element
    def h2(self, element):
        return element % self.num

    # method to resolve collision by quadratic probing method
    def doubleHashing(self, element, position):
        pos_found = False
        # limit variable is used to restrict the function from going into infinite loop
        # limit is useful when the table is 80% full
        limit = 4
        i = 2
        # start a loop to find the position
        while i <= limit:
            # calculate new position by quadratic probing
            new_position = (i * self.h1(element) + self.h2(element)) % self.size
            # if new_position is empty then break out of loop and return new Position
            if self.table[new_position] == 0:
                pos_found = True
                break
            else:
                i += 1
        return pos_found, new_position

    # method that inserts element inside the hash table
    def insert(self, element):
        # checking if the table is full
        if self.isFull():
            return False
        pos_found = False
        position = self.h1(element)
        # checking if the position is empty
        if self.table[position] == 0:
            self.table[position] = element
            isStored = True
            self.elementCount += 1
        # collision occured hence we do linear probing
        else:
            count = 1
            while not pos_found and count < self.size:
                pos_found, position = self.doubleHashing(element, position)
                if pos_found:
                    self.table[position] = element
                    self.elementCount += 1
                count += 1
        return pos_found

    def search(self, element):
        found = False
        position = self.h1(element)
        self.comparisons += 1
        if self.table[position] == element:
            return position
        # if element is not found at position returned hash function
        # then we search element using double hashing
        else:
            limit = 4
            i = 2
            new_position = position
            # start a loop to find the position
            while i <= limit:
                # calculate new position by double Hashing
                position = (i * self.h1(element) + self.h2(element)) % self.size
                self.comparisons += 1
                # if element at new_position is equal to the required element
                if self.table[position] == element:
                    found = True
                    break
                elif self.table[position] == 0:
                    found = False
                    break
                else:
                    # as the position is not empty increase i
                    i += 1
            if found:
                return position
            else:
                return found


# setup a list of random 64-bit values to be used by bit_hash
__bits = [0] * (64 * 1024)
__rnd = random.Random()

# seed the generator to produce repeatable results
__rnd.seed("bit_hash random numbers")

# fill the list
for i in range(64 * 1024):
    __bits[i] = __rnd.getrandbits(64)


def bit_hash(s, h=0):
    for c in s:
        h = (((h << 1) | (h >> 63)) ^ __bits[ord(c)])
        h &= 0xffffffffffffffff
    return h


# this function causes subsequent calls to bit_hash to be
# based on a new set of random numbers. This is useful
# in the event that client code needs a new hash function,
# for example, for Cuckoo Hashing.
def reset_bit_hash():
    global __bits
    for i in range(64 * 1024):
        __bits[i] = __rnd.getrandbits(64)


class cuckoo_hash(object):
    def __init__(self, size):
        self.__table1 = [None] * size
        self.__table2 = [None] * size
        self.__num_buckets = size
        self.__max_loop = 16
        self.__num_keys = 0

    def __h1(self, key):
        return bit_hash(key) % self.__num_buckets

    def __h2(self, key):
        return bit_hash(key, bit_hash(key)) % self.__num_buckets

    def find(self, key):
        if self.__table1[self.__h1(key)] and \
                self.__table1[self.__h1(key)][0] == key:
            return 1, self.__table1[self.__h1(key)][1]
        elif self.__table2[self.__h2(key)] and \
                self.__table2[self.__h2(key)][0] == key:
            return 2, self.__table2[self.__h2(key)][1]
        return None

    def insert(self, key, data):
        self.__insert(key, [data])

    def __insert(self, key, data):
        if self.find(key):
            if self.find(key)[0] == 1:
                self.__table1[self.__h1(key)][1].append(data)
            elif self.find(key)[0] == 2:
                self.__table2[self.__h2(key)][1].append(data)
            return
        if self.__num_keys >= .5 * self.__num_buckets:
            self.__growTables()
        key_data = (key, data)
        for i in range(self.__max_loop):
            if not self.__table1[self.__h1(key_data[0])]:
                self.__table1[self.__h1(key_data[0])] = key_data
                self.__num_keys += 1
                return
            temp = self.__table1[self.__h1(key_data[0])]
            self.__table1[self.__h1(key_data[0])] = key_data
            key_data = temp
            if not self.__table2[self.__h2(key_data[0])]:
                self.__table2[self.__h2(key_data[0])] = key_data
                self.__num_keys += 1
                return
            temp = self.__table2[self.__h2(key_data[0])]
            self.__table2[self.__h2(key_data[0])] = key_data
            key_data = temp

    def __growTables(self):
        self.__num_buckets *= 2
        tab1 = [None] * self.__num_buckets
        tab2 = [None] * self.__num_buckets
        for key_data in self.__table1:
            if key_data:
                for i in range(self.__max_loop):
                    if not tab1[self.__h1(key_data[0])]:
                        tab1[self.__h1(key_data[0])] = key_data
                        break
                    temp = tab1[self.__h1(key_data[0])]
                    tab1[self.__h1(key_data[0])] = key_data
                    key_data = temp
                    if not tab2[self.__h2(key_data[0])]:
                        tab2[self.__h2(key_data[0])] = key_data
                        break
                    temp = tab2[self.__h2(key_data[0])]
                    tab2[self.__h2(key_data[0])] = key_data
                    key_data = temp

        for key_data in self.__table2:
            if key_data:
                for i in range(self.__max_loop):
                    if not tab1[self.__h1(key_data[0])]:
                        tab1[self.__h1(key_data[0])] = key_data
                        break
                    temp = tab1[self.__h1(key_data[0])]
                    tab1[self.__h1(key_data[0])] = key_data
                    key_data = temp
                    if not tab2[self.__h2(key_data[0])]:
                        tab2[self.__h2(key_data[0])] = key_data
                        break
                    temp = tab2[self.__h2(key_data[0])]
                    tab2[self.__h2(key_data[0])] = key_data
                    key_data = temp
        self.__table1 = tab1
        self.__table2 = tab2


def plot_time(dict_builds, sizes, builds, trials, dict_searches, searches):
    build_num = 0
    plt.xticks([j for j in range(len(sizes))], [str(size) for size in sizes])
    for build in builds:
        build_num += 1
        d = dict_builds[build.__name__]
        x_axis = [j + 0.05 * build_num for j in range(len(sizes))]
        y_axis = [d[i] for i in sizes]
        plt.bar(x_axis, y_axis, width=0.1, alpha=0.75, label=build.__name__)
    plt.legend()
    plt.title("Run time of Algorithms")
    plt.xlabel("Size of data")
    plt.ylabel("Time for " + str(trials) + " trials (ms)")
    plt.savefig("Assignment 4a")
    plt.show()
    search_num = 0
    plt.xticks([j for j in range(len(sizes))], [str(size) for size in sizes])
    for search in searches:
        search_num += 1
        d = dict_searches[search.__name__]
        x_axis = [j + 0.05 * search_num for j in range(len(sizes))]
        y_axis = [d[i] for i in sizes]
        plt.bar(x_axis, y_axis, width=0.1, alpha=0.75, label=search.__name__)
    plt.legend()
    plt.title("Run time of Algorithms")
    plt.xlabel("Size of data")
    plt.ylabel("Time for " + str(trials) + " trials (ms)")
    plt.savefig("Assignment 4b")
    plt.show()


def main():
    hash_chaining = 1
    hash_quadratic = 1
    hash_double = 1
    hash_cuckoo = 1

    chaining_hash = {}

    sizes = [100 * i for i in range(1, 11)]
    trials = 1
    searches = [search_chaining_hash, search_quad_hash, search_double_hash, search_cuckoo_hash]
    builds = [build_chaining_hash, build_quad_hash, build_double_hash, build_cuckoo_hash]

    dict_builds = {}
    dict_searches = {}
    for build in builds:
        dict_builds[build.__name__] = {}
    for search in searches:
        dict_searches[search.__name__] = {}
    for size in sizes:
        for build in builds:
            dict_builds[build.__name__][size] = 0
    for size in sizes:
        for search in searches:
            dict_searches[search.__name__][size] = 0
        for trial in range(1, trials + 1):
            data = pseudo_random_list(size)
            sublist = get_random_sublist(data, 100)
            hash_tables = []
            for build in builds:
                start_time = time.time()
                hash_tables.append(build(data))
                end_time = time.time()
                net_time = end_time - start_time
                dict_builds[build.__name__][size] += 1000 * net_time
            # recursive equation
            for i in range(len(searches)):
                search = searches[i]
                table = hash_tables[i]
                start_time = time.time()
                for item in sublist:
                    search(table, item)
                end_time = time.time()
                net_time = end_time - start_time
                dict_searches[search.__name__][size] += 1000 * net_time

    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 1000)
    print("Builds: ")
    df = pd.DataFrame.from_dict(dict_builds).T
    print(df)
    print("\nSearches: ")
    df = pd.DataFrame.from_dict(dict_searches).T
    print(df)

    plot_time(dict_builds, sizes, builds, trials, dict_searches, searches)


if __name__ == "__main__":
    main()

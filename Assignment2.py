# CSCI 323/700
# Summer 2022
# Assignment 2 - Sorting Algorithms
# Andrew Zheng
# Note: Most of the used sorting algorithm came from geeksforgeeks as given in assignment 2 pdf
# The algorithm for bucketlist came from https://pythonwife.com/sorting-algorithms-in-python/
# the copy.copy was used on the initial random list to be reused for each sorting algorithm
# The code for checking whether the list is sorted or not for also came from geeksforgeeks https://www.geeksforgeeks.org/python-check-if-list-is-sorted-or-not/

import copy
import sys
import random
import time
import math
import pandas as pd
import matplotlib.pyplot as plt

sys.setrecursionlimit(30000)


def random_list(range_max, size):
    numbers = []
    i = 0
    while i < size:
        rnd = random.randint(1, range_max)
        numbers.append(rnd)
        i += 1

    return numbers


def native_sort(arr):
    return arr.sort()


def bubble_sort(arr):
    n = len(arr)

    # Traverse through all array elements
    for i in range(n):

        # Last i elements are already in place
        for j in range(0, n - i - 1):

            # traverse the array from 0 to n-i-1
            # Swap if the element found is greater
            # than the next element
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]


def selection_sort(arr):
    n = len(arr)
    for i in range(n - 1):
        min_value_index = i

        for j in range(i + 1, n):
            if arr[j] < arr[min_value_index]:
                min_value_index = j

        if min_value_index != i:
            temp = arr[i]
            arr[i] = arr[min_value_index]
            arr[min_value_index] = temp

    return arr


def insertion_sort(arr):
    # Traverse through 1 to len(arr)
    for i in range(1, len(arr)):

        key = arr[i]

        # Move elements of arr[0..i-1], that are
        # greater than key, to one position ahead
        # of their current position
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key


def cocktail_sort(arr):
    n = len(arr)
    swapped = True
    start = 0
    end = n - 1
    while swapped == True:

        # reset the swapped flag on entering the loop,
        # because it might be true from a previous
        # iteration.
        swapped = False

        # loop from left to right same as the bubble
        # sort
        for i in range(start, end):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                swapped = True

        # if nothing moved, then array is sorted.
        if swapped == False:
            break

        # otherwise, reset the swapped flag so that it
        # can be used in the next stage
        swapped = False

        # move the end point back by one, because
        # item at the end is in its rightful spot
        end = end - 1

        # from right to left, doing the same
        # comparison as in the previous stage
        for i in range(end - 1, start - 1, -1):
            if a[i] > a[i + 1]:
                a[i], a[i + 1] = a[i + 1], a[i]
                swapped = True

        # increase the starting point, because
        # the last stage would have moved the next
        # smallest number to its rightful spot.
        start = start + 1


def shell_sort(arr, n):
    gap = n // 2

    while gap > 0:
        j = gap
        # Check the array in from left to right
        # Till the last possible index of j
        while j < n:
            i = j - gap  # This will keep help in maintain gap value

            while i >= 0:
                # If value on right side is already greater than left side value
                # We don't do swap else we swap
                if arr[i + gap] > arr[i]:

                    break
                else:
                    arr[i + gap], arr[i] = arr[i], arr[i + gap]

                i = i - gap  # To check left side also
                # If the element present is greater than current element
            j += 1
        gap = gap // 2


def merge_sort(arr):
    if len(arr) > 1:

        # Finding the mid of the array
        mid = len(arr) // 2

        # Dividing the array elements
        L = arr[:mid]

        # into 2 halves
        R = arr[mid:]

        # Sorting the first half
        merge_sort(L)

        # Sorting the second half
        merge_sort(R)

        i = j = k = 0

        # Copy data to temp arrays L[] and R[]
        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1

        # Checking if any element was left
        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1

        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1


# Code to print the list


def partition(array, low, high):
    # Choose the rightmost element as pivot
    pivot = array[high]

    # Pointer for greater element
    i = low - 1

    # Traverse through all elements
    # compare each element with pivot
    for j in range(low, high):
        if array[j] <= pivot:
            # If element smaller than pivot is found
            # swap it with the greater element pointed by i
            i = i + 1

            # Swapping element at i with element at j
            (array[i], array[j]) = (array[j], array[i])

    # Swap the pivot element with the greater element specified by i
    (array[i + 1], array[high]) = (array[high], array[i + 1])

    # Return the position from where partition is done
    return i + 1


# Function to perform quicksort
def quick_sort(array, low, high):
    if low < high:
        # Find pivot element such that
        # element smaller than pivot are on the left
        # element greater than pivot are on the right
        pi = partition(array, low, high)

        # Recursive call on the left of pivot
        quick_sort(array, low, pi - 1)

        # Recursive call on the right of pivot
        quick_sort(array, pi + 1, high)


def heapify(arr, n, i):
    largest = i  # Initialize largest as root
    l = 2 * i + 1  # left = 2*i + 1
    r = 2 * i + 2  # right = 2*i + 2

    # See if left child of root exists and is
    # greater than root
    if l < n and arr[largest] < arr[l]:
        largest = l

    # See if right child of root exists and is
    # greater than root
    if r < n and arr[largest] < arr[r]:
        largest = r

    # Change root, if needed
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]  # swap

        # Heapify the root.
        heapify(arr, n, largest)


# The main function to sort an array of given size


def heap_sort(arr):
    n = len(arr)

    # Build a maxheap.
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    # One by one extract elements
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]  # swap
        heapify(arr, i, 0)


def count_sort(arr):
    max_element = int(max(arr))
    min_element = int(min(arr))
    range_of_elements = max_element - min_element + 1
    # Create a count array to store count of individual
    # elements and initialize count array as 0
    count_arr = [0 for _ in range(range_of_elements)]
    output_arr = [0 for _ in range(len(arr))]

    # Store count of each character
    for i in range(0, len(arr)):
        count_arr[arr[i] - min_element] += 1

    # Change count_arr[i] so that count_arr[i] now contains actual
    # position of this element in output array
    for i in range(1, len(count_arr)):
        count_arr[i] += count_arr[i - 1]

    # Build the output character array
    for i in range(len(arr) - 1, -1, -1):
        output_arr[count_arr[arr[i] - min_element] - 1] = arr[i]
        count_arr[arr[i] - min_element] -= 1

    # Copy the output array to arr, so that arr now
    # contains sorted characters
    for i in range(0, len(arr)):
        arr[i] = output_arr[i]

    return arr


def bucket_sort(temp_list):
    number_of_buckets = round(math.sqrt(len(temp_list)))
    max_val = max(temp_list)
    arr = []

    for i in range(number_of_buckets):
        arr.append([])
    for j in temp_list:
        index_b = math.ceil(j * number_of_buckets / max_val)
        arr[index_b - 1].append(j)

    for i in range(number_of_buckets):
        insertion_sort(arr[i])

    k = 0
    for i in range(number_of_buckets):
        for j in range(len(arr[i])):
            temp_list[k] = arr[i][j]
            k += 1
    return temp_list


def counting_sort(arr, exp1):
    n = len(arr)

    # The output array elements that will have sorted arr
    output = [0] * n

    # initialize count array as 0
    count = [0] * 10

    # Store count of occurrences in count[]
    for i in range(0, n):
        index = arr[i] // exp1
        count[index % 10] += 1

    # Change count[i] so that count[i] now contains actual
    # position of this digit in output array
    for i in range(1, 10):
        count[i] += count[i - 1]

    # Build the output array
    i = n - 1
    while i >= 0:
        index = arr[i] // exp1
        output[count[index % 10] - 1] = arr[i]
        count[index % 10] -= 1
        i -= 1

    # Copying the output array to arr[],
    # so that arr now contains sorted numbers
    i = 0
    for i in range(0, len(arr)):
        arr[i] = output[i]


# Method to do Radix Sort
def radix_sort(arr):
    # Find the maximum number to know number of digits
    max1 = max(arr)

    # Do counting sort for every digit. Note that instead
    # of passing digit number, exp is passed. exp is 10^i
    # where i is current digit number
    exp = 1
    while max1 / exp > 1:
        counting_sort(arr, exp)
        exp *= 10


def cycle_sort(arr):
    writes = 0

    # Loop through the array to find cycles to rotate.
    for cycleStart in range(0, len(arr) - 1):
        item = arr[cycleStart]

        # Find where to put the item.
        pos = cycleStart
        for i in range(cycleStart + 1, len(arr)):
            if arr[i] < item:
                pos += 1

        # If the item is already there, this is not a cycle.
        if pos == cycleStart:
            continue

        # Otherwise, put the item there or right after any duplicates.
        while item == arr[pos]:
            pos += 1
        arr[pos], item = item, arr[pos]
        writes += 1

        # Rotate the rest of the cycle.
        while pos != cycleStart:

            # Find where to put the item.
            pos = cycleStart
            for i in range(cycleStart + 1, len(arr)):
                if arr[i] < item:
                    pos += 1

            # Put the item there or right after any duplicates.
            while item == arr[pos]:
                pos += 1
            arr[pos], item = item, arr[pos]
            writes += 1

    return writes


def array_sorted_or_not(arr, n):
    # Array has one or no element
    if n == 0 or n == 1:
        return True

    for i in range(1, n):

        # Unsorted pair found
        if arr[i - 1] > arr[i]:
            return False

    # No unsorted pair found
    return True


def plot_time(dict_sorting, sizes, sorting):
    sort_num = 0
    plt.xticks([j for j in range(len(sizes))], [str(size) for size in sizes])
    for sort in sorting:
        sort_num += 1
        d = dict_sorting[sort.__name__]
        x_axis = [j + 0.05 * sort_num for j in range(len(sizes))]
        y_axis = [d[i] for i in sizes]
        plt.bar(x_axis, y_axis, width=0.05, alpha=0.75, label=sort.__name__)
    plt.legend()
    plt.title("Run time of sorting algorithms")
    plt.xlabel("Different Sizes of n")
    plt.ylabel("Time (ms)")
    plt.savefig("Assignment2.png")
    plt.show()


def main():
    sizes = [10 * i for i in (1, 10, 100, 1000)]

    trials = 1

    sorting = [bubble_sort, native_sort, selection_sort, insertion_sort,
               cocktail_sort, shell_sort, merge_sort,
               quick_sort, heap_sort, count_sort,
               bucket_sort, radix_sort, cycle_sort]

    dict_sorting = {}
    for sort in sorting:
        dict_sorting[sort.__name__] = {}
    for size in sizes:

        for sort in sorting:
            dict_sorting[sort.__name__][size] = 0
        for trial in range(1, trials + 1):
            rand_list = random_list(10000, size)

            newList = copy.copy(rand_list)
            newList2 = copy.copy(rand_list)
            newList3 = copy.copy(rand_list)
            newList4 = copy.copy(rand_list)
            newList5 = copy.copy(rand_list)
            newList6 = copy.copy(rand_list)
            newList7 = copy.copy(rand_list)
            newList8 = copy.copy(rand_list)
            newList9 = copy.copy(rand_list)
            newList10 = copy.copy(rand_list)
            newList11 = copy.copy(rand_list)
            newList12 = copy.copy(rand_list)
            newList13 = copy.copy(rand_list)

        for sort in sorting:
            start_time = time.time()
            native_sort(newList)
            bubble_sort(newList)
            selection_sort(newList)
            insertion_sort(newList)
            cocktail_sort(newList)
            shell_sort(newList, len(newList))
            merge_sort(newList)
            quick_sort(newList, 0, len(newList) - 1)
            heap_sort(newList5)
            count_sort(newList)
            bucket_sort(newList)
            radix_sort(newList)
            cycle_sort(newList)
            end_time = time.time()
            net_time = end_time - start_time

            dict_sorting[sort.__name__][size] += 1000 * net_time
    n = len(rand_list)
    if array_sorted_or_not(newList, n):
        print("The list is sorted")
    else:
        print("The list is not sorted")

    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 1000)
    df = pd.DataFrame.from_dict(dict_sorting).T
    print(df)
    plot_time(dict_sorting, sizes, sorting)


if __name__ == "__main__":
    main()

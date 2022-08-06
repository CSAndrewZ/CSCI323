# CSCI 323/700
# Summer 2022
# Assignment 5 - "Palindromic Substrings and Subsequences"
# Andrew Zheng
# Note: All the codes used we're from geeksforgeeks which is listed/given in the assignment 5 instructions.
import time
import re
import texttable


def lpsst(s):
    n = len(s)
    table = [[0 for x in range(n)] for y in range(n)]
    max_length = 1
    i = 0
    while i < n:
        table[i][i] = True
        i = i + 1
    start = 0
    i = 0
    while i < n - 1:
        if s[i] == s[i + 1]:
            table[i][i + 1] = True
            start = i
            max_length = 2
        i = i + 1
    k = 3
    while k <= n:
        i = 0
        while i < (n - k + 1):
            j = i + k - 1
            if table[i + 1][j - 1] and s[i] == s[j]:
                table[i][j] = True
                if k > max_length:
                    start = i
                    max_length = k
            i = i + 1
        k = k + 1
    return s[start: start + max_length]


def lcs(X, Y):
    m = len(X)
    n = len(Y)
    L = [[None] * (n + 1) for i in range(m + 1)]
    for i in range(n + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0;
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1;
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1]);
    index = L[m][n];
    lcs2 = [""] * (index + 1)
    i, j = m, n
    while i > 0 and j > 0:
        if X[i - 1] == Y[j - 1]:
            lcs2[index - 1] = X[i - 1]
            i -= 1
            j -= 1
            index -= 1
        elif L[i - 1][j] > L[i][j - 1]:
            i -= 1
        else:
            j -= 1
    ans = ""
    for x in range(len(lcs2)):
        ans += lcs2[x]
    return ans


def lpssq(s):
    rev = s[:: -1]
    return lcs(s, rev)


def process_file(file_name):
    results = []
    with open(file_name, encoding='windows-1252') as file:
        lines = file.readlines()
        for line in lines:
            line = line.upper()  # convert to upper case
            line = re.sub(r'[^A-Z]', line)  # remove all non-alphabet chars
            start_time = time.time()
            st = lpsst(line)
            end_time = time.time()
            time_st = end_time - start_time
            start_time = time.time()
            sq = lpssq(line)
            end_time = time.time()
            time_sq = end_time - start_time
            results.append([line, len(line), st, len(st), time_st, sq, len(sq), time_sq])
    return results


def main():
    s = "QUEENSCOLLEGEOFCUNY"
    ss = lpsst(s)
    print(s, "has string length of", len(s), "and its Longest Palindromic Substring is", ss, "with a length of ",
          len(ss))
    ss2 = lpssq(s)
    print(s, "has string length of", len(s), "and its Longest Palindromic Subsequence is", ss2, "with a length of ",
          len(ss2))
    results2 = process_file("sentences.txt")
    results = process_file("palindromes.txt")
    print(results2)
    headers = ["String", "Length", "LPSST", "Length", "Time", "LPSSQ", "Length", "Time"]
    tt = texttable.Texttable(500)
    tt.set_cols_align(["l", "r", "l", "r", "r", "l", "r", "r"])
    tt.set_cols_dtype(["t", "i", "t", "i", "f", "t", "i", "f"])
    tt.add_rows(results2)
    tt.add_row(results2[1])
    tt.add_rows(results)

    tt.header(headers)
    print(tt.draw())


if __name__ == "__main__":
    main()

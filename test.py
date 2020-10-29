import numpy as np

def checkRow(sudoku_matrix, row_indx, num):
    row = sudoku_matrix[row_indx,:]
    if num in row:
        return True
    return False
def checkColumn(sudoku_matrix, col_indx, num):
    column = sudoku_matrix[:,col_indx]
    if num in column:
        return True
    return False

def checkBox(sudoku_matrix, row_indx, col_indx, num):
    sr = row_indx//3*3
    sc = col_indx//3*3
    box = sudoku_matrix[sr:sr+3,sc:sc+3]
    if num in box:
        return True
    return False

def isSafe(sudoku_matrix,ri,ci,num):
    return (not checkRow(sudoku_matrix,ri,num) and
           not checkColumn(sudoku_matrix,ci,num) and
           not checkBox(sudoku_matrix,ri,ci,num) )    

def findEmpty(sudoku_matrix,empty_loc):
    for i in range(9):
        for j in range(9):
            if(sudoku_matrix[i,j]==0):
                empty_loc[0]=i
                empty_loc[1]=j
                return True
    return False

#solution with backtracking
def solve(sudoku_matrix, empty_loc):
    if(not findEmpty(sudoku_matrix,empty_loc)):
        return True
    else:
        row=empty_loc[0]
        column = empty_loc[1]
    for num in range(1,10):
        if(isSafe(sudoku_matrix,row,column,num)):
            sudoku_matrix[row,column]=num
            if(solve(sudoku_matrix,empty_loc)):
                return True
            sudoku_matrix[row,column]=0
    return False




sudoku_matrix = np.array(
[[5, 3, 0, 0, 7, 0, 0, 0, 0],
 [6, 0, 0, 1, 9, 5, 0, 0, 0],
 [0, 9, 8, 0, 0, 0, 0, 6, 0],
 [8, 0, 0, 0, 6, 0, 0, 0, 3],
 [4, 0, 0, 8, 0, 3, 0, 0, 1],
 [7, 0, 0, 0, 2, 0, 0, 0, 6],
 [0, 6, 0, 0, 0, 0, 2, 8, 0],
 [0, 0, 0, 4, 1, 9, 0, 0, 5],
 [0, 0, 0, 0, 8, 0, 0, 7, 9]])

empty_loc = [0,0]

# print(isSafe(sudoku_matrix,8,8,4))
# check, empty_loc = findEmpty(sudoku_matrix,empty_loc)
# print(check)
# print(empty_loc)
# check,sudoku_matrix = solve(sudoku_matrix,empty_loc)
# changeMatrix(sudoku_matrix)
# print(sudoku_matrix)
# findEmpty(sudoku_matrix,empty_loc)
# print(empty_loc)
solve(sudoku_matrix,empty_loc)
print(sudoku_matrix)
# import time
# import numpy as np
#
#
# def multiply_matrices():
#     """
#     Nhân 2 ma trận cùng kích thước.
#
#     Nhập các phần tử của 2 ma trận từ người dùng, tính tích của chúng,
#     và in thời gian thực hiện của thuật toán.
#     """
#
#     # Nhập kích thước ma trận
#     rows = int(input("Nhập số hàng của ma trận: "))
#     cols = int(input("Nhập số cột của ma trận: "))
#
#     # Nhập các phần tử của ma trận A
#     print("\nNhập các phần tử của ma trận A:")
#     A = []
#     for i in range(rows):
#         row = [int(x) for x in input(f"Nhập các phần tử hàng {i + 1}: ").split()]
#         A.append(row)
#     A = np.array(A)
#
#     # Nhập các phần tử của ma trận B
#     print("\nNhập các phần tử của ma trận B:")
#     B = []
#     for i in range(rows):
#         row = [int(x) for x in input(f"Nhập các phần tử hàng {i + 1}: ").split()]
#         B.append(row)
#     B = np.array(B)
#
#     # Tính tích của 2 ma trận
#     start_time = time.time()
#     C = np.dot(A, B)
#     end_time = time.time()
#
#     # In kết quả
#     print("\nKết quả:")
#     print(C)
#     print(f"\nThời gian thực hiện: {end_time - start_time:.6f} giây")
#
#
# multiply_matrices()

import time


def input_matrix_from_file(filename):
    matrix = []
    with open(filename, 'r') as file:
        # Đọc kích thước ma trận từ dòng đầu tiên
        rows, cols = map(int, file.readline().strip().split())

        # Đọc từng hàng của ma trận
        for _ in range(rows):
            row = list(map(int, file.readline().strip().split()))
            matrix.append(row)

    return matrix, rows, cols


def print_matrix(matrix):
    for row in matrix:
        print(" ".join(map(str, row)))


def multiply_matrices(A, B, rowsA, colsA, colsB):
    # Khởi tạo ma trận kết quả với các phần tử là 0
    result = [[0 for _ in range(colsB)] for _ in range(rowsA)]
    for i in range(rowsA):
        for j in range(colsB):
            for k in range(colsA):
                result[i][j] += A[i][k] * B[k][j]
    return result


def main():
    # Nhập tên tệp chứa ma trận A

    A, rowsA, colsA = input_matrix_from_file("tt_1000.txt")

    # Nhập tên tệp chứa ma trận B

    B, rowsB, colsB = input_matrix_from_file("tt_1000.txt")

    # Kiểm tra tính khả thi của phép nhân
    if colsA != rowsB:
        print("Không thể nhân hai ma trận với kích thước đã cho.")
        return

    # Bắt đầu đo thời gian
    start_time = time.time()

    # Thực hiện phép nhân
    result = multiply_matrices(A, B, rowsA, colsA, colsB)

    # Kết thúc đo thời gian
    end_time = time.time()
    time_taken = end_time - start_time

    # In kết quả
    # print("Kết quả của phép nhân hai ma trận là:")
    # print_matrix(result)

    # In thời gian thực hiện
    print(f"Thời gian thực hiện: {time_taken:.6f} giây")


if __name__ == "__main__":
    main()

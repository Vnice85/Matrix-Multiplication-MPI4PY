# from mpi4py import MPI
# import numpy as np
#
# # Khởi tạo môi trường MPI
# comm = MPI.COMM_WORLD
# size = comm.Get_size()  # Số lượng process
# rank = comm.Get_rank()  # Xếp hạng của process hiện tại
#
# N = 3  # Kích thước ma trận
# sqrt_p = int(np.sqrt(size))  # Chia khối theo căn bậc hai của số process
# assert N % sqrt_p == 0, "Matrix size must be divisible by sqrt(number of processes)."
#
# block_size = N // sqrt_p  # Kích thước của mỗi khối con
#
# # Khởi tạo các ma trận (A, B, C) cục bộ
# A_local = np.zeros((block_size, block_size))
# B_local = np.zeros((block_size, block_size))
# C_local = np.zeros((block_size, block_size))
#
# # Hàm nhập ma trận từ người dùng
# def input_matrix(name):
#     matrix = np.zeros((N, N), dtype=int)
#     if rank == 0:
#         print(f"Nhập ma trận {name} ({N}x{N}):")
#         for i in range(N):
#             row = input(f"Dòng {i+1}: ").split()
#             matrix[i] = [int(x) for x in row]
#     return matrix
#
# # Nhập ma trận A và B từ process 0 và phân chia các khối con
# if rank == 0:
#     A = input_matrix("A")
#     B = input_matrix("B")
# else:
#     A = None
#     B = None
#
# # Chia các khối con của A và B cho các process
# A_blocks = np.split(A, sqrt_p, axis=0) if rank == 0 else None
# A_sub_blocks = [np.split(block, sqrt_p, axis=1) for block in A_blocks] if rank == 0 else None
# B_blocks = np.split(B, sqrt_p, axis=1) if rank == 0 else None
# B_sub_blocks = [np.split(block, sqrt_p, axis=0) for block in B_blocks] if rank == 0 else None
#
# # Scatter các khối đến từng process
# A_local = comm.scatter([A_sub_blocks[i][j] for i in range(sqrt_p) for j in range(sqrt_p)], root=0)
# B_local = comm.scatter([B_sub_blocks[j][i] for i in range(sqrt_p) for j in range(sqrt_p)], root=0)
#
# # Thuật toán Fox: nhân các khối cục bộ và cộng dồn vào C_local
# for k in range(sqrt_p):
#     # Broadcast khối A theo thứ tự cần thiết
#     A_broadcast = comm.bcast(A_local if (rank // sqrt_p) == (k % sqrt_p) else None, root=(k % sqrt_p) * sqrt_p)
#     # Nhân và cộng dồn kết quả vào C_local
#     C_local += np.dot(A_broadcast, B_local)
#     # Dịch vòng B_local sang trái
#     B_local = np.roll(B_local, shift=-1, axis=0)
#
# # Thu thập các khối từ tất cả các process để tạo ra ma trận C hoàn chỉnh
# C = None
# if rank == 0:
#     C = np.zeros((N, N), dtype=int)
# C_blocks = comm.gather(C_local, root=0)
#
# # Tổng hợp các khối C_local thành ma trận C trên process 0
# if rank == 0:
#     for i in range(sqrt_p):
#         for j in range(sqrt_p):
#             C[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = C_blocks[i*sqrt_p + j]
#     print("\nMa trận A:")
#     print(A)
#     print("\nMa trận B:")
#     print(B)
#     print("\nMa trận tích C = A * B:")
#     print(C)

# from mpi4py import MPI
# import numpy as np
# import time
#
#
# def benchmark_fox_algorithm(N, num_runs=5):
#     comm = MPI.COMM_WORLD
#     size = comm.Get_size()
#     rank = comm.Get_rank()
#     sqrt_p = int(np.sqrt(size))
#     block_size = N // sqrt_p
#
#     times = []
#
#     for run in range(num_runs):
#         # Khởi tạo ma trận ngẫu nhiên cho mỗi lần chạy
#         if rank == 0:
#             A = np.random.randint(0, 10, size=(N, N))
#             B = np.random.randint(0, 10, size=(N, N))
#         else:
#             A = None
#             B = None
#
#         # Đồng bộ tất cả các process trước khi bắt đầu đo thời gian
#         comm.Barrier()
#         start_time = time.time()
#
#         # Chia các khối
#         A_blocks = np.split(A, sqrt_p, axis=0) if rank == 0 else None
#         A_sub_blocks = [np.split(block, sqrt_p, axis=1) for block in A_blocks] if rank == 0 else None
#         B_blocks = np.split(B, sqrt_p, axis=1) if rank == 0 else None
#         B_sub_blocks = [np.split(block, sqrt_p, axis=0) for block in B_blocks] if rank == 0 else None
#
#         # Scatter các khối
#         A_local = comm.scatter([A_sub_blocks[i][j] for i in range(sqrt_p) for j in range(sqrt_p)], root=0)
#         B_local = comm.scatter([B_sub_blocks[j][i] for i in range(sqrt_p) for j in range(sqrt_p)], root=0)
#         C_local = np.zeros((block_size, block_size))
#
#         # Thuật toán Fox
#         for k in range(sqrt_p):
#             A_broadcast = comm.bcast(A_local if (rank // sqrt_p) == (k % sqrt_p) else None,
#                                      root=(k % sqrt_p) * sqrt_p)
#             C_local += np.dot(A_broadcast, B_local)
#             B_local = np.roll(B_local, shift=-1, axis=0)
#
#         # Thu thập kết quả
#         C_blocks = comm.gather(C_local, root=0)
#
#         if rank == 0:
#             C = np.zeros((N, N))
#             for i in range(sqrt_p):
#                 for j in range(sqrt_p):
#                     C[i * block_size:(i + 1) * block_size,
#                     j * block_size:(j + 1) * block_size] = C_blocks[i * sqrt_p + j]
#
#         # Đồng bộ tất cả các process và ghi nhận thời gian
#         comm.Barrier()
#         end_time = time.time()
#         times.append(end_time - start_time)
#
#     # Tính toán và in kết quả thống kê
#     if rank == 0:
#         avg_time = np.mean(times)
#         std_time = np.std(times)
#         min_time = np.min(times)
#         max_time = np.max(times)
#
#         # Tính toán số phép tính dấu phẩy động (FLOPS)
#         flops = 2 * N ** 3  # Số phép nhân và cộng cho nhân ma trận
#         gflops = (flops / (avg_time * 1e9))  # Convert to GFLOPS
#
#         print(f"\nKết quả benchmark với ma trận {N}x{N} trên {size} processes:")
#         print(f"Thời gian trung bình: {avg_time:.4f} giây")
#         print(f"Độ lệch chuẩn: {std_time:.4f} giây")
#         print(f"Thời gian nhanh nhất: {min_time:.4f} giây")
#         print(f"Thời gian chậm nhất: {max_time:.4f} giây")
#         print(f"Hiệu năng: {gflops:.2f} GFLOPS")
#
#         # Tính speedup và hiệu suất nếu có số liệu tuần tự
#         sequential_time = run_sequential_multiplication(N)
#         if sequential_time:
#             speedup = sequential_time / avg_time
#             efficiency = speedup / size
#             print(f"Speedup: {speedup:.2f}x")
#             print(f"Hiệu suất: {efficiency:.2%}")
#
#
# def run_sequential_multiplication(N):
#     """Chạy nhân ma trận tuần tự để so sánh"""
#     A = np.random.randint(0, 10, size=(N, N))
#     B = np.random.randint(0, 10, size=(N, N))
#
#     start_time = time.time()
#     C = np.dot(A, B)
#     end_time = time.time()
#
#     return end_time - start_time
#
#
# if __name__ == "__main__":
#     # Test với các kích thước ma trận khác nhau
#     matrix_sizes = [480, 960, 1920]  # Phải chia hết cho sqrt(số process)
#     for N in matrix_sizes:
#         benchmark_fox_algorithm(N)

# from mpi4py import MPI
# import numpy as np
# import math
#
# def matrix_creation(size):
#     A = np.random.randint(1, 1000, size=(size, size)).astype('float64')
#     B = np.random.randint(1, 1000, size=(size, size)).astype('float64')
#     C = np.zeros((size, size), dtype='float64')
#     return A, B, C
#
# def FoxAlgorithm(A, B, C, m_size, n_blocks, block_size, comm):
#     rank = comm.Get_rank()
#     size = comm.Get_size()
#
#     # Determine grid position of the process
#     i1 = rank // n_blocks
#     j1 = rank % n_blocks
#
#     # Scatter A and B blocks among processes
#     local_A = np.empty((block_size, block_size), dtype='float64')
#     local_B = np.empty((block_size, block_size), dtype='float64')
#     local_C = np.zeros((block_size, block_size), dtype='float64')
#
#     for stage in range(n_blocks):
#         # Broadcast the required row block of A to all processes in the row
#         root = (i1 + stage) % n_blocks
#         if j1 == root:
#             local_A = A[i1 * block_size:(i1 + 1) * block_size, (i1 + stage) % n_blocks * block_size:(i1 + stage + 1) % n_blocks * block_size]
#         comm.Bcast(local_A, root=root)
#
#         # Get the appropriate block of B
#         local_B = B[((i1 + stage) % n_blocks) * block_size:((i1 + stage + 1) % n_blocks) * block_size, j1 * block_size:(j1 + 1) * block_size]
#
#         # Perform block matrix multiplication
#         local_C += np.dot(local_A, local_B)
#
#     # Gather the result into C
#     comm.Gather(local_C, C if rank == 0 else None, root=0)
#
# def main():
#     comm = MPI.COMM_WORLD
#     rank = comm.Get_rank()
#     size = comm.Get_size()
#
#     # Define matrix size and calculate block sizes
#     m_size = 100
#     n_blocks = int(math.sqrt(size))
#     if m_size % n_blocks != 0:
#         if rank == 0:
#             print("Matrix size must be divisible by the number of blocks.")
#         MPI.Finalize()
#         return
#
#     block_size = m_size // n_blocks
#
#     # Initialize matrices A, B, and C
#     if rank == 0:
#         A, B, C = matrix_creation(m_size)
#     else:
#         A = B = C = None
#
#     # Broadcast matrix size and create local arrays
#     if rank != 0:
#         A = np.empty((m_size, m_size), dtype='float64')
#         B = np.empty((m_size, m_size), dtype='float64')
#         C = np.empty((m_size, m_size), dtype='float64')
#
#     # Broadcast the matrices A and B to all processes
#     comm.Bcast(A, root=0)
#     comm.Bcast(B, root=0)
#
#     # Perform Fox algorithm
#     start_time = MPI.Wtime()
#     FoxAlgorithm(A, B, C, m_size, n_blocks, block_size, comm)
#     end_time = MPI.Wtime() - start_time
#
#     # Output results
#     if rank == 0:
#         print(f"FoxAlgorithm_time: {end_time:.5f} seconds, n_threads: {size}, m_size: {m_size}")
#
# if __name__ == "__main__":
#     main()

# from mpi4py import MPI
# import numpy as np
# import math
#
# def input_matrix(size, name, rank):
#     matrix = None
#     if rank == 0:
#         print(f"Nhập các phần tử cho ma trận {name} ({size}x{size}):")
#         matrix = np.zeros((size, size), dtype='float64')
#         for i in range(size):
#             row = input(f"Dòng {i + 1}: ").split()
#             matrix[i] = [float(x) for x in row]
#     return matrix
#
# def FoxAlgorithm(A, B, C, m_size, n_blocks, block_size, comm):
#     rank = comm.Get_rank()
#     i1 = rank // n_blocks  # hàng lưới của tiến trình
#     j1 = rank % n_blocks   # cột lưới của tiến trình
#
#     # Khởi tạo ma trận cục bộ cho A, B, và C
#     local_A = np.zeros((block_size, block_size), dtype='float64')
#     local_B = np.zeros((block_size, block_size), dtype='float64')
#     local_C = np.zeros((block_size, block_size), dtype='float64')
#
#     for stage in range(n_blocks):
#         # Tính root của broadcast
#         root = (i1 + stage) % n_blocks
#         # Broadcast khối A cần thiết
#         if j1 == root:
#             local_A = A[i1 * block_size:(i1 + 1) * block_size, root * block_size:(root + 1) * block_size]
#         comm.Bcast(local_A, root=root)
#
#         # Lấy khối B cần thiết
#         local_B = B[root * block_size:(root + 1) * block_size, j1 * block_size:(j1 + 1) * block_size]
#
#         # Thực hiện nhân khối và cộng dồn vào C
#         local_C += np.dot(local_A, local_B)
#
#     return local_C
#
# def main():
#     comm = MPI.COMM_WORLD
#     rank = comm.Get_rank()
#     size = comm.Get_size()
#
#     # Chỉ tiến trình 0 nhập kích thước ma trận từ người dùng
#     if rank == 0:
#         m_size = int(input("Nhập kích thước ma trận vuông (m_size): "))
#     else:
#         m_size = None
#
#     # Phát kích thước ma trận đến các tiến trình
#     m_size = comm.bcast(m_size, root=0)
#     n_blocks = int(math.sqrt(size))
#
#     # Kiểm tra điều kiện ma trận chia đều cho số tiến trình
#     if m_size % n_blocks != 0:
#         if rank == 0:
#             print("Kích thước ma trận phải chia hết cho căn bậc hai của số tiến trình.")
#         MPI.Finalize()
#         return
#
#     block_size = m_size // n_blocks
#
#     # Chỉ tiến trình 0 nhập ma trận A và B
#     if rank == 0:
#         A = input_matrix(m_size, "A", rank)
#         B = input_matrix(m_size, "B", rank)
#         C = np.zeros((m_size, m_size), dtype='float64')
#     else:
#         A = None
#         B = None
#         C = None
#
#     # Phát ma trận A và B đến các tiến trình
#     A = comm.bcast(A, root=0)
#     B = comm.bcast(B, root=0)
#
#     # Bắt đầu đo thời gian
#     comm.Barrier()
#     start_time = MPI.Wtime()
#
#     # Chạy thuật toán Fox
#     local_C = FoxAlgorithm(A, B, C, m_size, n_blocks, block_size, comm)
#
#     # Thu thập kết quả vào ma trận C (chỉ trên tiến trình 0)
#     gathered_C = comm.gather(local_C, root=0)
#
#     # Tạo ma trận C từ các khối đã thu thập
#     if rank == 0:
#         for i in range(n_blocks):
#             for j in range(n_blocks):
#                 C[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = gathered_C[i*n_blocks + j]
#
#         # Kết thúc đo thời gian
#         end_time = MPI.Wtime() - start_time
#
#         # Hiển thị kết quả
#         print(f"\nThời gian chạy thuật toán Fox: {end_time:.5f} giây")
#         # print("\nMa trận A:")
#         # print(A)
#         # print("\nMa trận B:")
#         # print(B)
#         print("\nMa trận tích C = A * B:")
#         print(C)
#
# if __name__ == "__main__":
#     main()


# from mpi4py import MPI
# import numpy as np
# import math
# def read_matrix_from_file(filename, rank):
#     if rank == 0:
#         with open(filename, 'r') as f:
#             # Đọc kích thước ma trận
#             m_size = int(f.readline().strip())
#             # Khởi tạo ma trận
#             A = np.zeros((m_size, m_size), dtype='float64')
#             B = np.zeros((m_size, m_size), dtype='float64')
#             # Đọc ma trận A
#             for i in range(m_size):
#                 row = f.readline().strip().split()
#                 A[i] = [float(x) for x in row]
#             # Đọc ma trận B
#             for i in range(m_size):
#                 row = f.readline().strip().split()
#                 B[i] = [float(x) for x in row]
#         return m_size, A, B
#     else:
#         return None, None, None
# def FoxAlgorithm(A, B, C, m_size, n_blocks, block_size, comm):
#     rank = comm.Get_rank()
#     i1 = rank // n_blocks  # hàng lưới của tiến trình
#     j1 = rank % n_blocks   # cột lưới của tiến trình
#     # Khởi tạo ma trận cục bộ cho A, B, và C
#     local_A = np.zeros((block_size, block_size), dtype='float64')
#     local_B = np.zeros((block_size, block_size), dtype='float64')
#     local_C = np.zeros((block_size, block_size), dtype='float64')
#     for stage in range(n_blocks):
#         # Xác định root để broadcast
#         root = (i1 + stage) % n_blocks
#         # Broadcast khối A cần thiết
#         if j1 == root:
#             local_A = A[i1 * block_size:(i1 + 1) * block_size, root * block_size:(root + 1) * block_size]
#         comm.Bcast(local_A, root=root)
#         # Lấy khối B cần thiết
#         local_B = B[root * block_size:(root + 1) * block_size, j1 * block_size:(j1 + 1) * block_size]
#         # Thực hiện nhân khối và cộng dồn vào C
#         local_C += np.dot(local_A, local_B)
#     return local_C
# def main():
#     comm = MPI.COMM_WORLD
#     rank = comm.Get_rank()
#     size = comm.Get_size()
#     # Chỉ tiến trình 0 đọc ma trận từ file
#     if rank == 0:
#         # filename = input("Nhập tên file chứa ma trận A và B: ")
#         m_size, A, B = read_matrix_from_file("fox_1000.txt", rank)
#         C = np.zeros((m_size, m_size), dtype='float64')
#     else:
#         m_size, A, B, C = None, None, None, None
#     # Phát kích thước ma trận đến các tiến trình
#     m_size = comm.bcast(m_size, root=0)
#     n_blocks = int(math.sqrt(size))
#     # Kiểm tra nếu kích thước ma trận chia hết cho căn bậc hai của số tiến trình
#     if m_size % n_blocks != 0:
#         if rank == 0:
#             print("Kích thước ma trận phải chia hết cho căn bậc hai của số tiến trình.")
#         MPI.Finalize()
#         return
#     block_size = m_size // n_blocks
#     A = comm.bcast(A, root=0)
#     B = comm.bcast(B, root=0)
#     comm.Barrier()
#     start_time = MPI.Wtime()
#     # Thực hiện thuật toán Fox
#     local_C = FoxAlgorithm(A, B, C, m_size, n_blocks, block_size, comm)
#     # Thu thập kết quả vào ma trận C (chỉ trên tiến trình 0)
#     gathered_C = comm.gather(local_C, root=0)
#     # Tạo ma trận C từ các khối đã thu thập
#     if rank == 0:
#         for i in range(n_blocks):
#             for j in range(n_blocks):
#                 C[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size] = gathered_C[i * n_blocks + j]
#         # Kết thúc đo thời gian
#         end_time = MPI.Wtime() - start_time
#         # Hiển thị kết quả
#         print(f"\nThời gian chạy thuật toán Fox: {end_time:.5f} giây")
#         # print("\nMa trận C = A * B:")
#         # print(C)
# if __name__ == "__main__":
#     main()

from mpi4py import MPI
import numpy as np
import math

def read_matrix_from_file(filename, rank):
    if rank == 0:
        with open(filename, 'r') as f:
            # Đọc kích thước ma trận
            m_size = int(f.readline().strip())
            # Khởi tạo ma trận
            A = np.zeros((m_size, m_size), dtype='float64')
            B = np.zeros((m_size, m_size), dtype='float64')
            # Đọc ma trận A
            for i in range(m_size):
                row = f.readline().strip().split()
                A[i] = [float(x) for x in row]
            # Đọc ma trận B
            for i in range(m_size):
                row = f.readline().strip().split()
                B[i] = [float(x) for x in row]
        return m_size, A, B
    else:
        return None, None, None

def FoxAlgorithm(A, B, C, m_size, n_blocks, block_size, comm):
    rank = comm.Get_rank()
    i1 = rank // n_blocks  # hàng lưới của tiến trình
    j1 = rank % n_blocks   # cột lưới của tiến trình
    # Khởi tạo ma trận cục bộ cho A, B, và C
    local_A = np.zeros((block_size, block_size), dtype='float64')
    local_B = np.zeros((block_size, block_size), dtype='float64')
    local_C = np.zeros((block_size, block_size), dtype='float64')
    for stage in range(n_blocks):
        # Xác định root để broadcast
        root = (i1 + stage) % n_blocks
        # Broadcast khối A cần thiết
        if j1 == root:
            local_A = A[i1 * block_size:(i1 + 1) * block_size, root * block_size:(root + 1) * block_size]
        comm.Bcast(local_A, root=root)
        # Lấy khối B cần thiết
        local_B = B[root * block_size:(root + 1) * block_size, j1 * block_size:(j1 + 1) * block_size]
        # Thực hiện nhân khối và cộng dồn vào C
        local_C += np.dot(local_A, local_B)
    return local_C

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    # Chỉ tiến trình 0 đọc ma trận từ file
    if rank == 0:
        # filename = input("Nhập tên file chứa ma trận A và B: ")
        m_size, A, B = read_matrix_from_file("fox_1000.txt.txt", rank)
        C = np.zeros((m_size, m_size), dtype='float64')
    else:
        m_size, A, B, C = None, None, None, None
    # Phát kích thước ma trận đến các tiến trình
    m_size = comm.bcast(m_size, root=0)
    n_blocks = int(math.sqrt(size))
    # Kiểm tra nếu kích thước ma trận chia hết cho căn bậc hai của số tiến trình
    if m_size % n_blocks != 0:
        if rank == 0:
            print("Kích thước ma trận phải chia hết cho căn bậc hai của số tiến trình.")
        MPI.Finalize()
        return
    block_size = m_size // n_blocks
    # Phát ma trận A và B đến các tiến trình
    A = comm.bcast(A, root=0)
    B = comm.bcast(B, root=0)
    # Bắt đầu đo thời gian
    comm.Barrier()
    start_time = MPI.Wtime()
    # Thực hiện thuật toán Fox
    local_C = FoxAlgorithm(A, B, C, m_size, n_blocks, block_size, comm)
    # Thu thập kết quả vào ma trận C (chỉ trên tiến trình 0)
    gathered_C = comm.gather(local_C, root=0)
    # Tạo ma trận C từ các khối đã thu thập
    if rank == 0:
        for i in range(n_blocks):
            for j in range(n_blocks):
                C[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size] = gathered_C[i * n_blocks + j]
        # Kết thúc đo thời gian
        end_time = MPI.Wtime() - start_time
        # Hiển thị kết quả
        print(f"\nThời gian chạy thuật toán Fox: {end_time:.5f} giây")
        # print("\nMa trận C = A * B:")
        # print(C)

if __name__ == "__main__":
    main()






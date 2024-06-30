import numpy as np
import struct
import numba
from numba import cuda

# CUDAカーネル関数を定義
@cuda.jit
def vector_addition(a, b, c):
    idx = cuda.grid(1)
    if idx < a.size:
        c[idx] = a[idx] + b[idx]

class DynamicUnifiedMemoryProcessor:
    def __init__(self, initial_size):
        self.memory_size = initial_size
        self.memory = np.zeros(self.memory_size, dtype=np.uint8)
    
    def allocate_memory(self, new_size):
        if new_size > self.memory_size:
            new_memory = np.zeros(new_size, dtype=np.uint8)
            new_memory[:self.memory_size] = self.memory
            self.memory = new_memory
            self.memory_size = new_size
        else:
            self.memory = self.memory[:new_size]
            self.memory_size = new_size
    
    def write_data(self, address, data):
        if address + len(data) > self.memory_size:
            self.allocate_memory(address + len(data))
        self.memory[address:address+len(data)] = np.frombuffer(data, dtype=np.uint8)
    
    def read_data(self, address, length):
        if address + length > self.memory_size:
            raise MemoryError("Reading beyond virtual memory size")
        return self.memory[address:address+length].tobytes()
    
    def write_int(self, address, value):
        self.write_data(address, struct.pack('I', value))
    
    def read_int(self, address):
        return struct.unpack('I', self.read_data(address, 4))[0]
    
    def process_with_gpu(self, address1, address2, result_address, length):
        # メモリからデータを読み込む
        data1 = np.frombuffer(self.read_data(address1, length * 4), dtype=np.float32)
        data2 = np.frombuffer(self.read_data(address2, length * 4), dtype=np.float32)
        
        # Unified Memoryを使用してメモリを確保
        d_a = cuda.to_device(data1)
        d_b = cuda.to_device(data2)
        d_c = cuda.device_array(length, dtype=np.float32)
        
        # スレッドとブロックの設定
        threads_per_block = 256
        blocks_per_grid = (length + (threads_per_block - 1)) // threads_per_block
        
        # CUDAカーネルを呼び出し
        vector_addition[blocks_per_grid, threads_per_block](d_a, d_b, d_c)
        
        # 結果をホストにコピー
        result_data = d_c.copy_to_host()
        
        # メモリに結果を書き込む
        self.write_data(result_address, result_data.tobytes())

# 使用例
processor = DynamicUnifiedMemoryProcessor(1024)  # 初期メモリサイズ1KB

# メモリにデータを書き込む
data1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
data2 = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)

processor.write_data(0, data1.tobytes())
processor.write_data(16, data2.tobytes())

# GPUでデータを処理
processor.process_with_gpu(0, 16, 32, 4)

# 結果を読み込む
result = np.frombuffer(processor.read_data(32, 16), dtype=np.float32)
print("Processed data:", result)  # 出力: Processed data: [ 6.  8. 10. 12.]
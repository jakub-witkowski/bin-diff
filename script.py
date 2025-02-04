import os
import math
from scipy.fft import fft, ifft
import numpy as np
import tracemalloc

buffer_size = 4**10*8

filename1 = 'sd_xl_base_1.0.safetensors'
filename2 = 'sd_xl_base_1.0_0.9vae.safetensors'

# assumes the files for comparison are the same length
file_size = os.path.getsize(filename1);
last_round_buffer_size = file_size - (buffer_size * math.floor(file_size/buffer_size))

if (file_size % buffer_size == 0):
    chunks_to_read = file_size / buffer_size
else:
    chunks_to_read = math.ceil(file_size / buffer_size)

def determine_header_size(filename):

    with open(filename, 'rb') as f:
        n = f.read(8)
    
    header_size = int.from_bytes(n, "little")

    return header_size

def headers_excluded_read_stepwise_and_compare(filename1, filename2, buffer_size, header_size):

    diff = 0
    chunks_read = 0
    positions = []

    with open(filename1, 'rb') as f1, open(filename2, 'rb') as f2:
        chunk1 = f1.read(buffer_size)
        chunk2 = f2.read(buffer_size)

        while (chunks_read < chunks_to_read):
            file1_chunk_content = bytes(chunk1)
            file2_chunk_content = bytes(chunk2)

            if (chunks_read == chunks_to_read - 1):
                limit = last_round_buffer_size
            else:
                limit = buffer_size

            for i in range(limit):
                if chunks_read == 0 and i < header_size:
                    continue
                else:
                    if file1_chunk_content[i] != file2_chunk_content[i]:
                        diff+=1
                        positions.append(i)

            chunks_read+=1

            print("Finished reading chunk %d, found %d differences thus far." %(chunks_read, diff))

            if (file_size % buffer_size == 0):
                chunk1 = f1.read(buffer_size)
                chunk2 = f2.read(buffer_size)
            else:
                if (chunks_read == (int)(file_size / buffer_size)):
                    chunk1 = f1.read(last_round_buffer_size)
                    chunk2 = f2.read(last_round_buffer_size)
                else:
                    chunk1 = f1.read(buffer_size)
                    chunk2 = f2.read(buffer_size)

    return positions
    # print("Read %d chunks, found %d byte differences." %(chunks_read, diff))

# header_size = determine_header_size(filename1)
print("Header size: %d bytes" % determine_header_size(filename1))
print("Header size: %d bytes" % determine_header_size(filename2))

header_size1 = determine_header_size(filename1)
header_size2 = determine_header_size(filename2)

header_size = header_size1 if header_size1 > header_size2 else header_size2 if header_size1 else header_size1

# print("File size: %d bytes, %d chunks to read." %(file_size, chunks_to_read))
# read_stepwise_and_compare(filename1, filename2, buffer_size, header_size)
# print(read_stepwise_and_compare(filename1, filename2, buffer_size, header_size))

# data = read_stepwise_and_compare(filename1, filename2, buffer_size, header_size)

# code or function for which memory
# has to be monitored
def app():
    input = np.array(headers_excluded_read_stepwise_and_compare(filename1, filename2, buffer_size, header_size))
    y = fft(input)

    print("The resultant Fourier transform:")
    print(y)

# starting the monitoring
tracemalloc.start()

# function call
app()

# displaying the memory
print(tracemalloc.get_traced_memory())

# stopping the library
tracemalloc.stop()
import logging as log
import sys
from typing import IO, Any, List, NamedTuple

import numpy as np

class FileData(NamedTuple):
    keys: List[str]
    utterances: List[np.ndarray]


def read_ark_file(file_name: str) -> FileData:
    def read_key(input_file: IO[Any]) -> str:
        key = ''
        char = input_file.read(1).decode()

        while char not in ('', ' '):
            key += char
            char = input_file.read(1).decode()

        return key

    def read_matrix(input_file: IO[Any]) -> np.ndarray:
        header = input_file.read(5).decode()
        if 'FM' in header:
            num_of_bytes = 4
            dtype = 'float32'
        elif 'DM' in header:
            num_of_bytes = 8
            dtype = 'float64'
        else:
            log.error(f'The utterance header "{header}" does not contain information about a type of elements.')
            sys.exit(-7)

        _, rows, _, cols = np.frombuffer(input_file.read(10), 'int8, int32, int8, int32')[0]
        buffer = input_file.read(rows * cols * num_of_bytes)
        vector = np.frombuffer(buffer, dtype)
        matrix = np.reshape(vector, (rows, cols))

        return matrix

    keys = []
    utterances = []
    with open(file_name, 'rb') as input_file:
        key = read_key(input_file)

        while key:
            utterances.append(read_matrix(input_file))
            keys.append(key)
            key = read_key(input_file)

    return FileData(keys, utterances)


def write_ark_file(file_name: str, keys: List[str], utterances: List[np.ndarray]):
    with open(file_name, 'wb') as output_file:
        for key, matrix in zip(keys, utterances):
            output_file.write(key.encode())
            output_file.write(' '.encode())
            output_file.write('\0B'.encode())

            if matrix.dtype == 'float32':
                output_file.write('FM '.encode())
            elif matrix.dtype == 'float64':
                output_file.write('DM '.encode())

            output_file.write('\04'.encode())
            output_file.write(matrix.shape[0].to_bytes(4, byteorder='little', signed=False))
            output_file.write('\04'.encode())
            output_file.write(matrix.shape[1].to_bytes(4, byteorder='little', signed=False))

            output_file.write(matrix.tobytes())


def read_utterance_file(file_name: str) -> FileData:
    file_extension = file_name.split('.')[-1]

    if file_extension == 'ark':
        return read_ark_file(file_name)
    elif file_extension == 'npz':
        data = dict(np.load(file_name))
        return FileData(list(data.keys()), list(data.values()))
    else:
        log.error(f'The file {file_name} cannot be read. The sample supports only .ark and .npz files.')
        sys.exit(-1)


def write_utterance_file(file_name: str, keys: List[str], utterances: List[np.ndarray]):
    file_extension = file_name.split('.')[-1]

    if file_extension == 'ark':
        write_ark_file(file_name, keys, utterances)
    elif file_extension == 'npz':
        np.savez(file_name, **dict(zip(keys, utterances)))
    else:
        log.error(f'The file {file_name} cannot be written. The sample supports only .ark and .npz files.')
        sys.exit(-2)

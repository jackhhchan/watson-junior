import numpy as np


##### INDEX COMPRESSION ####
def int_encode(num):
    """ Returns the optimal size integer for the give number """
    assert num >= 0, "Index must be positive."
    uint_types = [np.uint8, np.uint16, np.uint32, np.uint64]
    if num <= 255:
        return uint_types[0](num)       # 1 byte
    elif num > 255 and num <= 65535:
        return uint_types[1](num)       # 2 bytes
    elif num > 65535 and num <= 4294967295:
        return uint_types[2](num)       # 4 bytes
    elif num > 4294967295 and num <= 18446744073709551615:
        return uint_types[3](num)       # 8 bytes
    else:
        return num                      # typically 32 bytes or larger (standard python ints size)




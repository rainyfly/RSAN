import os
from data import multiscalesrdata

class DIV2K(multiscalesrdata.SRData):
    def __init__(self, args, name='DIV2K', train=True, benchmark=False):
        super(DIV2K, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _set_filesystem(self, dir_data):
        super(DIV2K, self)._set_filesystem(dir_data)


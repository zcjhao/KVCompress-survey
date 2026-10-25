"""One file per KV-cache compression method, looked up by common.build_press().

`--press <name>` loads presses/<name>.py and calls its build(compression_ratio), which returns a
kvpress press. Use a lowercase name with underscores (it must be a valid Python module name).
For example, presses/snapkv.py:

    from kvpress import SnapKVPress


    def build(compression_ratio: float):
        return SnapKVPress(compression_ratio=compression_ratio)

"none" (the uncompressed baseline) is built into build_press() and has no file here.
"""

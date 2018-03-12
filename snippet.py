from onnx.helper import make_attribute

attr = make_attribute('test', [1, 2])
print(attr.ints)
# print(all([isinstance(i, long) for i in attr.ints]))
print(all([isinstance(i, int) for i in attr.ints]))

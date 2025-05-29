from verl import DataProto

data1 = {}

test_batch16 = DataProto.from_single_dict(data1)

if test_batch16 is None:
    print("test_batch16 is None")
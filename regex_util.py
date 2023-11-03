import re

def plate_regex(test_string):
    pattern = r'\d{2}[가-힣]\d{4}'
    result = re.findall(pattern, test_string)
    return result

test_sting1 = '61보5612'
test_sting2 = '(61보56120'
test_sting3 = '6145612'
test_sting4 = '91[5612'

plate_regex(test_sting1)
plate_regex(test_sting2)
plate_regex(test_sting3)
plate_regex(test_sting4)
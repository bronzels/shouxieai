from pprint import pprint
from paddlenlp import Taskflow

schema = ['时间', '选手', '赛事名称']
ie = Taskflow('information_extraction', schema=schema)
pprint(ie("2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！"))
'''
[{'时间': [{'end': 6,
          'probability': 0.9857378532473966,
          'start': 0,
          'text': '2月8日上午'}],
  '赛事名称': [{'end': 23,
            'probability': 0.8503083858300187,
            'start': 6,
            'text': '北京冬奥会自由式滑雪女子大跳台决赛'}],
  '选手': [{'end': 31,
          'probability': 0.898153923148989,
          'start': 28,
          'text': '谷爱凌'}]}]

'''
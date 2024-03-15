from wordcloud import WordCloud
import matplotlib.pyplot as plt

text = open("test.txt", encoding="utf-8").read()
wc = WordCloud(font_path="simhei.ttf", width=800, height=600, mode="RGBA", background_color=None).generate(text)
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()
wc.to_file("2.worldcloud2.png")

import jieba
text = ' '.join(jieba.cut(text))
print(text[:100])
wc = WordCloud(font_path="simhei.ttf", width=800, height=600, mode="RGBA", background_color=None).generate(text)
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()
wc.to_file("2.2worldcloud2.png")

import numpy as np
from PIL import Image
mask = np.array(Image.open("3.alice3.png"))
wc = WordCloud(mask=mask, font_path="simhei.ttf", width=800, height=600, mode="RGBA", background_color=None).generate(text)
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()
wc.to_file("3.2worldcloud2.png")


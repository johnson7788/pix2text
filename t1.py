from pix2text import Pix2Text

img_fp = './docs/examples/formula.jpg'
p2t = Pix2Text()
out_text = p2t(img_fp)  # 也可以使用 `p2t.recognize(img_fp)` 获得相同的结果
print(out_text)

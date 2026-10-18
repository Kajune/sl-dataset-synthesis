import random
from PIL import Image, ImageEnhance, ImageOps, ImageFilter

def augment_image(img):
	if img.mode != "RGB":
		img = img.convert("RGB")

	# スケーリング
	if random.random() < 0.5:
		scale_x = random.uniform(0.7, 1.3)
		scale_y = random.uniform(0.7, 1.3)
		original_width = img.width
		original_height = img.height
		new_w = int(img.width * scale_x)
		new_h = int(img.height * scale_y)
		img = img.resize((new_w, new_h), Image.BICUBIC)
		img = img.resize((original_width, original_height), Image.BICUBIC)

	# 明るさ
	if random.random() < 0.5:
		enhancer = ImageEnhance.Brightness(img)
		img = enhancer.enhance(random.uniform(0.1, 1.0))
	
	if random.random() < 0.5:
		enhancer = ImageEnhance.Contrast(img)
		img = enhancer.enhance(random.uniform(0.5, 1.5))

	# 彩度
	if random.random() < 0.5:
		enhancer = ImageEnhance.Color(img)
		img = enhancer.enhance(random.uniform(0.5, 1.5))

	# シャープネス
	if random.random() < 0.5:
		enhancer = ImageEnhance.Sharpness(img)
		img = enhancer.enhance(random.uniform(0.5, 2.0))

	# RGBゲイン
	if random.random() < 0.5:
		r_gain = random.uniform(0.8, 1.2)
		g_gain = random.uniform(0.8, 1.2)
		b_gain = random.uniform(0.8, 1.2)
		r, g, b = img.split()
		r = r.point(lambda i: i * r_gain)
		g = g.point(lambda i: i * g_gain)
		b = b.point(lambda i: i * b_gain)
		img = Image.merge("RGB", (r, g, b))

	# 上下・左右反転
	if random.random() < 0.5:
		img = ImageOps.mirror(img)
	if random.random() < 0.5:
		img = ImageOps.flip(img)

	# ピンボケ（ブラー）
	if random.random() < 0.5:
		radius = random.uniform(0.5, 2.0)
		img = img.filter(ImageFilter.GaussianBlur(radius))

	# 色収差（カラーチャンネルごとに微小にtranslate）
	if random.random() < 0.5:
		max_shift = 3  # ピクセル単位のシフト量
		r, g, b = img.split()
		r = r.transform(r.size, Image.AFFINE, (1, 0, random.randint(-max_shift, max_shift),
											   0, 1, random.randint(-max_shift, max_shift)))
		g = g.transform(g.size, Image.AFFINE, (1, 0, random.randint(-max_shift, max_shift),
											   0, 1, random.randint(-max_shift, max_shift)))
		b = b.transform(b.size, Image.AFFINE, (1, 0, random.randint(-max_shift, max_shift),
											   0, 1, random.randint(-max_shift, max_shift)))
		img = Image.merge("RGB", (r, g, b))

	return img


if __name__ == '__main__':
	import sys
	img = Image.open(sys.argv[1])
	img = augment_image(img)
	img.save("result.png")

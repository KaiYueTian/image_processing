# image_processing
# assignment_1:暗通道去雾
	land.jpg：原始图像
	defog.jpg：去雾图像
# assignment_2:维纳滤波
	019.jpg：原图
	result.png：六张图片分别代表-添加运动模糊、信噪比已知、信噪比未知；添加随机噪声、信噪比已知、信噪比未知
# 2.3.2：不同空间分辨率和灰度分辨率处理lena图像
	对比图.png：最终结果图对比
	lena.jpg:原图  x_lena.jpg：分辨率为x  resultx.jpg：灰度级为x
# 3.1.2：点运算 对灰度图进行线性、分段线性、非线性点运算
	gray_lena.jpg：初始图片
	Linear.jpg：线性点运算后的图片  y = x +55
	PiecewiseLinear.jpg：分段线性运算后的图片 
		x<60:y = 0
		x>160:y = 255
		60<=x<=160:y = x+55
	ExpTran.jpg：非线性点运算后的图片
		三次幂运算
	result.jpg：四张图片对比 
# 3.1.4：集合运算 对图片实现 平移、旋转、镜像操作，以及复合运算
	使用的是lena.jpg作为初始图
	img_H.jpg:水平镜像  img_V.jpg:垂直镜像  img_HV.jpg:对角镜像   img_pan.jpg:向左向上平移50px  img_rotate.jpg:顺时针旋转45度   
	对比图.jpg:各种效果对比
	result.jpg：复合结果 - 水平镜像-顺时针旋转45度-向左向上平移50px-垂直镜像
# 8.1.2：自适应阈值
	test.jpg：原图
	OTSU.jpg：大律法
	iterate.jpg：迭代法

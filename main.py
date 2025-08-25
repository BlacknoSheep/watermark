import cv2
import numpy as np
import pymupdf  # PyMuPDF
import os


def imread_unicode(path: str):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)


def imwrite_unicode(path: str, img) -> bool:
    ext = os.path.splitext(path)[1]  # 文件扩展名
    success, buf = cv2.imencode(ext, img)
    if success:
        buf.tofile(path)
    return success


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    """将 BGR 或 BGRA 格式转为 RGB 或 RGBA 格式"""
    if img.ndim == 3:  # 彩色图像
        if img.shape[2] == 3:  # BGR -> RGB
            return img[:, :, ::-1]
        elif img.shape[2] == 4:  # BGRA -> RGBA
            return img[:, :, [2, 1, 0, 3]]
    return img


def cv2img_to_pixmap(cv_img: cv2.typing.MatLike) -> pymupdf.Pixmap:
    cv_img = bgr_to_rgb(cv_img)
    h, w, c = cv_img.shape
    pix = pymupdf.Pixmap(pymupdf.csRGB, w, h, cv_img.tobytes(), (c == 4))
    return pix


def find_max_white_rect(
    image: cv2.typing.MatLike, threshold=240
) -> tuple[float, float, float, float]:
    """通过直方图内最大矩形算法找到图像中面积最大的白色矩形区域"""
    # 二值化图片
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    h, w = binary_img.shape
    hist = np.zeros((h, w), dtype=int)

    # Calculate the height of consecutive white pixels for each column
    for i in range(h):
        for j in range(w):
            if binary_img[i, j] == 255:
                hist[i, j] = hist[i - 1, j] + 1 if i > 0 else 1
            else:
                hist[i, j] = 0

    # 逐行做直方图内最大矩形算法
    max_area, max_rect = 0, (0, 0, 0, 0)
    for i in range(h):
        stack = []
        j = 0
        while j <= w:
            cur_height = hist[i, j] if j < w else 0
            if not stack or cur_height >= hist[i, stack[-1]]:
                stack.append(j)
                j += 1
            else:
                top = stack.pop()
                width = j if not stack else j - stack[-1] - 1
                area = hist[i, top] * width
                if area > max_area:
                    max_area = area
                    x = (stack[-1] + 1) if stack else 0
                    max_rect = (x, i - hist[i, top] + 1, width, hist[i, top])

    return max_rect  # (x, y, w, h)


def add_watermark(
    image: cv2.typing.MatLike,
    pdf_page: pymupdf.Page,
    watermark: cv2.typing.MatLike,
    max_size_ratio: float,
):
    """
    同时向图片和PDF页面添加水印
    """
    img_h, img_w = image.shape[:2]  # 图片的宽高
    pdf_w, pdf_h = pdf_page.rect.width, pdf_page.rect.height  # PDF页面的宽高

    # 找到最大空白矩形区域
    x, y, w, h = find_max_white_rect(image)

    # 计算水印的尺寸
    wm_h, wm_w = watermark.shape[:2]
    if max_size_ratio > 0:
        scale = min(
            w / wm_w,
            h / wm_h,
            (img_w * max_size_ratio) / wm_w,
            (img_h * max_size_ratio) / wm_h,
        )
    else:
        scale = min(w / wm_w, h / wm_h)
    wm_h, wm_w = (int(wm_h * scale), int(wm_w * scale))

    # 计算水印的位置（居中）
    x_img, y_img = (
        x + (w - wm_w) // 2,
        y + (h - wm_h) // 2,
    )  # 图片上水印左上角坐标
    watermark = cv2.resize(watermark, (wm_w, wm_h))

    # 将水印加到图片上（考虑透明通道）
    if watermark.shape[2] == 4:
        alpha_s = watermark[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        for c in range(3):
            image[y_img : y_img + wm_h, x_img : x_img + wm_w, c] = (
                alpha_s * watermark[:, :, c]
                + alpha_l * image[y_img : y_img + wm_h, x_img : x_img + wm_w, c]
            )
    else:
        image[y_img : y_img + wm_h, x_img : x_img + wm_w] = watermark

    # 将水印加到PDF页面上
    # 计算PDF上水印的坐标（左上，右下）
    x0, y0 = x_img * pdf_w / img_w, y_img * pdf_h / img_h
    x1, y1 = (x_img + wm_w) * pdf_w / img_w, (y_img + wm_h) * pdf_h / img_h
    rect = pymupdf.Rect(x0, y0, x1, y1)
    pdf_page.insert_image(
        rect, pixmap=cv2img_to_pixmap(watermark), keep_proportion=True
    )


def work(
    pdf_path: str,
    watermark_path: str,
    output_pdf_path: str,
    cnt_watermark: int = 1,
    max_size_ratio: float = 0,
):
    """
    向PDF文件的每一页添加水印
    """
    doc = pymupdf.open(pdf_path)

    # watermark = cv2.imread(watermark_path, cv2.IMREAD_UNCHANGED)
    # 解决中文路径问题
    watermark = imread_unicode(watermark_path)

    if watermark is None:
        raise ValueError("Failed to load watermark image!")

    for page_number in range(len(doc)):
        page = doc[page_number]

        # 渲染当前页为图片
        pix = pymupdf.utils.get_pixmap(page, dpi=50)
        # 转化为OpenCV格式
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        # If the image has an alpha channel, remove it
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # 同时向图片和PDF页面添加水印，循环以添加多个水印
        for _ in range(0, cnt_watermark):
            add_watermark(img, page, watermark, max_size_ratio)

    doc.save(output_pdf_path)
    doc.close()


if __name__ == "__main__":
    pdf_path = ""
    watermark_path = ""
    output_pdf_path = ""
    cnt_watermark = 1  # 每页水印数量
    max_size_ratio = 0.2  # 水印宽高占页面宽高的最大比例，0表示不限制
    work(
        pdf_path,
        watermark_path,
        output_pdf_path,
        cnt_watermark,
        max_size_ratio,
    )
    print("Done!")

import cv2
import numpy as np


def loadcv2dnnNetONNX(onnx_path):
    net = cv2.dnn.readNetFromONNX(onnx_path)

    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    print('load successful')
    return net


def pad2square_cv2(image):
    h, w, c = image.shape
    dim_diff = np.abs(h - w)
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    if h <= w:
        image = cv2.copyMakeBorder(image, pad1, pad2, 0, 0,
                                   cv2.BORDER_CONSTANT, value=0)
    else:
        image = cv2.copyMakeBorder(image, 0, 0, pad1, pad2,
                                   cv2.BORDER_CONSTANT, value=0)
    return image


def preprocess4cv2dnn(image_path):
    image = cv2.imread(image_path)

    image = pad2square_cv2(image)

    h, w = image.shape[:2]

    # 干嘛要水平翻转？
    # image = cv2.flip(image, 1)

    blobImage = cv2.dnn.blobFromImage(image, 1.0 / 255.0, (640, 640),
                                      None, True, False)

    return blobImage, image


def nms(dets, thresh=0.3, conf=0.6):
    dets = dets[dets[:, 4] > conf, :]

    x0 = dets[:, 0] 
    y0 = dets[:, 1] 
    w1 = dets[:, 2] 
    h1 = dets[:, 3] 
    x1 = x0 - w1 / 2
    y1 = y0 - h1 / 2
    x2 = x0 + w1 / 2
    y2 = y0 + h1 / 2
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:  # 还有数据
        i = order[0]
        keep.append(i)
        if order.size == 1: break
        # 计算当前概率最大矩形框与其他矩形框的相交框的坐标
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # 计算相交框的面积
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 计算重叠度IOU：重叠面积/（面积1+面积2-重叠面积）
        IOU = inter / (areas[i] + areas[order[1:]] - inter)

        # 找到重叠度不高于阈值的矩形框索引
        left_index = (np.where(IOU <= thresh))[0]

        # 将order序列更新，由于前面得到的矩形框索引要比矩形框在原order序列中的索引小1，所以要把这个1加回来
        order = order[left_index + 1]

    return keep, dets


def draw(img, pred):
    img_ = img.copy()
    img_ = cv2.resize(img_, (640, 640))
    num1, num2 = pred.shape

    for i in range(num1):
        cv2.rectangle(img_, (int(pred[i, 0] - pred[i, 2] / 2), int(pred[i, 1] - pred[i, 3] / 2)),
                      (int(pred[i, 0] + pred[i, 2] / 2), int(pred[i, 1] + pred[i, 3] / 2)),
                      (170, 234, 242),
                      1, cv2.LINE_AA)
    return img_


def gen_result(outs, class_names, thresh=0.3, conf=0.6):
    outs = outs[0][0, :, :]
    keep, outs = nms(outs, thresh=0.3, conf=0.6)
    outs = outs[keep, :]
    num_results = outs.shape[0]
    r_dict = {'result': [{
        'bbox': {
            'x_center': float(outs[i, 0]),
            'y_center': float(outs[i, 1]),
            'w': float(outs[i, 2]),
            'h': float(outs[i, 3]), },
        'pr_obj': float(outs[i, 4]),
        'class': class_names[np.argmax(outs[i, 5:])],
        'pr_classes_obj': [float(ii) for ii in outs[i, 5:].tolist()],
    } for i in range(num_results)]}
    return r_dict


if __name__ == '__main__':
    n = loadcv2dnnNetONNX('models/yolov7.onnx')

    print(n)
    blobImage, image = preprocess4cv2dnn('bus.jpg')
    outNames = n.getUnconnectedOutLayersNames()
    n.setInput(blobImage)
    outs = n.forward(outNames)
    names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
             'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
             'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
             'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
             'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
             'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
             'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
             'cell phone',
             'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
             'hair drier', 'toothbrush']
    gen_result(outs, names)
    print(np.argmax(np.array([1111, 11111, 11, 1, 1, 1111, 1])[2:]))

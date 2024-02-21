import os
import gradio as gr
import matplotlib
import cv2
import numpy as np

# matplotlib.use('TkAgg')


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


def preprocess4cv2dnn(image):
    image = pad2square_cv2(image)
    h, w = image.shape[:2]
    blobImage = cv2.dnn.blobFromImage(image, 1.0 / 255.0, (640, 640),
                                      None, True, False)

    return blobImage, image


n = loadcv2dnnNetONNX(os.path.join('/','www','fg','models', 'best.onnx'))


def nms(dets, thresh=0.3, conf=0.6):
    """Pure Python NMS baseline."""
    # x1、y1、x2、y2、以及score赋值

    dets = dets[dets[:, 4] > conf, :]

    x0 = dets[:, 0]  # xmin
    y0 = dets[:, 1]  # ymin
    w1 = dets[:, 2]  # xmax
    h1 = dets[:, 3]  # ymax
    x1 = x0 - w1 / 2
    y1 = y0 - h1 / 2
    x2 = x0 + w1 / 2
    y2 = y0 + h1 / 2
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # argsort()返回数组值从小到大的索引值
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


def gen_result(outs, class_names, thresh=0.3, conf=0.6):
    outs = outs[0][0, :, :]
    keep, outs = nms(outs, thresh=thresh, conf=conf)
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


def draw(img, pred):
    ori_images = img.copy()
    ori_images = cv2.resize(ori_images, (640, 640))

    for i in pred['result']:
        bbox = [i['bbox']['x_center'] - i['bbox']['w'] / 2, i['bbox']['y_center'] - i['bbox']['h'] / 2,
                i['bbox']['x_center'] + i['bbox']['w'] / 2, i['bbox']['y_center'] + i['bbox']['h'] / 2, ]
        bbox = [int(ii) for ii in bbox]
        cv2.rectangle(ori_images,
                      bbox[:2], bbox[2:],
                      [225, 255, 255], 2)
        cv2.putText(ori_images, i['class'] + ' ' + str(round(i['pr_obj'], 3)), (bbox[0], bbox[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    [225, 255, 255], thickness=2)

    return ori_images


def process_img(iou_thresh, conf_thresh, the_img):
    blobImage, image = preprocess4cv2dnn(the_img)
    n.setInput(blobImage)
    outNames = n.getUnconnectedOutLayersNames()
    outs = n.forward(outNames)
    names = [ 'overflow', 'garbage', 'garbage_bin' ]
    r = gen_result(outs, names, iou_thresh, conf_thresh)

    return draw(image, r)


demo = gr.Interface(inputs=[gr.Slider(0, 1, value=0.3, label="iou_thresh", info="选择iou_thresh"),
                            gr.Slider(0, 1, value=0.7, label="conf_thresh", info="选择conf_thresh"),
                            "image"],
                    outputs=["image"], fn=process_img)

if __name__ == "__main__":
    demo.launch(server_name='0.0.0.0',server_port=1234)

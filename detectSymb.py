# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""
import telebot
import argparse
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode





@smart_inference_mode()
def myFunc(e):
  return e['x1s']
'''
def delete_double_frame(list):
    b = []
    for i in range(1, len(list)):
        if list[i] - list[i-1] < 3 and list[i] not in b:
            b.append(list[i])

    print('b = '+str(b))

    if len(b) == 1:
        list.remove(b[0])
    else:
        for i in range(len(b)):
            list.remove(b[i])

    return list
'''
def delete_double_frame(list):
    b = []
    s = list[0].get('mid_y')

    for i in range(1, len(list)):
        if list[i].get('x1s') - list[i-1].get('x1s') < 3:# and list[i].get('x1s') not in b:
            b.append(list[i].get('x1s'))
    #print(b)
    #print(s)
    #Чистка от лишних рамок
    if len(b) == 1:
        for i in range(len(list)):
            if list[i].get('x1s') == b[0]:
                list.pop(i)
                break
    else:
        pp = len(list)
        for i in range(pp-len(b)):
            for j in range(len(b)):
                if list[i].get('x1s') == b[j]:
                   # print(list[i].get('x1s'))
                    list.pop(i)
                   # print(list)
  #  print(1)
    #Добавление степени
    for i in range(1,len(list)):
        s += list[i].get('mid_y')
    mid_y = s/len(list)
    c = []
    for i in range(len(list)):
        if list[i].get('mid_y') < mid_y*0.9:
            #list.insert(i,{'x1s': list[i].get('x1s') - 1, 'mid_y': list[i].get('mid_y'), 'class': '^('})
            c.append(i)
   # print(f"c={c}")
    for i in range(len(c)):

        if i == 0 or (i != 0 and c[i] - c[i-1] > 1):
           # print("open+",c)
            list.insert(c[i], {'x1s': list[c[i]].get('x1s') - 1, 'mid_y': list[c[i]].get('mid_y'), 'class': '^('})

          #  print("open",c)
            c[i]+= 1

            for j in range(i+1,len(c)):
                c[j] += 1
          #  print("open",c)

        if i == len(c)-1 or (i != len(c)-1 and c[i+1] - c[i] > 1):
            if i == len(c)-1:
                c[i] -= 1
                list.insert(c[i] + 2,
                            {'x1s': list[c[i] + 1].get('x1s') - 1, 'mid_y': list[c[i] + 1].get('mid_y'), 'class': ')'})
              #  print("close",c)
            else:
                list.insert(c[i]+1, {'x1s': list[c[i]+1].get('x1s') - 1, 'mid_y': list[c[i]+1].get('mid_y'), 'class': ')'})
                for j in range(i + 1, len(c)):
                    c[j] += 1
           # print(list)

           # print("close",c)

    st = []
    for i in range(len(list)):
        st.append(list[i].get('class'))

    return st

def runSymb():
    weights = ROOT / 'best0804Sym.pt'  # model path or triton URL
    #weights = ROOT / 'bb.pt'
    source = ROOT / 'detectFormula/exp/crops'  # file/dir/URL/glob/screen/0(webcam)

    data = ROOT / 'data/coco128.yaml'  # dataset.yaml path
    imgsz = (640, 640)  # inference size (height, width)
    conf_thres = 0.25  # confidence threshold
    iou_thres = 0.45 # NMS IOU threshold
    max_det = 1000  # maximum detections per image
    device = '' # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img = False  # show results
    save_txt = False  # save results to *.txt
    save_conf = False  # save confidences in --save-txt labels
    save_crop = False  # save cropped prediction boxes
    nosave = False  # do not save images/videos
    classes = None  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms = False  # class-agnostic NMS
    augment = False  # augmented inference
    visualize = False  # visualize features
    update = False  # update all models
    project = ROOT / 'detectSymb'  # save results to project/name
    name = 'expSymb'  # save results to project/name
    exist_ok = False  # existing project/name ok, do not increment
    line_thickness = 3  # bounding box thickness (pixels)
    hide_labels = False  # hide labels
    hide_conf = False  # hide confidences
    half = False  # use FP16 half-precision inference
    dnn = False  # use OpenCV DNN for ONNX inference
    vid_stride = 1  # video frame-rate stride

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories


    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run

    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size

    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    try:
        open('tut.txt', 'w').close()
    except IOError:
        print('Failure')
    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
        file_i = 1
        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt

            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            # Тут запись в строку
            tut = list()
            tut_dict = dict()
            tut_end = ""
            mid_y = list()
            print("165 строка"+str(tut))


            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        x1 = int(xyxy[0].item())
                        y1 = int(xyxy[1].item())
                        x2 = int(xyxy[2].item())
                        y2 = int(xyxy[3].item())
                        bbox_points=[x1, y1, x2, y2]
                        confidence_score = conf
                        class_index = cls
                        object_name = names[int(cls)]

                        """print('bounding box is ', x1, y1, x2, y2)
                        print('class index is ', class_index)
                        print('detected object name is ', object_name)
                        print('label name is ', label)
                        print('')"""
                        original_img = im0
                        
                        i=0
                        
                        """for i in bbox_points:
                          cropped_img = im0[y1:y2, x1:x2]
                          save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                          cv2.imwrite("/content/drive/MyDrive/yolov5/crop/%i.png" %i, cropped_img) ###-----put the output folder path here---####
                          i+=1"""
                        #cropped_img = im0[y1:y2, x1:x2]
                        #save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                        #cv2.imwrite("/content/drive/MyDrive/yolov5/crop/%i.png" %i, cropped_img) ###-----put the output folder path here---####
                        i+=1   # Потом убрать это в функцию  

                    #cv2.imwrite('test.png',cropped_img)
                    tut.append(x1)
                    tut_dict.update({x1 : object_name})
                    #print('221 = '+str(tut))
                    #print(f"y1 = {y1} y2 = {y2} d(y2 -y1) = {(y2 -y1)/2 + y1} class = {object_name}")
                    mid_y.append({"x1s" : x1,"mid_y" : (y2 -y1)/2 + y1, "class" :object_name})

                    #if save_crop:
                     #   save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
           # print(mid_y)
            mid_y.sort(key=myFunc)
           # print(mid_y)
            tut.sort()
            tuta = []

          #  print("tuta1= "+ str(tuta))
            tuta = delete_double_frame(mid_y)

            if tuta[0] == "0" and str(tuta[1]).isdigit() :
                tuta.pop(0)
          #  print("tuta2= " + str(tuta))
            for i in range(len(tuta)):
                tut_end += str(tuta[i]) + " "
           # print("tuta3= " + str(tuta))
          #  print(tut_end)
          #  print(tut_dict)
            my_file = open("tut.txt", "a")
            my_file.write('\n')
            my_file.write(tut_end)
            my_file.close()
            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        #print(tut)
        tut.sort()

        #print(tut_dict)
        for i in range(len(tut)):
          tut_end += str(tut_dict.get(tut[i])) + " "
      #  print("tut_end = ")
      #  print(tut_end)
      #  print('mid_y = ')
        #print(mid_y)









    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)



def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detectFormula', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

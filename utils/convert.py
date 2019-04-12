## This file help convert icdr2019:$PATH_PYTORCH_YOLO_TEXT/data/icdr2019 to a voc format 
## which can be trained through YOLOv3 model. 

import numpy as np
import cv2

def polylines(img,points):
    '''
    @points: boundry points
    '''
    im = np.zeros(img.shape[:2],dtype='uint8')
    for point in points:
        b = np.array([point],dtype=np.int32)
        cv2.fillPoly(im,b,255)
    return im

def check_points(points,w,h):
    # See if the points are correct
    check = False
    for point in points :
        for x,y in point :
            if x>w or y>h :
                check = True
                break
        if check :
            break
    return check

def get_points(res):
    points = []
    for line in res :
        points.append(line['points'])
    return points

def resize_im(img,scale=416,max_scale=608):
    h,w = img.shape[:2]
    f = float(scale)/min(h,w)
    if max_scale is not None:
        if f*max(h,w)>max_scale :
            f = float(max_scale)/max(h,w)
    newW,newH = int(w*f),int(h*f)
    newW,newH = newW-(newW%32),newH-(newH%32)
    fw = w/newW
    fh = h/newH
    tmpImg = cv2.resize(img,None,None,fx=1/fw,fy=1/fh,interpolation=cv2.INTER_LINEAR)
    return tmpImg,fw,fh

def clean_im(im):
    avg = 127
    im[im>avg] = 255
    im[im<=avg] = 0
    y,x =np.where(im==255)
    xmin,ymin,xmax,ymax = min(x),min(y),max(x),max(y)
    return xmin,ymin,xmax,ymax

def adjust_height(h):
    heights = [11,16,23,33,48,68,97,139,198,283]
    N = len(heights)
    for i in range(N-1):
        if h <= heights[i]+heights[i]*0.44/2:
            return heights[i]
    return h

def img_split_to_box(im,splitW = 15,adjust=True):
    tmpIm = im==255
    h,w = tmpIm.shape[:2]
    num = w//splitW+1

    box = []
    for i in range(num-1):
        # Split detected box to several grids with width 15
        xmin,ymin,xmax,ymax = splitW*i,0,splitW*(i+1),h

        # Child grid
        childIm = tmpIm[ymin:ymax,xmin:xmax]
        checkYmin = False
        checkYmax = False
        for j in range(ymax):
            if not checkYmin:
                if childIm[j].max():
                    ymin = j
                    checkYmin = True
            if not checkYmax:
                if childIm[ymax-j-1].max():
                    ymax = ymax-j
                    checkYmax = True
        if adjust:
            childH = ymax-ymin+1
            cy = (ymax+ymin)/2
            childH = adjust_height(childH)
            ymin = cy-childH/2
            ymax = cy+childH/2
        box.append([xmin,ymin,xmax,ymax])

    return box

def resize_img_box(p,root,train_labels,scale=416,max_scale=608,splitW=15,adjust=True):
    path = root.format(p)
    img = cv2.imread(path)
    if img is None:
        return None,[]
    points = get_points(train_labels[f'{p}'])
    h,w = img.shape[:2]
    check = check_points(points,w,h)
    if check:
        return None,[]
    img,fw,fh = resize_im(img,scale=scale,max_scale=max_scale)
    boxes = []
    for point in points : 
        point = [[bx[0]/fw,bx[1]/fh] for bx in point]
        im = polylines(img,[point])
        if im.max()==0:
            continue
    xmin,ymin,xmax,ymax = clean_im(im)
    tmp = im[ymin:ymax,xmin:xmax]
    box = img_split_to_box(tmp,splitW=splitW,adjust=adjust)
    childBoxes = []
    for bx in box:
        xmin_,ymin_,xmax_,ymax_ = bx
        xmin_,ymin_,xmax_,ymax_ = xmin+xmin_,ymin+ymin_,xmax+xmax_,ymax+ymax_
        boxes.append([xmin_,ymin_,xmax_,ymax_])

    return img,boxes

def convert(size,box):
    
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0]+box[2])/2.0-1
    y = (box[1]+box[3])/2.0-1
    w = box[2]-box[0]
    h = box[3]-box[1]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return [str(x),str(y),str(w),str(h)]

def convert_annotation(p,root,train_labels,scale=608,max_scale=1024,splitW=8,adjust=False):
    img,boxes = resize_img_box(p,root=root,train_labels=train_labels,scale=scale,max_scale=scale,splitW=splitW,adjust=adjust)
    if img is None or len(boxes) == 0:
        return None,''
    h,w = img.shape[:2]
    newBoxes = []
    for bx in boxes:
        cls_id = 1
        bb = convert((w,h),bx)
        newBoxes.append(' '.join([str(cls_id)]+bb))
    return img,'\n'.join(newBoxes)

def write_for_darknet(img,newBoxes,filename,img_path,label_path):
    imgP = os.path.join(img_path,filename+'.jpg')
    txtP = os.path.join(label_path,filename+'.txt')
    cv2.imwrite(imgP,img)
    with open(txtP,'w') as f:
        f.write(newBoxes)

if __name__ == '__main__':
    import os
    import json
    dataRoot = '/home/zhijue/Desktop/Yufeng/Pytorch-YOLO-text/data/icdr2019/'
    darknetRoot = '/home/zhijue/Desktop/Yufeng/Pytorch-YOLO-text'
    wP = '/home/zhijue/Desktop/Yufeng/Pytorch-YOLO-text/weights/darknet53.conv.74'

    root = dataRoot + 'train_images/{}.jpg'
    with open(dataRoot+'train_labels.json') as f:
        train_labels = json.loads(f.read())

    labelP = os.path.join(darknetRoot,'VOCdevkit','VOC2007','labels')
    JPEGP = os.path.join(darknetRoot,'VOCdevkit','VOC2007','JPEGImages')
    if not os.path.exists(labelP):
        os.makedirs(labelP)
    if not os.path.exists(JPEGP):
        os.makedirs(JPEGP)

    for p in train_labels.keys():
        img,newBoxes = convert_annotation(p,root,train_labels,scale=608,max_scale=1024,splitW=8,adjust=True)
        if img is None or len(newBoxes)==0:
            continue
        write_for_darknet(img,newBoxes,p,JPEGP,labelP)

    from sklearn.model_selection import train_test_split
    from glob import glob
    jpgPaths = glob(os.path.join(JPEGP,'*.jpg'))
    train,test = train_test_split(jpgPaths,test_size=0.1)
    trainP = os.path.join(darknetRoot,'VOCdevkit','VOC2007','train.txt')
    testP = os.path.join(darknetRoot,'VOCdevkit','VOC2007','test.txt')
    with open(trainP,'w') as f:
        f.write('\n'.join(train))
    with open(testP,'w') as f:
        f.write('\n'.join(test))


    vocDataP = os.path.join(darknetRoot,'VOCdevkit','VOC2007','voc.data')
    vocNameP = os.path.join(darknetRoot,'VOCdevkit','VOC2007','voc.name')
    with open(vocDataP,'w') as f:
        f.write('class = 2\n')
        f.write('train = {}\n'.format(trainP))
        f.write('valid = {}\n'.format(testP))
        f.write('names = {}\n'.format(vocNameP))
        f.write('backup = backup')

    with open(vocNameP,'w') as f:
        f.write('none\n')
        f.write('text\n') 

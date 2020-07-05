#pip install opencv-contrib-python
import cv2 as cv
import numpy as np
from arMarker import detect_markers
import colorTransfer

## Load the predefined dictionary
#dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250)
#
## Generate the marker
#markerImage = np.zeros((500, 500), dtype=np.uint8)
#markerImage = cv.aruco.drawMarker(dictionary, 13, 500, markerImage, 1);
#
#cv.imwrite("marker13.png", markerImage);
#
#exit()

import numpy as np
import cv2

import config
from paperDetect import getPaperCoords, drawContours, getPaperCoords2
from qrDetector import drawQrCodeBorder, qrDetect
from PIL import Image
import math

import dbrar.dbrar as dbrar
import adnet.adnet as adnet
import ycbcr
import lab
import hsi
from objloader_simple import *
import os

hsiModel = hsi.HSI()
dbrarModel = dbrar.DBRAR()
ycbcrModel = ycbcr.YCBCR()
labModel = lab.LAB()
adnetModel = adnet.ADNET()

cap = cv2.VideoCapture(0)

textureImg = cv2.imread('assets/grass.png')
textureImgOrig = textureImg.copy()
boxPoint=None

moveShift = [0, 0]
scaleShift = [0, 0]

shiftForBrightTransfer = [0, 0]

homography = None 
camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]]) 
obj = OBJ('assets/models/pirate-ship-fat.obj', swapyz=True)
modelScale = 50

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))

def render(img, obj, projection, color=False):
    vertices = obj.vertices
    scale_matrix = np.eye(3) * modelScale
    h = 500
    w = 500

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, (137, 27, 211))
        else:
            #print(face)
            color = hex_to_rgb(face[-1])
            color = color[::-1]  # reverse
            cv2.fillConvexPoly(img, imgpts, color)

    return img
def projection_matrix(camera_parameters, homography):
    # Compute rotation along the x and y axis as well as the translation
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    # normalise vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(camera_parameters, projection)

def getTransformM(s_points, w, h):
    global homography
    # Euclidean distance - calculate maximum height and width
    #height = max(np.linalg.norm(s_points[0] - s_points[1]),
    #             np.linalg.norm(s_points[2] - s_points[3]))
    #width = max(np.linalg.norm(s_points[1] - s_points[2]),
    #             np.linalg.norm(s_points[3] - s_points[0]))
    
    pts1 = np.float32([[0,0],[0,h],[w,h], [w,0]])
    pts2 = np.float32([[s_points[0][0], s_points[0][1]],
                       [s_points[1][0], s_points[1][1]],
                       [s_points[2][0], s_points[2][1]],
                       [s_points[3][0], s_points[3][1]]])

    homography, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    #M = cv2.getPerspectiveTransform(pts1,pts2)
    #return M

def persp_transform_inv(img, s_points, w, h):
    if len(s_points) < 4:
        return img
    # Euclidean distance - calculate maximum height and width
    height = max(np.linalg.norm(s_points[0] - s_points[1]),
                 np.linalg.norm(s_points[2] - s_points[3]))
    width = max(np.linalg.norm(s_points[1] - s_points[2]),
                 np.linalg.norm(s_points[3] - s_points[0]))
    
    pts1 = np.float32([[0,0],[0,h],[w,h], [w,0]])
    pts2 = np.float32([[s_points[0][0], s_points[0][1]],
                       [s_points[1][0], s_points[1][1]],
                       [s_points[2][0], s_points[2][1]],
                       [s_points[3][0], s_points[3][1]]])

    M = cv2.getPerspectiveTransform(pts1,pts2)
    
    return cv2.warpPerspective(img, M, (int(w), int(h)))
def rotate(A,B,C):
  return (B[0]-A[0])*(C[1]-B[1])-(B[1]-A[1])*(C[0]-B[0])

def jarvismarch(A):
    n = len(A)
    P = [x for x in range(n)]
    for i in range(1,n):
        if A[P[i]][0]<A[P[0]][0]: 
            P[i], P[0] = P[0], P[i]  
    H = [P[0]]
    del P[0]
    P.append(H[0])
    while True:
        right = 0
        for i in range(1,len(P)):
            if rotate(A[H[-1]],A[P[right]],A[P[i]])<0:
                right = i
        if P[right]==H[0]: 
            break
        else:
            H.append(P[right])
            del P[right]
    res = [A[h] for h in H]
    res.reverse()
    return np.array(res)




def shadow(im, mask, brightness=0.5):
    im = im.resize((200,200))
    mask = mask.resize((200,200))
    result = Image.new('RGB', im.size)
    for x in range(im.size[0]):
        for y in range(im.size[1]):
            _r, _g, _b = mask.getpixel((x,y))
            r, g, b = im.getpixel((x, y))
            if (_r==0 and _g==0 and _b==0) and (r!=0 or g!=0 or b!=0):
                red = int(r * brightness)
                red = min(255, max(0, red))

                green = int(g * brightness)
                green = min(255, max(0, green))

                blue = int(b * brightness)
                blue = min(255, max(0, blue))

                result.putpixel((x, y), (red, green, blue))
            else:
                result.putpixel((x, y), (r, g, b))
    #result.save(result_name, "JPEG")
    #implt(result)
    return result

def shadow2(im, mask, brightness=0.5):
    #result = im #Image.new('RGB', im.size)
    
    maxX =-100000
    minX = 100000
    maxY =-100000
    minY = 100000
    if boxPoint is not None:
        for p in boxPoint:
            if p[1] > maxY:
                maxY = p[1]
            if p[1] < minY:
                minY = p[1]
            if p[0] > maxX:
                maxX = p[0]
            if p[0] < minX:
                minX = p[0]
    if maxX == -100000:
        maxX = im.shape[1]-1
    if minX == 100000:
        minX = 0
    if maxY == -100000:
        maxY = im.shape[0]-1
    if minY == 100000:
        minY = 0

    b,g,r = cv2.split(im)
    
    brShift = 0.08
    
    for x in range(minY, maxY):
        for y in range(minX, maxX):
            if (not mask[x,y].any(0) and im[x,y].any(0)):
                if checkX(x, y, 5, mask, im, minX, maxX):
                    r.itemset(x, y, r.item(x, y) * (brightness+brShift*3))
                    g.itemset(x, y, g.item(x, y) * (brightness+brShift*3))
                    b.itemset(x, y, b.item(x, y) * (brightness+brShift*3))
                elif checkX(x, y,4, mask, im, minX, maxX):
                    r.itemset(x, y, r.item(x, y) * (brightness+brShift*2))
                    g.itemset(x, y, g.item(x, y) * (brightness+brShift*2))
                    b.itemset(x, y, b.item(x, y) * (brightness+brShift*2))
                elif checkX(x,y, 3, mask, im, minX, maxX):
                    r.itemset(x, y, r.item(x, y) * (brightness+brShift*2))
                    g.itemset(x, y, g.item(x, y) * (brightness+brShift*2))
                    b.itemset(x, y, b.item(x, y) * (brightness+brShift*2))
                elif checkX(x,y, 2, mask, im, minX, maxX):
                    r.itemset(x, y, r.item(x, y) * (brightness+brShift*1))
                    g.itemset(x, y, g.item(x, y) * (brightness+brShift*1))
                    b.itemset(x, y, b.item(x, y) * (brightness+brShift*1))
                elif checkX(x,y, 1, mask, im, minX, maxX):
                    r.itemset(x, y, r.item(x, y) * (brightness+brShift*1))
                    g.itemset(x, y, g.item(x, y) * (brightness+brShift*1))
                    b.itemset(x, y, b.item(x, y) * (brightness+brShift*1))
                else:
                    r.itemset(x, y, r.item(x, y) * brightness)
                    g.itemset(x, y, g.item(x, y) * brightness)
                    b.itemset(x, y, b.item(x, y) * brightness)

    result = cv2.merge((b, g, r))
    return result

def checkX(y, x, shift, mask, im, minX, maxX):
    if x-shift > minX and mask[y, x-shift].any(0):
        return True
    if x+shift < maxX and mask[y, x+shift].any(0):
        return True
    return False

def shadow3(im, mask, brightness=0.5):
    h, w, c = mask.shape
    # append Alpha channel -- required for BGRA (Blue, Green, Red, Alpha)
    mask_bgra = np.concatenate([mask, np.full((h, w, 1), 255, dtype=np.uint8)], axis=-1)
    #im_bgra = np.concatenate([im, np.full((h, w, 1), 255, dtype=np.uint8)], axis=-1)
    # create a mask where white pixels ([255, 255, 255]) are True
    white = np.all(mask == [240, 240, 240], axis=-1)
    
    #print(white)
    #cv.imwrite("white.png", mask);

    black = np.all(mask == [0, 0, 0], axis=-1)
    # change the values of Alpha to 0 for all the white pixels
    mask_bgra[white, -1] = 0
        
    mask_bgra[black, -1] = 255 * brightness
    overlay_img = mask_bgra[:,:,:3] # Grab the BRG planes
    overlay_mask = mask_bgra[:,:,3:]  # And the alpha plane
    background_mask = 255 - overlay_mask
    # Turn the masks into three channel, so we can use them as weights
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    # Create a masked out face image, and masked out overlay
    # We convert the images to floating point in range 0.0 - 1.0
    face_part = (im * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))
    res = np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))
    res =  cv2.cvtColor(res, cv2.COLOR_RGBA2RGB)
    #cv2.imwrite("out.png", res)
    return res

def makeMagic3(im, mask):
    h, w, c = mask.shape
    # append Alpha channel -- required for BGRA (Blue, Green, Red, Alpha)
    mask_bgra = np.concatenate([mask, np.full((h, w, 1), 255, dtype=np.uint8)], axis=-1)
    #im_bgra = np.concatenate([im, np.full((h, w, 1), 255, dtype=np.uint8)], axis=-1)

    black = np.all(mask == [0, 0, 0], axis=-1)
    mask_bgra[black, -1] = 0

    # Split out the transparency mask from the colour info
    overlay_img = mask_bgra[:,:,:3] # Grab the BRG planes
    overlay_mask = mask_bgra[:,:,3:]  # And the alpha plane

    # Again calculate the inverse mask
    background_mask = 255 - overlay_mask

    # Turn the masks into three channel, so we can use them as weights
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    # Create a masked out face image, and masked out overlay
    # We convert the images to floating point in range 0.0 - 1.0
    face_part = (im * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    res = np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))
    res =  cv2.cvtColor(res, cv2.COLOR_RGBA2RGB)
    return res

def makeMagic(im, mask, brightness=0.5):
    im = im.resize((200,200))
    mask = mask.resize((200,200))
    result = Image.new('RGB', im.size)
    for x in range(im.size[0]):
        for y in range(im.size[1]):
            _r, _g, _b = mask.getpixel((x,y))
            r, g, b = im.getpixel((x, y))
            if (_r==0 and _g==0 and _b==0):
                result.putpixel((x, y), (r, g, b))
            else:
                result.putpixel((x, y), (_r, _g, _b))
    #result.save(result_name, "JPEG")
    #implt(result)
    return result

def makeMagic2(im, mask, brightness=0.5):
    #res = cv2.addWeighted(im, 0.5, mask, 0.5, 0)
    #new = cv2.cvtColor(im, cv2.COLOR_RGB2RGBA)
    #res = cv2.bitwise_and(new, new, mask=mask)
    #return res
    maxX =-100000
    minX = 100000
    maxY =-100000
    minY = 100000
    if boxPoint is not None:
        for p in boxPoint:
            if p[1] > maxY:
                maxY = p[1]
            if p[1] < minY:
                minY = p[1]
            if p[0] > maxX:
                maxX = p[0]
            if p[0] < minX:
                minX = p[0]
    if maxX == -100000:
        maxX = im.shape[1]-1
    if minX == 100000:
        minX = 0
    if maxY == -100000:
        maxY = im.shape[0]-1
    if minY == 100000:
        minY = 0



    b,g,r = cv2.split(im)
    bm,gm,rm = cv2.split(mask)
    for x in range(minY, maxY):
        for y in range(minX, maxX):
            if (mask[x,y].any(0)):
                r.itemset(x, y, rm.item(x, y))
                g.itemset(x, y, gm.item(x, y))
                b.itemset(x, y, bm.item(x, y))

            #_r, _g, _b = mask.getpixel((x,y))
            #r, g, b = im.getpixel((x, y))
            #if (_r==0 and _g==0 and _b==0):
            #    result.putpixel((x, y), (r, g, b))
            #else:
            #    result.putpixel((x, y), (_r, _g, _b))
    result = cv2.merge((b, g, r))
    return result

BLACK_LINE_SIZE = 100

applyColorTransfer=False

while(True):
    ret, frame = cap.read()
    frame = frame[BLACK_LINE_SIZE:frame.shape[0]-BLACK_LINE_SIZE, :]
    height, width, channels = frame.shape

    if config.COLOR_TRANSFER and not applyColorTransfer:
        applyColorTransfer = True
        textureImg = colorTransfer.colorTransfer(textureImg, frame)
        #cv.imwrite("colorTransfer.png", textureImg)
    textureRes = None
    if config.DETECT_TYPE == config.DETECT_PAPER:
        paperCoords, frame = getPaperCoords2(frame)
        paperCoords = jarvismarch(paperCoords)

        if paperCoords is None:
            continue
        if config.DRAW_CONTOUROS:
            drawContours(frame, paperCoords)

        cv2.imshow('frame', frame)

        imageRoad = cv2.cvtColor(textureImg, cv2.COLOR_BGR2RGB)

        imageRoad = cv2.resize(imageRoad, (width,height))
        textureRes = imageRoad
        textureRes = persp_transform_inv(imageRoad, paperCoords, width, height)
        boxPoint = paperCoords
        if config.DRAW_HELP_WINDOWS:
            cv2.imshow('frame2', textureRes)

        
    elif config.DETECT_TYPE == config.DETECT_QR:
        #cv2.imshow('frame', frame)
        res, box = qrDetect(frame)
        if config.DRAW_HELP_WINDOWS:
            cv2.imshow('frameRes', res)

        imageRoad = cv2.cvtColor(textureImg, cv2.COLOR_BGR2RGB)
        imageRoad = cv2.resize(imageRoad, (width,height))
        textureRes = imageRoad
        if box is not None:
            box = [[a[0][0], a[0][1]] for a in box]
            textureRes = persp_transform_inv(imageRoad, np.array(box), width, height)
            boxPoint = box
    elif config.DETECT_TYPE == config.DETECT_AR:
        markers = detect_markers(frame)
        imageRoad = cv2.cvtColor(textureImg, cv2.COLOR_BGR2RGB)
        imageRoad = cv2.resize(imageRoad, (width,height))
        textureRes = imageRoad
        if (markers != []):
            for marker in markers:
                marker.highlite_marker(frame)
            if config.DRAW_HELP_WINDOWS:
                cv2.imshow('frame2', frame)
            
            if len(markers) > 0:
                ## remove marker from frame
                #box = np.array([[a[0][0], a[0][1]] for a in markers[0].contours])
                #_color = frame[box[0][1]-5][box[0][0]]
                #_color = (int(255), int(255), int(255))
                #cv2.fillPoly(frame, pts=[box], color=_color)

                #apply shift
                box = [[a[0][0]+moveShift[0], a[0][1]+moveShift[1]] for a in markers[0].contours]
                box[0][0]-=scaleShift[0]
                box[0][1]+=scaleShift[1]

                box[1][0]+=scaleShift[0]
                box[1][1]+=scaleShift[1]

                box[2][0]+=scaleShift[0]
                box[2][1]-=scaleShift[1]

                box[3][0]-=scaleShift[0]
                box[3][1]-=scaleShift[1]

                #textureRes = persp_transform_inv(imageRoad, np.array(box), width, height)
                
                
                ## (1) Crop the bounding rect
                pts = np.array(box)
                croped = imageRoad.copy()
                ## (2) make mask
                shift = pts.min(axis=0)
                shiftForBrightTransfer = shift
                #print(shift)
                pts = pts - shift
                mask = np.zeros(croped.shape[:2], np.uint8)
                cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
                ## (3) do bit-op
                textureRes = cv2.bitwise_and(croped, croped, mask=mask)
                M = np.float32([[1,0,shift[0]],[0,1,shift[1]]])
                textureRes = cv.warpAffine(textureRes,M,(textureRes.shape[1], textureRes.shape[0]))


                boxPoint = box
                
        else:
            if (boxPoint is not None):
                #textureRes = persp_transform_inv(imageRoad, np.array(boxPoint), width, height)
                ## (1) Crop the bounding rect
                pts = np.array(box)
                croped = imageRoad.copy()
                ## (2) make mask
                shift = pts.min(axis=0)
                #print(shift)
                pts = pts - shift
                mask = np.zeros(croped.shape[:2], np.uint8)
                cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
                ## (3) do bit-op
                textureRes = cv2.bitwise_and(croped, croped, mask=mask)
                M = np.float32([[1,0,shift[0]],[0,1,shift[1]]])
                textureRes = cv.warpAffine(textureRes,M,(textureRes.shape[1], textureRes.shape[0]))
    shadowFrame = None
    if config.SHADOW_TYPE == config.SHADOW_ADNET:
        shadowFrame = adnetModel.getShadow(frame, width, height)
        if config.DRAW_HELP_WINDOWS:
            cv2.imshow('frameShadow', shadowFrame)
    elif config.SHADOW_TYPE == config.SHADOW_HSI:
        shadowFrame = hsiModel.getShadow(frame)
        if config.DRAW_HELP_WINDOWS:
            cv2.imshow('frameShadow', shadowFrame)
    elif config.SHADOW_TYPE == config.SHADOW_DBRAR:
        shadowFrame = dbrarModel.getShadow(frame)
        if config.DRAW_HELP_WINDOWS:
            cv2.imshow('frameShadow', shadowFrame)
    elif config.SHADOW_TYPE == config.SHADOW_YCBCR:
        shadowFrame = ycbcrModel.getShadow(frame)
        if config.DRAW_HELP_WINDOWS:
            cv2.imshow('frameShadow', shadowFrame)
    elif config.SHADOW_TYPE == config.SHADOW_LAB:
        shadowFrame = labModel.getShadow(frame)
        if config.DRAW_HELP_WINDOWS:
            cv2.imshow('frameShadow', shadowFrame)
    #make result frame
    
    if boxPoint is not None:
        if config.BRIGHTNESS_TRANSFER:
            #textureImg = colorTransfer.brightnessTransfer(frame, textureImgOrig, boxPoint)
            textureImg = colorTransfer.brightnessTransfer2(frame, textureImgOrig, boxPoint, shiftForBrightTransfer)


    #Texture AR
    ret, thresh1 = cv2.threshold(shadowFrame, 200, 240, cv2.THRESH_BINARY_INV)
    _mask = cv2.cvtColor(thresh1, cv2.COLOR_GRAY2BGR)
    textureWithShadow = shadow2(textureRes, _mask)
    result = makeMagic3(frame, textureWithShadow)
    cv2.imshow('frameResult', result)

    #if boxPoint is not None:
    #    w, h =500, 500
    #    pts1 = np.float32([[0,0],[0,h],[w,h], [w,0]])
    #    pts2 = np.float32([[boxPoint[0][0], boxPoint[0][1]],
    #                   [boxPoint[1][0], boxPoint[1][1]],
    #                   [boxPoint[2][0], boxPoint[2][1]],
    #                   [boxPoint[3][0], boxPoint[3][1]]])
    #    homography, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    #    if homography is not None:
    #        M = projection_matrix(camera_parameters, homography)  
    #        frameWithObj = render(frame, obj, M, True)
    #        result = shadow2(frameWithObj, _mask)
    #        cv2.imshow('frameObj', result)
    

    #textureWithShadow = shadow(Image.fromarray(textureRes), Image.fromarray(res))
    #result = makeMagic(Image.fromarray(frame), textureWithShadow)
    #result = result.resize((400,400))
    

    #print(moveShift)
    #print(scaleShift)
    #print("----------")

    #Events
    #exit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    #move
    if key == ord('w'):
        moveShift[1]+=10
    elif key== ord('s'):
        moveShift[1]-=10
    elif key== ord('a'):
        moveShift[0]-=10
    elif key == ord('d'):
        moveShift[0]+=10

    #X scale
    elif key == ord('1'):
        scaleShift[0]-=10
    elif key == ord('2'):
        scaleShift[0]+=10
    #Y scale
    elif key== ord('3'):
        scaleShift[1]-=10
    elif key == ord('4'):
        scaleShift[1]+=10

    #Y scale
    elif key== ord('5'):
        modelScale-=10
    elif key == ord('6'):
        modelScale+=10

    

cap.release()
cv2.destroyAllWindows()
"""Conservative, editable solder/lead candidates; these are not defect labels."""
import cv2
import numpy as np
from pcb_alignment import read_image

PROFILES={
    "precise":dict(area=120,thickness=3.1,fill=.43,texture=20),
    "balanced":dict(area=40,thickness=2.5,fill=.36,texture=10),
    "dense":dict(area=22,thickness=1.8,fill=.28,texture=6),
}


def detect_sites(path,sensitivity="precise"):
    if sensitivity not in PROFILES:
        raise ValueError("请选择有效的定位精度")
    profile=PROFILES[sensitivity]
    image=read_image(path,limit=1920)
    height,width=image.shape[:2]
    unit=max(height,width)/1920
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    blue,green,red=cv2.split(image.astype(np.int16))
    substrate=(green>red*1.12)&(green>blue*1.02)&(green>45)
    board=substrate.astype(np.uint8)*255
    size=max(3,int(round(35*unit))|1)
    board=cv2.morphologyEx(board,cv2.MORPH_CLOSE,np.ones((size,size),np.uint8))
    contours,_=cv2.findContours(board,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("未找到电路板区域，请使用清楚的正面标准图")
    board[:]=0
    cv2.drawContours(board,[max(contours,key=cv2.contourArea)],-1,255,-1)
    erosion=max(3,int(round(9*unit))|1)
    board=cv2.erode(board,np.ones((erosion,erosion),np.uint8))
    # Broader neutral highlights include the grey portions of solder, not only glare.
    metal=((hsv[:,:,1]<100)&(hsv[:,:,2]>100)&(green-red<35)&(board>0)).astype(np.uint8)*255
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    opening=max(3,int(round((5 if sensitivity!="dense" else 3)*unit))|1)
    metal=cv2.morphologyEx(metal,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(opening,opening)))
    metal=cv2.morphologyEx(metal,cv2.MORPH_CLOSE,kernel)
    count,labels,stats,_=cv2.connectedComponentsWithStats(metal)
    rectangles=[]
    for index in range(1,count):
        x,y,w,h,area=stats[index]
        if not (max(60 if sensitivity=="precise" else 20 if sensitivity=="balanced" else 8,profile["area"]*unit*unit)<=area<=width*height*.003):
            continue
        if min(w,h)<max(3,5*unit) or not 1.1<=max(w,h)/min(w,h)<=6 or w>width*.09 or h>height*.09:
            continue
        component=(labels[y:y+h,x:x+w]==index).astype(np.uint8)
        distance=cv2.distanceTransform(component,cv2.DIST_L2,3)
        if distance.max()<max(1.3,profile["thickness"]*unit) or area/(w*h)<max(.52,profile["fill"]):
            continue
        contours,_=cv2.findContours(component,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contour=max(contours,key=cv2.contourArea)
        _,sides,_=cv2.minAreaRect(contour)
        if not sides[0]*sides[1] or cv2.contourArea(contour)/(sides[0]*sides[1])<.66:
            continue
        values=hsv[y:y+h,x:x+w,2][distance>max(.9,unit)]
        if len(values)<5 or float(values.std())<profile["texture"]:
            continue
        center=hsv[y+int(h*.2):y+max(int(h*.8),int(h*.2)+1),x+int(w*.2):x+max(int(w*.8),int(w*.2)+1)]
        if np.mean((center[:,:,2]<100)&(center[:,:,1]<120))>.3:
            continue
        rectangles.append((int(x),int(y),int(w),int(h)))
    # SMD pads must be opposite terminals of a populated or empty footprint. An isolated bright
    # rectangle, printed character or entire component is not a joint target.
    supported=set()
    for i,first in enumerate(rectangles):
        x,y,w,h=first
        for j in range(i+1,len(rectangles)):
            a,b,c,d=rectangles[j]
            if max(w,c)/min(w,c)>2.1 or max(h,d)/min(h,d)>2.1:
                continue
            region=None;paired_shape=False
            if abs(y+h/2-b-d/2)<min(h,d)*.45:
                left,right=sorted([first,rectangles[j]],key=lambda r:r[0])
                gap=right[0]-left[0]-left[2]
                if min(w,c)*.2<=gap<=max(w,c)*7:
                    top=max(y,b);bottom=min(y+h,b+d)
                    region=hsv[top:bottom,left[0]+left[2]:right[0]]
                    paired_shape=min(h/w,d/c)>=1.4 and gap<=max(w,c)*5
            if region is None and abs(x+w/2-a-c/2)<min(w,c)*.45:
                top,bottom=sorted([first,rectangles[j]],key=lambda r:r[1])
                gap=bottom[1]-top[1]-top[3]
                if min(h,d)*.2<=gap<=max(h,d)*7:
                    left=max(x,a);right=min(x+w,a+c)
                    region=hsv[top[1]+top[3]:bottom[1],left:right]
                    paired_shape=min(w/h,c/d)>=1.4 and gap<=max(h,d)*5
            if region is None or not region.size:
                continue
            body=(region[:,:,2]<155)&(region[:,:,1]<155)
            green_gap=(region[:,:,1]>100)&(region[:,:,0]>25)&(region[:,:,0]<100)
            if np.mean(body)<(.4 if sensitivity=="precise" else .3) and not (paired_shape and np.mean(green_gap)>.4):
                continue
            supported.update((i,j))
    proposals=[]
    pad=max(1,round(2*unit))
    for index in supported:
        x,y,w,h=rectangles[index]
        proposals.append(([max(0,x-pad),max(0,y-pad),min(width,x+w+pad),min(height,y+h+pad)],"smd_joint"))
    # Plated through holes have a dark, neutral centre and a metallic ring.
    # A white silkscreen ring surrounding green substrate is rejected.
    dark=((hsv[:,:,2]<80)&(board>0)).astype(np.uint8)*255
    holes,_=cv2.findContours(dark,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    neutral_metal=(hsv[:,:,1]<115)&(hsv[:,:,2]>115)&(board>0)
    hole_candidates=[]
    for contour in holes:
        area=cv2.contourArea(contour);perimeter=cv2.arcLength(contour,True)
        if not perimeter or area<max(5,8*unit*unit) or 4*np.pi*area/perimeter**2<.55:
            continue
        (cx,cy),radius=cv2.minEnclosingCircle(contour)
        hole_candidates.append((cx,cy,radius))
    circles=cv2.HoughCircles(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),cv2.HOUGH_GRADIENT,1,
        max(6,8*unit),param1=100,param2=12,minRadius=max(3,round(3*unit)),maxRadius=max(9,round(30*unit)))
    if circles is not None:
        hole_candidates.extend((float(x),float(y),float(radius)/1.4) for x,y,radius in circles[0])
    for cx,cy,radius in hole_candidates:
        if not max(2,2.5*unit)<=radius<=max(6,17*unit):
            continue
        outer=radius*1.65
        left,top,right,bottom=int(cx-outer-2),int(cy-outer-2),int(cx+outer+3),int(cy+outer+3)
        if left<0 or top<0 or right>width or bottom>height:
            continue
        yy,xx=np.ogrid[top:bottom,left:right]
        distance=np.sqrt((xx-cx)**2+(yy-cy)**2)
        core=distance<radius*.65;ring=(distance>radius*1.03)&(distance<outer)
        roi=hsv[top:bottom,left:right]
        if not core.any() or not ring.any() or np.mean(roi[:,:,2][core]<100)<(.9 if sensitivity=="precise" else .8):
            continue
        metal_ring=neutral_metal[top:bottom,left:right]
        if np.mean(metal_ring[ring])<(.42 if sensitivity=="precise" else .3):
            continue
        coverage=0;green_context=0;context_values=[]
        for angle in np.linspace(0,2*np.pi,16,endpoint=False):
            px=int(round(cx+radius*1.3*np.cos(angle)));py=int(round(cy+radius*1.3*np.sin(angle)))
            coverage+=bool(neutral_metal[py,px])
            gx=int(round(cx+radius*2.3*np.cos(angle)));gy=int(round(cy+radius*2.3*np.sin(angle)))
            if 0<=gx<width and 0<=gy<height and substrate[gy,gx]:
                green_context+=1;context_values.append(hsv[gy,gx,2])
        if coverage<(12 if sensitivity=="precise" else 9) or green_context<(8 if sensitivity=="precise" else 6):
            continue
        if np.median(roi[:,:,2][core])>np.median(context_values)*.7:
            continue
        proposals.append(([left,top,right,bottom],"through_hole"))
    kept=[]
    for rect,kind in sorted(proposals,key=lambda item:(item[0][2]-item[0][0])*(item[0][3]-item[0][1]),reverse=True):
        area=(rect[2]-rect[0])*(rect[3]-rect[1])
        if any(max(0,min(rect[2],other[2])-max(rect[0],other[0]))*
               max(0,min(rect[3],other[3])-max(rect[1],other[1]))/area>.55 for other,_ in kept):
            continue
        kept.append((rect,kind))
    kept.sort(key=lambda item:(round((item[0][1]+item[0][3])/max(12,24*unit)),item[0][0]))
    return [dict(id=f"S{index:04}",x1=rect[0]/width,y1=rect[1]/height,
                 x2=rect[2]/width,y2=rect[3]/height,origin="auto",kind=kind)
            for index,(rect,kind) in enumerate(kept,1)]

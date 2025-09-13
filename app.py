# app.py
# -*- coding: utf-8 -*-
import base64, io, os, json
from datetime import datetime
from typing import Optional, Dict, List, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

app = FastAPI(title="OMR Web Demo")

# ===== paths =====
DEBUG_DIR = "./omr_debug"
TEMPLATE_PATH = "./template.json"
os.makedirs(DEBUG_DIR, exist_ok=True)

# ===== default template (จะถูก override ถ้ามี template.json) =====
PAPER_W, PAPER_H = 1654, 2339
DEFAULT_TEMPLATE = {
    "paper": {"width": PAPER_W, "height": PAPER_H},
    "sections": [
        {
            "name": "student_id",
            "roi": {"x0": 80, "y0": 300, "x1": 320, "y1": 780},
            "digits": 6,
            "choices_per_digit": 10,
            "layout": "cols-then-rows",
            "bubble": {
                "area_min": 150, "area_max": 3500,
                "circularity_min": 0.55,
                "fill_method": "ratio", "fill_threshold": 0.50,
                "row_tol": 14, "col_tol": 20
            }
        },
        {
            "name": "answers",
            "roi_blocks": [
                {"x0": 400, "y0": 250,  "x1": 620,  "y1": 1000, "q_start": 1,  "q_end": 10},
                {"x0": 700, "y0": 250,  "x1": 920,  "y1": 1000, "q_start": 11, "q_end": 20},
                {"x0": 1000,"y0": 250,  "x1": 1220, "y1": 1000, "q_start": 21, "q_end": 30},
                {"x0": 400, "y0": 1150, "x1": 620,  "y1": 1900, "q_start": 31, "q_end": 40},
                {"x0": 700, "y0": 1150, "x1": 920,  "y1": 1900, "q_start": 41, "q_end": 50},
                {"x0": 1000,"y0": 1150, "x1": 1220, "y1": 1900, "q_start": 51, "q_end": 60}
            ],
            "choices_per_q": 5,
            "bubble": {
                "area_min": 180, "area_max": 4500,
                "circularity_min": 0.60,
                "fill_method": "ratio", "fill_threshold": 0.55,
                "row_tol": 16, "col_tol": 16
            }
        }
    ]
}
TEMPLATE = json.loads(json.dumps(DEFAULT_TEMPLATE))  # copy
if os.path.exists(TEMPLATE_PATH):
    try:
        TEMPLATE = json.load(open(TEMPLATE_PATH, "r", encoding="utf-8"))
    except Exception:
        pass

ANSWER_KEY = None  # ใส่ "ABCDE"*12 ถ้าจะคิดคะแนน

# ===== utils =====
def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1); d = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(d)]; rect[3] = pts[np.argmax(d)]
    return rect

def find_four_markers(img_bgr: np.ndarray):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(cv2.GaussianBlur(gray,(5,5),0),255,
                               cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,31,10)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3,3),np.uint8), 1)
    cnts,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cands=[]
    for c in cnts:
        a=cv2.contourArea(c)
        if 500<a<50000:
            peri=cv2.arcLength(c,True); approx=cv2.approxPolyDP(c,0.02*peri,True)
            if len(approx)==4:
                x,y,w,h=cv2.boundingRect(approx)
                asp=w/float(h)
                if 0.7<=asp<=1.3: cands.append((a,approx.reshape(4,2)))
    if len(cands)<4: return None
    cands=sorted(cands,key=lambda x:x[0],reverse=True)[:4]
    pts=np.array([c[1].mean(axis=0) for c in cands],dtype=np.float32)
    H,W=gray.shape; ordered=[]
    for cx,cy in [(0,0),(W,0),(W,H),(0,H)]:
        d=np.linalg.norm(pts-np.array([cx,cy]),axis=1); idx=int(np.argmin(d))
        ordered.append(cands[idx][1]); pts[idx]=np.array([1e9,1e9])
    return np.array([p.mean(axis=0) for p in ordered],dtype=np.float32)

def warp_to_canvas(img_bgr: np.ndarray) -> np.ndarray:
    corners=find_four_markers(img_bgr)
    if corners is None: return cv2.resize(img_bgr,(PAPER_W,PAPER_H))
    src=order_points(corners)
    dst=np.array([[0,0],[PAPER_W-1,0],[PAPER_W-1,PAPER_H-1],[0,PAPER_H-1]],dtype=np.float32)
    M=cv2.getPerspectiveTransform(src,dst)
    return cv2.warpPerspective(img_bgr,M,(PAPER_W,PAPER_H))

def binarize(img_bgr: np.ndarray) -> np.ndarray:
    gray=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY)
    gray=cv2.createCLAHE(2.0,(8,8)).apply(gray)
    th=cv2.adaptiveThreshold(cv2.GaussianBlur(gray,(5,5),0),255,
                             cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,31,10)
    return cv2.morphologyEx(th,cv2.MORPH_OPEN,np.ones((3,3),np.uint8),1)

class BubbleParams:
    def __init__(self, **kw):
        self.area_min=kw.get("area_min",160)
        self.area_max=kw.get("area_max",4200)
        self.circularity_min=kw.get("circularity_min",0.6)
        self.fill_method=kw.get("fill_method","ratio")
        self.fill_threshold=kw.get("fill_threshold",0.55)
        self.row_tol=kw.get("row_tol",14)
        self.col_tol=kw.get("col_tol",16)

def is_circle(cnt,p:BubbleParams)->bool:
    a=cv2.contourArea(cnt)
    if not(p.area_min<=a<=p.area_max): return False
    peri=cv2.arcLength(cnt,True)
    if peri==0: return False
    circ=4*np.pi*a/(peri*peri)
    return circ>=p.circularity_min

def extract_bubbles(bin_img:np.ndarray,roi:Tuple[int,int,int,int],p:BubbleParams):
    x0,y0,x1,y1=roi
    crop=bin_img[y0:y1,x0:x1]
    cnts,_=cv2.findContours(crop,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    items=[]
    for c in cnts:
        if not is_circle(c,p): continue
        x,y,w,h=cv2.boundingRect(c); cx,cy=x+w/2,y+h/2
        items.append({"cnt":c,"center":(cx,cy)})
    items.sort(key=lambda b:(round(b["center"][1]/max(1,p.row_tol)), b["center"][0]))
    return items

def group_by_rows(items,row_tol):
    rows,cur=[],[]
    for b in items:
        if not cur: cur=[b]; continue
        if abs(b["center"][1]-cur[-1]["center"][1])<=row_tol: cur.append(b)
        else: rows.append(cur); cur=[b]
    if cur: rows.append(cur)
    return [sorted(r,key=lambda x:x["center"][0]) for r in rows]

def fill_score(bin_img,cnt,method="ratio")->float:
    mask=np.zeros(bin_img.shape,np.uint8); cv2.drawContours(mask,[cnt],-1,255,-1)
    region=bin_img[mask==255]
    if region.size==0: return 0.0
    if method=="mean": return float(region.mean())/255.0
    return float(np.count_nonzero(region==255))/float(region.size)

def decide_mark(scores:List[float],thr:float,margin:float=0.12)->Optional[int]:
    if not scores: return None
    arr=np.array(scores,float); top=int(arr.argmax())
    if arr[top]<thr: return None
    if len(arr)>=2 and arr[top]-sorted(arr,reverse=True)[1]<margin: return None
    return top

def read_student_id(bin_img:np.ndarray,sec:Dict)->str:
    p=BubbleParams(**sec["bubble"]); r=sec["roi"]
    bubbles=extract_bubbles(bin_img,(r["x0"],r["y0"],r["x1"],r["y1"]),p)
    rows=group_by_rows(bubbles,p.row_tol)
    allc=[(b["center"][0],b) for r in rows for b in r]
    xs=np.array([x for x,_ in allc])
    if xs.size==0: return ""
    qs=np.quantile(xs,[i/6 for i in range(1,6)])
    cols=[[] for _ in range(6)]
    for x,b in allc:
        idx=int(np.searchsorted(qs,x,side="right")); cols[idx].append(b)
    digits=[]
    for col in cols:
        if not col: digits.append(None); continue
        col=sorted(col,key=lambda v:v["center"][1])
        scores=[fill_score(bin_img,b["cnt"],p.fill_method) for b in col]
        d=decide_mark(scores,p.fill_threshold,0.10)
        digits.append(d if d is not None else None)
    return "".join([str(d) if d is not None else "-" for d in digits])

CHOICE_LABELS="ABCDE"

def read_answers(bin_img:np.ndarray,sec:Dict)->Dict[int,Optional[str]]:
    p=BubbleParams(**sec["bubble"]); out={}
    for blk in sec["roi_blocks"]:
        bubbles=extract_bubbles(bin_img,(blk["x0"],blk["y0"],blk["x1"],blk["y1"]),p)
        rows=group_by_rows(bubbles,p.row_tol)
        n_q=blk["q_end"]-blk["q_start"]+1
        if len(rows)!=n_q and len(rows)>0:
            idxs=np.linspace(0,len(rows)-1,n_q).round().astype(int).tolist()
            rows=[rows[i] for i in idxs]
        for i,row in enumerate(rows):
            cps=sec["choices_per_q"]
            if len(row)!=cps and len(row)>0:
                idxs=np.linspace(0,len(row)-1,cps).round().astype(int).tolist()
                row=[row[j] for j in idxs]
            scores=[fill_score(bin_img,b["cnt"],p.fill_method) for b in row]
            sel=decide_mark(scores,p.fill_threshold)
            out[blk["q_start"]+i]=CHOICE_LABELS[sel] if sel is not None else None
    return out

def score_answers(answers:Dict[int,Optional[str]], key:Optional[str])->Dict:
    if not key: return {"total": len(answers), "correct": None, "percent": None}
    key=key.strip().upper(); total=min(len(answers),len(key))
    correct=sum(1 for i in range(1,total+1) if answers.get(i)==key[i-1])
    return {"total": total, "correct": correct, "percent": round(100.0*correct/max(1,total),2)}

def im_to_base64(img_bgr: np.ndarray) -> str:
    img_rgb=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)
    buf=io.BytesIO(); Image.fromarray(img_rgb).save(buf,format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ===== static =====
app.mount("/static", StaticFiles(directory="static"), name="static")
@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(open("static/index.html","r",encoding="utf-8").read())

# ===== template API =====
@app.get("/api/template")
def get_template():
    return JSONResponse(TEMPLATE)

@app.post("/api/template")
def set_template(payload: dict = Body(...)):
    global TEMPLATE
    TEMPLATE = payload
    json.dump(TEMPLATE, open(TEMPLATE_PATH,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
    return {"ok": True}

# ===== grade API =====
@app.post("/api/grade")
async def api_grade(image: UploadFile = File(...), key: Optional[str] = None):
    content = await image.read()
    file = np.frombuffer(content, np.uint8)
    bgr = cv2.imdecode(file, cv2.IMREAD_COLOR)
    if bgr is None:
        return JSONResponse({"error": "อ่านไฟล์ภาพไม่สำเร็จ"}, status_code=400)

    warped = warp_to_canvas(bgr)
    binary = binarize(warped)

    sid_sec = next(s for s in TEMPLATE["sections"] if s["name"]=="student_id")
    ans_sec = next(s for s in TEMPLATE["sections"] if s["name"]=="answers")

    student_id = read_student_id(binary, sid_sec)
    answers = read_answers(binary, ans_sec)
    scoring = score_answers(answers, key if key else ANSWER_KEY)

    viz = warped.copy()
    # draw ROI
    cv2.rectangle(viz,(sid_sec["roi"]["x0"],sid_sec["roi"]["y0"]),
                       (sid_sec["roi"]["x1"],sid_sec["roi"]["y1"]),(0,128,255),2)
    cv2.putText(viz,f"SID:{student_id}",(sid_sec["roi"]["x0"],sid_sec["roi"]["y0"]-10),
                cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,128,255),2,cv2.LINE_AA)
    for blk in ans_sec["roi_blocks"]:
        cv2.rectangle(viz,(blk["x0"],blk["y0"]),(blk["x1"],blk["y1"]),(100,200,100),2)

    # save debug
    ts=datetime.now().strftime("%Y%m%d-%H%M%S")
    cv2.imwrite(os.path.join(DEBUG_DIR,f"{ts}_warped.png"), warped)
    cv2.imwrite(os.path.join(DEBUG_DIR,f"{ts}_binary.png"), binary)
    cv2.imwrite(os.path.join(DEBUG_DIR,f"{ts}_viz.png"), viz)

    return JSONResponse({
        "student_id": student_id,
        "answers": {str(k): v for k,v in sorted(answers.items())},
        "scoring": scoring,
        "images": {
            "warped_png_base64": im_to_base64(warped),
            "binary_png_base64": im_to_base64(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)),
            "visualize_png_base64": im_to_base64(viz),
        }
    })

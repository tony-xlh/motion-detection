#coding=utf8
from dbr import *
import argparse
import datetime
import time
import cv2


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--index", type=int, default=0, help="camera index")
ap.add_argument("-f", "--fps", type=float, default=25.0, help="frame per second")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())

fps = args["fps"]
camera = cv2.VideoCapture(args["index"])

size = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print('size:'+repr(size))

reader = BarcodeReader()
reader.init_license("t0069fQAAADni8mnJeS0cnoLp85KEXFCh78ltXDT3x52OWWW0qsnvVBOkG7nz+do12XxdqoHCJQ+U+Bbg+RPP/7nyQsQkDtOC")

detected_barcode_text=""
history={}

reference_frame = None
reference_frame_gray = None
occupied=False

capture_times = 0

def show_detected_barcode_frame(frame, resized_width,resized_height, result):
    frame_clone=frame.copy()
    points=result.localization_result.localization_points;
    cv2.line(frame_clone,points[0],points[1], (0, 255, 0), 2)
    cv2.line(frame_clone,points[1],points[2], (0, 255, 0), 2)
    cv2.line(frame_clone,points[2],points[3], (0, 255, 0), 2)
    cv2.line(frame_clone,points[3],points[0], (0, 255, 0), 2)
    cv2.putText(frame_clone, "Text: {}".format(result.barcode_text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frame_clone, "Confidence: {}".format(result.extended_results[0].confidence), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    resized=cv2.resize(frame_clone,(resized_width,resized_height))
    cv2.imshow("Detected Frame", resized)
    return resized

while True:
    start = time.time()
    grabbed, frame = camera.read()
    text = "Unoccupied"
    if grabbed == False:
        break
        
    end = time.time()
    
    seconds = end - start
    if seconds < 1.0 / fps:
        time.sleep(1.0 / fps - seconds)
    
    width=frame.shape[1]
    height=frame.shape[0]
    
    
    resized_width=500
    scale=resized_width/width
    resized_height=int(height*scale)
    resized = cv2.resize(frame, (resized_width, resized_height))
    resized_clone = resized.copy()
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    if capture_times<25:
        reference_frame = resized
        reference_frame_gray = gray
        capture_times = capture_times+1
        continue
    

    
    frame_delta = cv2.absdiff(reference_frame_gray, gray)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if cv2.contourArea(c) < args["min_area"]:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Occupied"
        
    # draw the text and timestamp on the frame
    cv2.putText(resized, "Status: {}".format(text), (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(resized, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
        (10, resized.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        
    if text == "Occupied":
        if detected_barcode_text=="":
            text_results = reader.decode_buffer(frame)
            if text_results!=None:
                text_result=text_results[0]
                barcode_text=text_result.barcode_text
                detected_barcode_text=barcode_text
                print("Found barcode: "+barcode_text)
                confidence = text_result.extended_results[0].confidence;
                print("Confidence: "+str(confidence))
                if confidence<30:
                    print("Confidence low. Abandoned.")
                    continue

                img = show_detected_barcode_frame(frame,resized_width,resized_height,text_result)
                
                #write frames to files
                cv2.imwrite("reference_frame.jpg",reference_frame)
                cv2.imwrite("occupied.jpg",resized)
                cv2.imwrite("frame.jpg",resized_clone)
                cv2.imwrite("thresh.jpg",thresh)
                cv2.imwrite("frame_delta.jpg",frame_delta)
                cv2.imwrite("detected_frame.jpg",img)
                
                #add to history
                times=0
                if barcode_text in history:
                    times=history[barcode_text]
                history[barcode_text]=times+1
                print("Scan history:")
                print(history)
    else:
        detected_barcode_text=""
        
    # show the frame and record if the user presses a key
    cv2.imshow("Reference Frame", reference_frame)
    cv2.imshow("Video Stream", resized)
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Frame Delta", frame_delta)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key is pressed, break from the lop
    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
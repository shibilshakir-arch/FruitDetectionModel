import re
import cv2
from tflite_runtime.interpreter import Interpreter
import numpy as np

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

def input_labels(path='/home/shibil/Desktop/new_model/labelmap.txt'):
  """Loads the labelmap file. Supports files with or without index numbers."""
  with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    labelmap = {}
    for row_number, content in enumerate(lines):
      pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
      if len(pair) == 2 and pair[0].strip().isdigit():
        labelmap[int(pair[0])] = pair[1].strip()
      else:
        labelmap[row_number] = pair[0].strip()
  return labelmap

def input_tensor(interpreter, image):
  """Sets the input tensor."""
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = np.expand_dims((image-255)/255, axis=0)


def output_tensor(interpreter, index):
  """Returns the output tensor at the given index."""
  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor


def define_objects(interpreter, image, threshold):
  """Returns a list of detection objectName, each a dictionary of object info."""
  input_tensor(interpreter, image)
  interpreter.invoke()
  # Get all output details
  output_box = output_tensor(interpreter, 1)
  output_class = output_tensor(interpreter, 3)
  output_score = output_tensor(interpreter, 0)
  cnt = int(output_tensor(interpreter, 2))

  objectNames = []
  for i in range(cnt):
    if output_score[i] >= threshold:
      result = {'bounding_box': output_box[i], 'class_id': output_class[i],'score': output_score[i]}
      objectNames.append(objectName)
  return objectNames
  

try:
        sqliteConnection = sqlite3.connect('/home/shibil/fruitsdb.db', timeout=20)
        cursor = sqliteConnection.cursor()
        print("Connected to SQLite")

        sqlite_select_query = """SELECT price from fruits where name='?'", objectNames""
        cursor.execute(sqlite_select_query)
        objectprice = cursor.fetchone()
        
        sqlite_select_query2 = """SELECT weight from fruits where name='?'", objectNames""
        cursor.execute(sqlite_select_query2)
        objectweight = cursor.fetchone()
        cursor.close()

except sqlite3.Error as error:
        print("Failed to read data from sqlite table", error)
finally:
        if sqliteConnection:
            sqliteConnection.close()
            print("The Sqlite connection is closed")


def mainClass():
    frame_calc = 1
    freq = cv2.getTickFrequency()
    labelmap = input_labels()
    interpreter = Interpreter('/home/shibil/Desktop/new_model/detect.tflite')
    interpreter.allocate_tensors()
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        t1 = cv2.getTickCount()
        ret, frame = cap.read()
        imagein = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (320,320))
        obj = define_objects(interpreter, imagein, 0.8)
        print(obj)

        for objectName in obj:
            ymin, xmin, ymax, xmax = objectName['bounding_box']
            xmin = int(max(1,xmin * CAMERA_WIDTH))
            xmax = int(min(CAMERA_WIDTH, xmax * CAMERA_WIDTH))
            ymin = int(max(1, ymin * CAMERA_HEIGHT))
            ymax = int(min(CAMERA_HEIGHT, ymax * CAMERA_HEIGHT))
            
            cv2.rectangle(frame,(xmin, ymin),(xmax, ymax),(0,255,0),3)
            cv2.putText(frame,labelmap[int(objectName['class_id'])],(xmin, min(ymax, CAMERA_HEIGHT-20)), cv2.FONT_HERSHEY_COMPLEX, 1,(255,255,255),2,cv2.LINE_AA) 

        cv2.putText(frame, 'FPS: {0:.2f}'.format(frame_calc), (30,50), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0),2,cv2.LINE_AA)
        
        cv2.imshow("OUTPUT WINDOW",frame)
        
        print("The total price for the " + objectNames + " is: " + objectprice + " DHS / " + objectweight + " KILOGRAMS" )

        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_calc = 1/time1

        if cv2.waitKey(10) & 0xFF ==ord('q'):
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    mainClass()
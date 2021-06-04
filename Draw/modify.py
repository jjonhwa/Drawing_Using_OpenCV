import cv2
import pandas as pd
import numpy as np

def image_scale(img) :
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return thresh

def image_scale_sub(img) :
  '''
  If you had a table with too soft line, you can use this code
  Canny(edge algorithm) is very good to recognize line.
  '''
  canny = cv2.Canny(img, 50, 50)
  thresh = cv2.threshold(canny,  0, 255, cv2.THRESH_OTSU)[1]
  return thresh

def cut_image(scale_img, threshold=800) :
  '''
  threshold : The difference between the y-coordinate value of the line and the line
  
  If you had only table without another text, you can skip this code.
  '''
  horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (81,1))
  detect_horizontal = cv2.morphologyEx(scale_img, cv2.MORPH_OPEN,
                                        horizontal_kernel, iterations = 3)
  cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]

  for i in range(len(cnts)) :
    if i == 0 :
      continue
    first_line = cv2.boundingRect(cnts[i-1])[1]
    second_line = cv2.boundingRect(cnts[i])[1]
    
    if abs(first_line - second_line) >= threshold : 
      start_line = second_line-5
      break
  clean = scale_img[start_line:, :]
  return start_line, clean

def remove_horizontal(scale_image) :
  clean = scale_image.copy()
  horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
  detect_horizontal = cv2.morphologyEx(scale_image, cv2.MORPH_OPEN,
                                        horizontal_kernel, iterations  = 2)
  cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]
  
  for c in cnts :
    cv2.drawContours(clean, [c], -1, 0, 3)
      
  return clean

def remove_vertical(scale_image) :
  clean = scale_image.copy()
  vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
  detect_vertical = cv2.morphologyEx(scale_image, cv2.MORPH_OPEN,
                                      vertical_kernel, iterations = 3)
  cnts = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]
  
  for c in cnts :
      cv2.drawContours(clean, [c], -1,  0, 3)
      
  return clean

def search_x(scale_image) :
  vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
  detect_vertical = cv2.morphologyEx(scale_image, cv2.MORPH_OPEN,
                                    vertical_kernel, iterations = 3)
  cnts = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]

  x_list = []
  for i in range(len(cnts)) :
    x_list.append(list(cv2.boundingRect(cnts[i][0])))
  
  tmp = pd.DataFrame(x_list)
  max_x = np.max(tmp[0])
  min_x = np.min(tmp[0])
  return min_x, max_x

def search_y(scale_image) :
  horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
  detect_horizontal = cv2.morphologyEx(scale_image, cv2.MORPH_OPEN,
                                      horizontal_kernel, iterations = 3)
  cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]

  y_list = []
  for i in range(len(cnts)) :
    y_list.append(list(cv2.boundingRect(cnts[i][0])))
  
  tmp = pd.DataFrame(y_list)
  max_y = np.max(tmp[1])
  min_y = np.min(tmp[1])
  return min_y, max_y

def dilate_and_erode(scale_image, dil_iterations = 5, erode_iterations = 5) :
  '''
  dil_iterations : how many you run dilate
  erode_iterations : how many you run erode
  '''
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
  dilate = cv2.dilate(scale_image, kernel, anchor = (-1, -1), iterations = dil_iterations)
  erode = cv2.erode(dilate, kernel, anchor = (-1, -1), iterations = erode_iterations)
  
  cnts = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]
  
  return cnts

def preprocess_image(contour) :
  '''
  Using contour value, Find where we have to draw some line.
  contour to location
  '''
  final_list = []
  for c in contour : 
    final_list.append(list(cv2.boundingRect(c)))
      
  final_data = pd.DataFrame()
  for i in range(len(final_list)) :
    new_row = final_list[i]
    new_row = pd.DataFrame(new_row).T
    
    final_data = pd.concat([final_data, new_row])
    
  final_data.reset_index(drop = True, inplace = True)
  final_data.columns = ['x', 'y', 'w', 'h']
  
  tmp = final_data.groupby('y').agg({'h' : 'max'})
  temp = tmp.reset_index()
  
  
  drop_list = []
  for i in range(len(temp)) :
    if i == 0 :
      continue
    if abs(temp['y'][i-1] - temp['y'][i]) <= 10 and \
      abs(temp['h'][i-1] - temp['h'][i]) <= 25:
      if temp['h'][i-1] + temp['y'][i-1] >= temp['h'][i] + temp['y'][i]:
        drop_list.append(i)
      else :
        drop_list.append(i-1)
              
  temp = temp.drop(drop_list, axis = 0)
  temp.reset_index(drop = True, inplace = True)
  
  drop_list = []
  for i in range(len(temp)) :
    if i == 0 :
      continue
    if abs(temp['y'][i-1] - temp['y'][i]) <= 15 and \
      abs(temp['h'][i-1] - temp['h'][i]) <= 25:
      if temp['h'][i-1] + temp['y'][i-1] >= temp['h'][i] + temp['y'][i] :
        drop_list.append(i)
      else :
        drop_list.append(i-1)
  temp = temp.drop(drop_list, axis = 0)
  temp.reset_index(drop = True, inplace = True)

  drop_list = []
  for i in range(len(temp)) :
    if i == 0 :
      continue
    if abs(temp['y'][i-1] - temp['y'][i]) <= 25 and \
      abs(temp['h'][i-1] - temp['h'][i]) <= 25:
      if temp['h'][i-1] + temp['y'][i-1] >= temp['h'][i] + temp['y'][i] :
        drop_list.append(i)
      else :
        drop_list.append(i-1)
  temp = temp.drop(drop_list, axis = 0)
  temp.reset_index(drop = True, inplace = True)
  
  temp['yh'] = temp['y'] + temp['h']
  temp = temp.sort_values('yh')

  drop_list = []
  for i in range(len(temp['yh'])) :
    if i == 0 :
      continue
    if abs(temp['yh'][i-1] - temp['yh'][i]) <= 25 :
      drop_list.append(i-1)
  temp = temp.drop(drop_list, axis = 0)
  temp.reset_index(drop = True, inplace = True)

  temp = temp.drop(['yh'], axis = 1)
  final = pd.merge(temp, final_data)
  return final

def draw_line(image, contour, data, min_x, max_x):
  '''
  contour : contour
  data : Coordinate values ​​to draw the line
  min_x : mininum value of x location
  max_x : maximum value of x location

  return : y location where we have to draw
  '''
  draw_line_list = []
  for c in contour :
    for i in range(len(data)) :
      if i == len(data) - 1 :
        x = data['x'][i]
        y = data['y'][i]
        w = data['w'][i]
        h = data['h'][i]
      else :
        x_after = data['x'][i+1]
        y_after = data['y'][i+1]
        w_after = data['w'][i+1]
        h_after = data['h'][i+1]
        x_before = data['x'][i]
        y_before = data['y'][i]
        w_before = data['w'][i]
        h_before = data['h'][i]
        if abs((y_before+h_before) - (y_after + h_after)) < 25 :
          x = data['x'][i+1]
          y = data['y'][i+1]
          w = data['w'][i+1]
          h = data['h'][i+1]
        else :
          x = data['x'][i]
          y = data['y'][i]
          w = data['w'][i]
          h = data['h'][i]
      area = cv2.contourArea(c)
      if area > 40 :
        ROI = image[y:y+h, x:x+w]
        ROI = cv2.GaussianBlur(ROI, (7,7), 0)
        draw_line_list.append(y+h-2)
  return draw_line_list














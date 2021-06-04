import cv2

from . import modify as md

def go_draw(image_path) :
  img = cv2.imread(image_path, cv2.IMREAD_COLOR)
  scale_image = md.image_scale(img)
  start_line, scale_cut_image = md.cut_image(scale_image)
  min_x, max_x = md.search_x(scale_image)
  scale_cut_image = md.remove_horizontal(scale_cut_image)
  scale_cut_image = md.remove_vertical(scale_cut_image)
  contour = md.dilate_and_erode(scale_cut_image, 5, 2)
  final = md.preprocess_image(contour)
  draw_line_list = md.draw_line(img, contour, final, min_x, max_x)

  for i in range(len(draw_line_list)) :
    y_h = draw_line_list[i]
    cv2.line(img, (min_x, y_h+start_line), (max_x, y_h+start_line), (0,0,0), 1)

  return img
#!/usr/bin/env python

import cv2, sys, os

if __name__ == '__main__':
  if len(sys.argv) < 2:
    print('Usage {} video_file'.format(sys.argv[0]))
    sys.exit(1)
          
  input_file = sys.argv[1]
  input_file_base = input_file.split('/')[-1]
  output_folder = '{}_frames'.format(input_file_base)
  os.mkdir(output_folder)
  print('Output files will be in {}'.format(output_folder))

  vidcap = cv2.VideoCapture(input_file)
  count = 0
  while True:
    success, image = vidcap.read()
    if not success:
      break
    count += 1
    cv2.imwrite(output_folder + '/frame%03d.jpg' % count, image)

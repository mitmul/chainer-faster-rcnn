// --------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see LICENSE for details]
// Written by Ross Girshick
// https://github.com/rbgirshick/py-faster-rcnn
// --------------------------------------------------------

void _nms(int* keep_out, int* num_out, const float* boxes_host, int boxes_num,
          int boxes_dim, float nms_overlap_thresh, int device_id);

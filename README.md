# PoseMachines
Learning pose machines

https://www.ri.cmu.edu/pub_files/2014/7/poseMachines.pdf

https://arxiv.org/pdf/1312.4659.pdf

https://github.com/shihenw/convolutional-pose-machines-release

https://arxiv.org/pdf/1612.00137.pdf

https://infinitescript.com/2019/07/compile-caffe-without-root-privileges/

https://stackoverflow.com/questions/65605972/cmake-unsupported-gnu-version-gcc-versions-later-than-8-are-not-supported

cmake .. -DPYTHON_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")  -DPYTHON_LIBRARY=$(python3 -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") -DPYTHON_EXECUTABLE:FILEPATH=/usr/lib  -DNUMPY_INCLUDE_DIR=/usr/lib/python3.8/site-packages -DNUMPY_VERSION=1.22.3

export PYTHONPATH="${PYTHONPATH}:/mnt/c/Users/matth/OneDrive/Documents/Storage/CSCI_5561/Final_Project/cpm/caffe/python"

pytorch implementations of cpm: 
https://github.com/SDUerFH/CPM
https://github.com/shshin1210/convolutional_pose_machines
https://github.com/jinchiniao

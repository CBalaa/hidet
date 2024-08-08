rm ./hidet_KGC-0.3.1-py3-none-any.whl
python setup.py bdist_wheel 
cd build
cmake ..
make
mkdir lib/hidet/include
cp -r ../include/* lib/hidet/include
cd ..
python ./setup.py bdist_wheel 

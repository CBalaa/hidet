python setup.py bdist_wheel 
cd build
cmake ..
make
mkdir lib/hidet/include
cp -r ../include/* lib/hidet/include
cd ..
python ./setup.py bdist_wheel 

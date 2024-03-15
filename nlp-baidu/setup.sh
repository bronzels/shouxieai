pip install paddlepaddle
pip install paddlenlp
#import paddlenlp
#paddlepaddle ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.20' not found
strings /usr/local/miniconda3/lib/libstdc++.so.6.0.29 | grep GLIBCXX|grep 20
ls -l /usr/lib64/libstdc++.so.6
#/usr/lib64/libstdc++.so.6 -> libstdc++.so.6.0.19
mv /usr/lib64/libstdc++.so.6 /usr/lib64/libstdc++.so.6.bk
ln -s /usr/local/miniconda3/lib/libstdc++.so.6.0.29 /usr/lib64/libstdc++.so.6
#export LD_LIBRARY_PATH=/usr/local/miniconda3/lib:$LD_LIBRARY_PATH
#不能用export引入其他so导致内存错误
#缓存挪到数据盘以免系统盘爆掉
mkdir -p /data0/paddlenlp
ln -s /data0/paddlenlp ~/.paddlenlp

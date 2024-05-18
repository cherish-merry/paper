## 论文数据说明

### 原始数据集说明

CICIDS2017数据集是一个用于网络入侵检测的公开的数据集，该数据集包含了在真实的网络环境下产生的网络流量数据。CICIDS2017提供了原始PCPA文件和特征工程源码CICFlowMeter来构建内核态网络流特征数据集。

数据集下载地址：https://www.unb.ca/cic/datasets/ids-2017.html



### 特征提取工具说明

CICFlowMeter项目地址：https://github.com/CanadianInstituteForCybersecurity/CICFlowMeter

运行环境：Java/IDEA

```
//linux:
$ sudo bash
$ gradle execute

//windows:
$ gradlew execute
```

### 内核态特征数据集构建说明

使用CICFlowMeter特征提取工具提取网络流特征

选取了7月3日的正常流量，7月5日、7月7日 DoS/DDoS攻击相关网络流量

| 特征名称      | 特征含义                 |
| ------------- | ------------------------ |
| Flow Duration | 网络流持续时间           |
| Flow Byt/s    | 每秒处理网络流字节大小   |
| Flow Pkt/s    | 每秒处理网络数据包数量   |
| Flow IAT Mean | 相邻数据包时间间隔平均值 |
| Flow IAT Std  | 相邻数据包时间间隔方差   |
| Flow IAT Max  | 相邻数据包时间间隔最大值 |
| Flow IAT Min  | 相邻数据包时间间隔最小值 |
| Pkt Num       | 数据包数量               |
| Pkt Len Min   | 数据包长度最小值         |
| Pkt Len Max   | 数据包长度最大值         |
| Pkt Len Mean  | 数据包长度平均值         |
| Pkt Len Std   | 数据包长度方差           |
| FIN Flag Cnt  | FIN标志位数量            |
| SYN Flag Cnt  | SYN标志位数量            |
| RST Flag Cnt  | RST标志位数量            |
| PSH Flag Cnt  | PSH标志位数量            |

| 流量类型         | 提取记录数量  (DDoS.csv) | 选取记录数量  （Sample-5w.csv） |
| ---------------- | ------------------------ | ------------------------------- |
| BENIGN           | 364682                   | 50000                           |
| DoS Hulk         | 151059                   | 5000                            |
| DDoS LOIT        | 92304                    | 5000                            |
| DoS Slowhttptest | 5279                     | 5000                            |
| DoS Slowloris    | 5598                     | 5000                            |
| DoS GoldenEye    | 7746                     | 5000                            |



## 论文算法程序说明

### 知识蒸馏算法运行环境

**Python 版本：python 3.9** 

**安装Python依赖：pip install -r requirements.txt**

```
sklearn==0.0.post4
xgboost==2.0.1
torch==1.12.0
torchvision== 0.13.0
pandas==1.2.4
numpy== 1.23.5
seaborn==0.12.1
```

**运行程序：python kd_mlp.py**

### XDP 运行环境

**内核版本要求：推荐 Ubuntu 20.04 5.13.0-30-generic**

**安装依赖**

```shell
apt install bison cmake flex libedit-dev libllvm12 llvm-12-dev libclang-12-dev libelf-dev netperf iperf3 arping gcc g++ python3-distutils
-DENABLE_LLVM_SHARED=1
```

**构建BCC项目**

```shell
git clone -b v0.18.0 https://github.com/iovisor/bcc.git
mkdir bcc/build; cd bcc/build
cmake ..
make
sudo make install
cmake -DPYTHON_CMD=python3 .. # build python3 binding
pushd src/python/
make
sudo make install
popd

vim /usr/src/linux-hwe-5.13-headers-5.13.0-30/include/linux/compiler-clang.h
#define __HAVE_BUILTIN_BSWAP32__
#define __HAVE_BUILTIN_BSWAP64__
#define __HAVE_BUILTIN_BSWAP16__
```

**运行XDP程序**

```shell
python3 xdp_loader.py  <网卡>
```



## 论文应用说明

**安装prometheus**

```
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

```

**启动prometheus**

```
helm install -n monitor prometheus prometheus-community/prometheus --set alertmanager.enabled=false --set kube-state-metrics.enabled=false --set prometheus-pushgateway.enabled=false
```

**安装grafana**

```
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update
```

**启动grafana**

```
helm install -n monitor grafana grafana/grafana
```

**运行XDP程序上传数据**

```
python3 xdp.py  <网卡>
```



## 实验结果

**使用ddos-preformence.py生成**

![ddos-performance](https://cdn.jsdelivr.net/gh/cherish-merry/img@master/blog/ddos-performance-20240518%2016:14:27.png)



**使用ddos-cpu.py 生成**

![ddos-cpu](/Users/ckz/Desktop/paper/论文结果/img/ddos-cpu.png)

**使用ddos-mitigate.py 生成**

![ddos-mitigate](https://cdn.jsdelivr.net/gh/cherish-merry/img@master/blog/ddos-mitigate-20240518%2016:12:48.png)

**使用ddos-mutilcore生成**

![multicore](https://cdn.jsdelivr.net/gh/cherish-merry/img@master/blog/multicore-20240518%2016:13:38.png)

**使用kd-loss.py生成**

![kd-loss](https://cdn.jsdelivr.net/gh/cherish-merry/img@master/blog/kd-loss-20240518%2016:14:10.png)
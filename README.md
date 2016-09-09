# Jetson TX1 - Tensorflow r0.8 설치

작성자 - 대덕SW마이스터고등학교 임베디드과 전승현 (e-mail : shtowever@gmail.com)  

이 문서는 이 Jetson TX1이라는 보드에 Tensoflow r0.8을 설치하는 방법을 작성한 문서 입니다.   
Jetson TX1은 엔비디아에서 만든 임베디드 보드로 256개의 cuda 코어가 있는 gpu가 달려있는 것이 특징인 보드입니다.   
(대략 geforce gt630m과 비슷한 성능이라고 합니다)  

## 참고 문서

다음 문서들을 참고하고 작성하였습니다.

https://devtalk.nvidia.com/default/topic/901337/jetson-tx1/cuda-7-0-jetson-tx1-performance-and-benchmarks/  

https://www.tensorflow.org/versions/r0.8/get_started/index.html  

http://cudamusing.blogspot.kr/2015/11/building-tensorflow-for-jetson-tk1.html

http://cudamusing.blogspot.kr/2016/06/tensorflow-08-on-jetson-tk1.html

https://github.com/tensorflow/tensorflow/issues/851


2016–09-07에 작성된 문서입니다.

--------------------------------------------------------------------------------

# 설치 환경

우리가 사용하는 보드의 사양과 여러 툴의 버전은 다음과 같습니다.

## TX1 사양

     장치      |                          성능
:----------: | :----------------:
    GPU      | 1 TFLOP/s 256-core with NVIDIA Maxwell™ Architecture
    CPU      |                 64-bit ARM® A57 CPUs
  Memor y    |                4 GB LPDDR4 25.6 GB/s
Video decode |                       4K 60 Hz
Video encode |                       4K 30 Hz
    CSI      |             Up to 6 cameras 1400 Mpix/s
  Display    |          2x DSI, 1x eDP 1.4, 1x DP 1.2/HDMI
Connectivity |     802.11ac Wi-Fi and Bluetooth-enabled devices
 Networking  |                  1 Gigabit Ethernet
    PCIE     |                   Gen 2 1x1 + 1x4
  Storage    |                16 GB eMMC, SDIO, SATA
   Other     |        3x UART, 3x SPI, 4x I2C, 4x I2S, GPIOs

## 프로그램 버전

```
Ubuntu 14.04  
cuDNN 5.0  
CUDA 7.0  
Protobuf 3.0.0  
Java 8  
Bazel 0.2.0  
Tensorflow r0.8
```

--------------------------------------------------------------------------------

# JetPack fot L4T

Jetson TX1은 엔비디아 사이트에서 JetPack for L4T라는 툴을 지원합니다. 이 툴을 이용하면 os로 Ubuntu 14.04가 설치되며 cuDNN 5.0, CUDA 7.0 등이 설치됩니다.

밑의 링크로 자세한 설치 방법을 배울 수 있습니다.<br>
<http://docs.nvidia.com/jetpack-l4t/2_1/content/developertools/mobile/jetpack/jetpack_l4t/2.0/jetpack_l4t_install.htm>

--------------------------------------------------------------------------------

# 기본적인 세팅

Tensorflow r0.8을 빌드하기 위해서는 Protobuf와 Bazel이 필요하게 됩니다. 또한 이 툴을 설치하기 위해서 다양한 것들의 설치를 요구합니다.

```
$ sudo add-apt-repository ppa:webupd8team/java
$ sudo apt-get update
$ sudo apt-get install oracle-java8-installer
$ sudo apt-get install git zip unzip autoconf automake libtool curl zlib1g-dev
$ sudo apt-get install maven
```

혹시나 펜이 작동 안하여 cpu온도가 높아 꺼지는 경우가 있다면 다음 명령어로 팬을 작동 시킬 수 있습니다.

```
$ sudo echo 255 > /sys/kernel/debug/tegra_fan/target_pwm
```

--------------------------------------------------------------------------------

# Protobuf 설치

Protobuf는 Bazel의 third_party 요소로 Bazel에서 Protobuf의 arm 버전을 지원 하지 않으므로 직접 빌드를 하여야 합니다.

우선적으로 git에서 Protobuf 소스를 받아줍니다.

```
$ git clone https://github.com/google/protobuf.git
```

그 뒤 다음 일렬의 명령을 입력하여 주면 Protobuf가 설치되게 됩니다.

```
$ cd protobuf
$ ./autogen.sh
$ ./configure --prefix=/usr
$ make -j 4
$ sudo make install
```

이렇게 설치가 완료되면 Protobuf의 하위 디렉토리에 있는 java/core로 이동하여 다음 명령어를 입력합니다.

```
$ cd java/core
$ mvn package
```

protobuf-java-3.0.0.jar파일이target디렉토리에 생성된 것을 확인하였다면 빌드가 완료된 것입니다.

## 이름이 다를 수 있으나 protobuf-java-3.0.0 의 형식으로 나오면 됩니다.

--------------------------------------------------------------------------------

# Bazel 설치

이제 Bazel을 설치할 준비가 완료되었습니다. 저는 Bazel 0.2.0버전을 이용하였습니다.

```
$ git clone https://github.com/bazelbuild/bazel.git
$ cd bazel
$ git checkout tags/0.2.0
```

소스를 컴파일 하기 전에 방금 설치한 Protobuf를 Bazel의 third_party에 복사해주어야 합니다.

복사하기전 bazel을 컴파일 하려 들면 protoc_linux__xxxx_.exe 파일이 없다고 할것입니다. bazel이 요구하는 파일이름으로 protoc파일을 복사해주시기 바랍니다. 그 뒤 우리가 빌드하여 확인한 protobuf-java-3.0.0 파일 역시 복사하여 옮깁니다.

저의 경우 protoc-linux-arm32.exe였습니다만 다른 버전에서는 다른 이름일 수도 있습니다.

이것의 핵심은 bazel을 속이는것입니다. 만약 bazel이 비슷한 형식의 다른 파일을 요구한다면 그것으로 복사한 파일의 이름을 바꿔주세요.

```
$ cp /usr/bin/protoc   third_party/protobuf/protoc-linux-arm32.exe
$ cp ~/protobuf/java/core/target/protobuf-java-3.0.0.jar  third_party/protobuf/protobuf-java-3.0.0-alpha-3.jar
```

# cpu unknown 문제

jetson-tx1은 aarch64 cpu입니다.  
그러나 bazel을 컴파일 해보면 cpu unknown이라는 에러 메세지를 뿜어내며 사망하는데

bazel의 arm enum에 aarch64를 추가하여 해결 가능합니다.

다음 bazel 디렉토러의 두 파일을 수정 합니다.

```
src/main/java/com/google/devtools/build/lib/util/CPU.java
scripts/bootstrap/buildenv.sh
```

## CPU.java
```
ARM("arm", ImmutableSet.of("arm", "armv7l")),  
```
```
ARM("arm", ImmutableSet.of("arm", "armv7l", "aarch64")),
```

## buildenv

```
34 MACHINE_IS_ARM='no'
35 if [ "${MACHINE_TYPE}" = 'arm' -o "${MACHINE_TYPE}" = 'armv7l' -o ]; then
36     MACHINE_IS_ARM='yes'
37 fi
```

```
34 MACHINE_IS_ARM='no'
35 if [ "${MACHINE_TYPE}" = 'arm' -o "${MACHINE_TYPE}" = 'armv7l' -o "${MACHINE_TYPE}" = 'aarch64' ]; then
36     MACHINE_IS_ARM='yes'
37 fi

```


이제 bazel을 컴파일 할 수 있습니다.

```
$ ./compile.sh
```

컴파일이 완료되면 bazel-output이라는 디렉토리에 bazel이 있을 것입니다. 이를 PATH 환경변수에 추가해주시거나 /usr/local/bin으로 옮겨주세요.

```
$ sudo cp bazel /usr/local/bin
```

--------------------------------------------------------------------------------

# Tensorflow 설치

위의 과정을 다 완료하였으면 이제 Tensorflow 소스코드를 다운 받아 빌드 할 일만 남았습니다.

```
$ git clone -recurse-submodules https://github.com/tensorflow/tensorflow
$ cd tensorflow
$ git checkout r0.8
$ git submodule update –init
```

이제 빌드를 위한 Tensorflow를 설정 해주세요 각각 python과 gcc, cuda, cudnn의 위치를 설정하여야 할 것 입니다. 경로가 모두 정상이면 binary size에 5.3을 입력하면 설정이 완료됩니다.

```
Ubuntu@tegra-ubuntu:~/tensorflow$ TF_UNOFFICIAL_SETTING=1 ./configure
Please specify the location of python. [Default is /usr/bin/python]:
Do you wish to build TensorFlow with GPU support? [y/N] y
GPU support will be enabled for TensorFlow
Please specify which gcc nvcc should use as the host compiler. [Default is /usr/bin/gcc]:
Please specify the Cuda SDK version you want to use, e.g. 7.0\. [Leave empty to use system default]:
Please specify the location where CUDA  toolkit is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:
Please specify the Cudnn version you want to use. [Leave empty to use system default]:
Please specify the location where cuDNN  library is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:
Please specify a list of comma-separated Cuda compute capabilities you want to build with.
You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus.
Please note that each additional compute capability significantly increases your build time and binary size.
[Default is: "3.5,5.2"]: 5.3
Setting up Cuda include
Setting up Cuda lib
Setting up Cuda bin
Setting up Cuda nvvm
Configuration finished
```

### lib 경로 문제
이때 파일의 경로가 lib64인 경우 인식을 못하는 경우가 있습니다. 이 경우는 소프트 링크로 lib64폴더를 lib로 걸어주시면 해결됩니다.


설정이 완료되었다면 몇 가지 파일을 tensorflow의 third_party에 옮겨주세요

```
$ cd third_party/gpus/cuda
$ rm –fr bin nvvm
$ cp –R /usr/local/cuda-7.0/bin/ bin
$ cp –R /usr/local/cuda-7.0/nvvm nvvm
```

이제 몇가지의 파일을 수정하면 빌드 준비가 완료됩니다.  
다음 파일들을 수정할 예정입니다.

```
tensorflow/core/kernels/conv_ops_gpu_2.cu.cc
tensorflow/core/kernels/conv_ops_gpu_3.cu.cc
tensorflow/stream_executor/cuda/cuda_gpu_executor.cc
tensorflow/core/common_runtime/gpu/process_state.cc
```


### #define \_\_arm\_\_ 문제
참고한 문서와 다르게 jetson-tx1에서 tensorflow 코드가 \_\_arm\_\_ 이 define이 안되어 있습니다.  
이를 #define \_\_arm\_\_구문을 추가하여 강제적으로 변경된 코드가 동작하게끔 하였습니다.    

다른 방법으로는 #ifndef \_\_arm\_\_ 부분의 소스는 삭제하고 #ifdef \_\_arm\_\_ 부분의 소스를 추가하시면 됩니다.

이것에 대한 더 좋은 해결책이 있다면 알려주시기 바랍니다.


## tensorflow/core/kernels/conv_ops_gpu_2.cu.cc

```
30 #ifndef __arm__
31 template struct functor::InflatePadAndShuffle<GPUDevice, float, 4,
32                                               Eigen::DenseIndex>;
33
34 #endif
```

## tensorflow/core/kernels/conv_ops_gpu_3.cu.cc
```
441 #ifndef __arm__
442
443 template struct functor::ShuffleAndReverse<GPUDevice, float, 4,
444                                            Eigen::DenseIndex>;
445
446 #endif
```

## tenorflow/stream_executor/cuda/cuda_gpu_executor.cc
```
870 static int TryToReadNumaNode(const string &pci_bus_id, int device_ordinal) {
871 #define __arm__
872 #ifdef __arm__
873   LOG(INFO) << "ARMV7 does not support NUMA";
874   return 0;
875 #else
```

## tensorflow/core/common_runtime/gpu/process_state.cc
```
192     if (kCudaHostMemoryUseBFC) {
193       allocator =
194 #ifdef __arm__
195           new BFCALLocator(new CUDAHostAllocator(se), 1LL << 31,
196                            true /*allow_growth*/, "cuda_host_bfc" /*name*/);
197 #else
198           new BFCAllocator(new CUDAHostAllocator(se), 1LL << 36 /*64GB max*/,
199                            true /*allow_growth*/, "cuda_host_bfc" /*name*/);
200 #endif

```


이제 모든 준비가 완료되었습니다.  bazel을 이용하여 빌드를 합니다.
```
bazel build -c opt --local_resources 2048,0.5,1.0 --verbose_failures -s --config=cuda //tensorflow/cc:tutorials_example_trainer
```

빌드가 완료되었다면 다음 명령어로 테스트를 합니다.

```
$ bazel-bin/tensorflow/cc/tutorials_example_trainer --use_gpu
```
성공적으로 빌드가 되었다면 다음과 비슷한 결과를 보여줄 것 입니다.

```
000009/000005 lambda = 2.000000 x = [0.894427 -0.447214] y = [1.788854 -0.894427]
000006/000001 lambda = 2.000000 x = [0.894427 -0.447214] y = [1.788854 -0.894427]
000009/000009 lambda = 2.000000 x = [0.894427 -0.447214] y = [1.788854 -0.894427]
```

아무런 문제가 없다면 다음 명령어로 python에서 이용하기 위해 pip package를 만들고 설치합니다.

```
$ bazel build -c opt --local_resources 2048,0.5,1.0 --verbose_failures --config=cuda //tensorflow/tools/pip_package:build_pip_package
$ bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
$ sudo pip install /tmp/tensorflow_pkg/tensorflow-0.8.0-cp27-none-linux_armv7l.whl
```

이제 모든 설치가 완료되었습니다.

python을 실행하여 import를 해봅니다.

### import 문제
저는 마냥 설치가 다된줄 알고 import를 했다가 에러가 나서 무작정 다시 빌드 했습니다.

나중에 보니 tensorflow 코드가 있는 디렉토리에 .py로 import를 처음 진행하면   
다음부터 어디서든 import가 잘 되었습니다.

--------------------------------------------------------------------------------


# 빌드를 해보며

 어떠한 에러를 해결하기 위해 다양한 사이트에서 영어로 올라와 있는 질문과 답변 문서들을 경험했습니다.
대략 25번정도의 빌드 실패를 경험하면서 정말 때려치고 싶더군요.

하지만 대부분 구글에 물어보면 대부분에 대한 답변이 있었고 그것을 보면서 시행착오를 거치며 결국 빌드를 완료했습니다.

구글신은 대부분 모든것을 알고계시더군요 검색능력은 정말로 중요한것 같습니다.

그렇지만 저처럼 몇일을 시간 쓰면서 빌드를 하는 분들을 도와주고자 제 경험을 공유해 봅니다.

이게 완벽한 정답은 아닙니다. 문서에 오류나 개선할 점이 있다면 메일(shtowever@gmail.com)로 알려주세요.

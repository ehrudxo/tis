# 큰 그림

## AI? Machine Learning? Deep Learning
* [What’s the Difference Between Artificial Intelligence, Machine Learning, and Deep Learning?](https://blogs.nvidia.com/blog/2016/07/29/whats-difference-artificial-intelligence-machine-learning-deep-learning-ai/)
  * Artificial Intelligence  —  Human Intelligence Exhibited by Machines
  * Machine Learning —  An Approach to Achieve Artificial Intelligence
  * Deep Learning — A Technique for Implementing Machine Learning : [쉽게 풀어쓴 딥러닝(Deep Learning)의 거의 모든 것](http://t-robotics.blogspot.kr/2015/05/deep-learning.html#.WMJKehLyiCQ)

  ![](https://blogs.nvidia.com/wp-content/uploads/2016/07/Deep_Learning_Icons_R5_PNG.jpg.png)
  * AI > ML > DL
  * Deep Learning 을 이용해서 기계가 학습해서 인공지능이 가능해진다~라고 한방에 정리하면 보기 좋음

## 머신러닝 학습방법
  * Supervised Learning ( 지도학습 )
    - https://ko.wikipedia.org/wiki/%EC%A7%80%EB%8F%84_%ED%95%99%EC%8A%B5
    - 가장 많이 사용하는 학습방법.
      학습하는 데이타가 입력값으로 존재하고 얻고자 하는 결과값이 레이블링되어 미리 결정되어 있음.
      모델을 통해 나온 결과값과 원하는 결과값의 차이를 통해 모델을 수정한다.
    - Cost Function을 줄이는 방법으로 동작
    - Regression(예측) : 유추된 함수 중 연속적인 값을 출력하는 것. 즉 데이타를 대표하는 선형모델을 만들고, 그 모델을 통해 값을 예측
    - Classification(분류) : 주어진 입력벡터가 어떤 값인지 표식하는 것. 이전까지 학습된 데이타를 근거로 새로운 데이타를 분류함
    - 일반화 : 이 목표를 달성하기 위해서는 학습기가 "알맞은" 방법을 통하여 기존의 훈련 데이터로부터 나타나지 않던 상황까지도 일반화하여 처리할 수 있어야 한다.
    - 지도학습을 이용한 알고리즘 : Suppor Vector Machine, Hidden Markov Model, Regression, Neural network, Naive Bayes Classification
    - 평가방법 : 교차검증

  * Unsupervied Learning ( 자율학습, 기계학습, 비지도학습 )
   - 미리 결과를 정하지 않고 인풋을 주고 컴퓨터가 알아서 분류
   - 비지도학습을 이용한 알고리즘 : Clustering, K means, Density Estimation, Expectation maximization, Pazen window, DBSCAN

  * Semi-Supervied Learning ( 준지도학습)
    - 목표값이 표시된 데이타와 표시되지 않은 데이타 모두 훈련에 사용함
    - 일반적으로 목표값이 표시된 데이타는 적고 목표값이 없는 데이타는 많은 형태
    - 경험적으로 목표값이 표시된 데이타를 통해 수식을 얻고 목표값이 없는 데이타를 사용할 경우 학습 정확도가 높아짐
    - 지도학습의 한 범주로 보면 될 것 같음.

  * Deep Learning
    - 딥러닝은 사람의 두뇌를 모방함.
    - 레이어를 만들어 데이타를 지속적으로 추상화
    - 추상화하는 레이어를 많이 만드는 것이 핵심적인 기술

## 딥러닝 플랫폼들
https://en.wikipedia.org/wiki/Comparison_of_deep_learning_software 을 참조해서 작성.
여기 전부 포함되어 있는 것은 아니겠지만 특성들을 통해서 딥러닝으로 할 수 있는 일들을 거꾸로 유추해 볼 수 있음.

기본적으로 모델을 어떻게 만들것이냐와 함께 어떤 일들을 할 수 있느냐. 이 두가지를 가지고 내가 하고 싶은 일들을 선택해서 실행할 수 있도록 결정하기 좋아보임

### 딥러닝 플랫폼들이 가지는 특성들 혹은 feature들
 * OpenMP 지원
  - Open Multi-Processing : 여러개의 프로세스가 공유된 메모리를 참조하는 환경에서 다중 스레드 병렬 프로그래밍을 위한 표준 스펙
  - http://www.mimul.com/pebble/default/2012/05/30/1338342349153.html
 * OpenCL 지원
  - 개방형 범용 병렬 컴퓨팅 프레임워크
  - "스노우 레오파드(MAC OS 10.6)는 오픈 컴퓨팅 언어(OpenCL)로 최신의 하드웨어에 대한 지원을 확장하였습니다. 이전에는 GPU의 방대한 기가플롭스 계산 능력을 그래픽 애플리케이션에만 사용해 왔지만, OpenCL을 통하여 이제 어떠한 응용 프로그램에서도 끌어와 쓸 수 있습니다. OpenCL은 C 프로그래밍 언어에 기반하고 있으며, 개방형 표준으로 제안되었습니다."  - 애플 스노우레오파드 부터
  - CUDA와는 언어의 차이
 * CUDA 지원
  - "CUDA ("Compute Unified Device Architecture", 쿠다)는 그래픽 처리 장치(GPU)에서 수행하는 (병렬 처리) 알고리즘을 C 프로그래밍 언어를 비롯한 산업 표준 언어를 사용하여 작성할 수 있도록 하는 GPGPU 기술이다" - 위키피디아
  - C 언어가 아닌 다른 프로그래밍언어에서의 개발을 위한 래퍼(Wrapper)도 있는데, 현재 파이썬, 펄, 포트란, 자바와 매트랩 등을 위한 것들이 있다
 * Automatic differentiation
  - 자동 미분 : 사람이 함수를 만들면 자동으로 미분식을 적용해 주는 형태
  - ![Automatic differentiation](https://upload.wikimedia.org/wikipedia/commons/thumb/3/3c/AutomaticDifferentiationNutshell.png/600px-AutomaticDifferentiationNutshell.png)
  - 딥러닝은 편미분 세상
 * 미리 훈련된 모델을 가지고 있는지 여부
  - 모델 지원여부는 무엇보다 중요할 듯
 * Neural network
  - 기계학습 그리고 인지과학에서의 인공신경망(人工神經網, artificial neural network 아티피셜 뉴럴 네트워크)은 생물학의 신경망(동물의 중추신경계, 특히 뇌)에서 영감을 얻은 통계학적 학습 알고리즘이다. 인공신경망은 시냅스의 결합으로 네트워크를 형성한 인공 뉴런(노드)이 학습을 통해 시냅스의 결합 세기를 변화시켜, 문제 해결 능력을 가지는 모델 전반을 가리킨다. - 위키피디아 "[인공신경망](https://ko.wikipedia.org/wiki/%EC%9D%B8%EA%B3%B5%EC%8B%A0%EA%B2%BD%EB%A7%9D)"
  - 좋은 글 : http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/
  - *실질적으로 인공신경망을 학습시키는 것은 비용을 최소화하는 모델을 허용된 모델의 집합에서 고르는 것이다*
 * Recurrent nets
  - 순환 인공 신경망 구조
  - 신경망 구조는 모든 입력과 출력이 각각 독립적으로 작동하도록 설계가 되어 있는 것을 이전 결과를 저장하고 활용하게 하는 구조
  - 출력 결과가 이전 계산의 영향을 받는다.(컨텍스트 유지?)
  - 짧은 시퀀스만 효율적일 확률이 높음
  - [RNN 개요](http://aikorea.org/blog/rnn-tutorial-1/)
  - 여러가지 종류 중에 가장 많이 사용되는 것은 LSTM
  - [RNN 종류는 많다](https://en.wikipedia.org/wiki/Recurrent_neural_network)
 * Convolution nets
  - 나선형 인공 신경망 구조(?)
  - [T-robotics 컨볼루셔널 뉴럴네트워크](http://t-robotics.blogspot.kr/2016/05/convolutional-neural-network_31.html#.WMOy9BKLSYU)
  - 뉴럴네트워크를 구성하는 것은 동일.
  - FNN과 비교해서 FNN은 하나의 vector로 인풋을 변환시켜 에러들을 뽑아내는 일들을 한다면, CNN은 행렬을 단순화시킨 행렬로 변경함
  - ![간단 하죠?](https://1.bp.blogspot.com/-XJ5K5HJtK-I/V04GqJaBlNI/AAAAAAAAyt0/vFxfX3F-A4QqcMrJYxWot_dB0TsnL5wAwCLcB/s400/Convolution_schematic.gif)
 * RBM/DBNs
  - Restricted Boltzman Machine : [초보자용 RBM 튜토리알](https://deeplearning4j.org/kr/kr-restrictedboltzmannmachine)
  - Deep Belif Network : 이 모델로 deep network를 pre-training하고 backpropagation 알고리즘을 돌렸더니 overfitting 문제가 크게 일어나지 않고 MNIST 등에서 좋은 성과를 거뒀기 때문이다 - [SanghyukChun's Blog](http://sanghyukchun.github.io/75/)
 * 병렬실행
  - 더 이상의 자세한 설명은 생략한다.

결국 병렬 컴퓨팅은 기본이며 중요한 알고리즘의 지원여부가 굉장한 포인트인데, 그것도 생각보다는 많지 않다!

### 눈여겨 볼 딥러닝 플랫폼들
라이센스가 쓸만한 것들만 일단 추려보았음. 병렬 컴퓨팅에 대해서는 CUDA는 대부분 지원하므로 강제 그래픽카드 ... 소환각
(학습용으로는 CPU 만으로 충분)
  * Apache Singa
  * DL4j - deeplearning4j
  * Keras
  * MSTK : MS Cognitive Toolkit
  * MXNet
  * Tensorflow
  * Theano
  * Torch
  * Caffe

Java 개발자가 접근하기 좋은 툴 :deeplearning4j, Apache Singa, Caffe
Python : CNTK, MXNet, Tensorflow,
C++ : Caffe

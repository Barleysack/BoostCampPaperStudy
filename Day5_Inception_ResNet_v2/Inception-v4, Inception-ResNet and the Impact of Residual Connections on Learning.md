# Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning

- [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](#inception-v4-inception-resnet-and-the-impact-of-residual-connections-on-learning)
  - [0. Abstract](#0-abstract)
  - [1. Introduction](#1-introduction)
  - [2. Related Work](#2-related-work)
  - [3. Architectural Choices](#3-architectural-choices)
  - [4. Training Methology](#4-training-methology)
  - [5. Experimental Results](#5-experimental-results)

## 0. Abstract

* 동기
  * Inception architecture : 적은 계산량으로 좋은 성능을 냄
  * Residual connections : 2015 ILSVRC SOTA 에서 나온 구조
  * 이 둘을 결합하면 좋지 않을까?
* 주제
  * **residual 버전의 Inception network**와 **non-residual 버전의 Inception network** 를 둘 다 제안한다.
  * residual connection이 inception network의 학습 속도를 빠르게 해주는지 증명하겠다. (residual inception network가 비슷한 계산량의 non-resudual inception network보다 근소한 차이로 성능이 더 좋았다.)
  * 적절한 activation scaling이 residual Inception network의 학습을 얼마나 안정화하는지 증명하겠다.
* 결론
  * residual 버전 3개와 non-residual 버전 1개 모델의 ensemble을 통해 좋은 성적을 거둠

## 1. Introduction

* 핵심
  * Inception architecture의 filter concatenation을 **residual connection으로 대체**한다.
* 딥러닝 프레임워크 변화
  * **Inception-v4 부터 DistBelief 에서 TensorFlow 로 이사**
    * DistBelief : 2011년, 첫 구글 브레인 머신러닝 시스템
    * TensorFlow : 2015년, 두번째 머신러닝 시스템, 역전파를 통해 사용되는 메모리를 최적화함
  * 따라서, 분산처리를 위해 필요했던 제약조건에서 자유로워져서 더 간단한 아키텍쳐를 가능하게 하였다.
* 실험 비교
  * Inception-v3, Inception-v4, Inception-ResNet-v1,v2
  * 모델의 매개변수와 computational complexity는 유사하게 만들었다.
  * (사실 Inception-ResNet 모델을 더 크고 넓게 만들어봤는데 성능은 비슷했다.)
* 앙상블 모델
  * Inception-v4 하고 Inception-ResNet-v2 하고 성능 비슷하게 제일 잘 나옴
  * 근데 앙상블 했다고 해서 더 크게 성능이 좋아진건 아님
  * 그럼에도 불구하고 SOTA 찍음

## 2. Related Work

* **Residual connection**
  * resnet 논문에서는 residual connection이 매우 깊은 convolutional model을 학습하는데에 필요하다고 주장했다.
  * 근데 우리는 residual connection을 쓰지 않고도 매우 깊은 모델을 학습하는 것이 어렵지 않다는 것을 증명했다.
  * 근데 residual connection을 쓰면 학습 속도는 빨라진다.
* **Inception architecture**
  * Inception-v1 (GoogLeNet) : 처음 소개됨
  * Inception-v2 (BN-Inception) : batch normalization 논문에서 소개됨
  * Inception-v3 : additional factorization 을 통해 향상시킴

## 3. Architectural Choices

* **Pure Inception Blocks** (for Inception-v4)
  * 요약
    * residual connection을 활용하지 않는 deep convolutional network
  * 특징
    * 분산처리를 위해 짊어지고 있던 필요없는 구조 버림
    * 각 그리드 크기에 대해 Inception block을 균일하게 선택함
    * 그림에서 V가 없으면 same-padding, V가 있으면 valid-padding(no padding)
  * 구조
    * ![inception-v4](https://user-images.githubusercontent.com/35680202/129518621-c9d0cab4-4d53-4eb6-a0d4-27904230ffe5.PNG)

* **Residual Inception Blocks** (for Inception-ResNet-v1,v2)

  * 요약
    * filter concatenation 대신 residual connection을 활용하는 Inception 스타일의 네트워크
  * 특징
    * 좀 더 저렴한 Inception block 사용
    * 각 Inception block 뒤에 **filter-expansion layer(1x1 conv)** : Inception block으로 인한 차원 감소를 보상하기 위함
    * **activation size가 큰 summation(concat) 레이어 위에서의 batchnorm은 제거**함으로써, 메모리를 크게 사용하지 않아, 전체 Inception block의 수를 늘릴 수 있었다.
  * 종류
    * Inception-ResNet-v1 : Inception-v3의 계산비용과 일치
    * Inception-ResNet-v2 : Inception-v4의 계산비용과 일치 (step time은 Inception-v4가 레이어 수가 더 많아서 더 느렸다.)
  * 구조
    * ![inception-resnet-v1](https://user-images.githubusercontent.com/35680202/129518710-9dc5fb93-3134-4dde-a753-e2b2d6e21fb2.PNG)
    * ![inception-resnet-v2](https://user-images.githubusercontent.com/35680202/129518757-06cd1950-d03b-408e-9903-13b109a96a13.PNG)

* **Scaling of the Residuals**
  * 문제
    * filter 개수가 1000개가 넘어가면 불안정해지기 시작
    * learning rate를 낮추거나 batchnorm을 추가로 넣는 것만으로는 해결 불가능
  * 해결
    * 이전 레이어 activation에 residual을 추가하기 전에 **residual를 축소**하는 것이 학습을 안정화하는 것처럼 보인다.

## 4. Training Methology

* TensorFlow distributed machine learning system, NVidia Kepler GPU
* SGD : learning rate는 0.045로 시작해서 2 epoch마다 0.94 지수 비율을 이용해서 감소시킴

## 5. Experimental Results

![image](https://user-images.githubusercontent.com/35680202/129485587-a3bac5a6-745f-421b-bb16-90fa1504053f.png)


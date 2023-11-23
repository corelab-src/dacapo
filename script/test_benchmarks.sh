#!/bin/bash -i
scriptPATH=$( cd -- "$( dirname -- "$BASH_SOURCE[0]" )" &> /dev/null && pwd )
deployPATH=${scriptPATH}/../..
currentPATH=$PWD
installPATH=${deployPATH}/install

management=eva
waterline=35

list_for_basic="
HarrisCornerDetection
SobelFilter
MLP
LinearRegression
PolynomialRegression
Multivariate
"

list_for_deep="
ResNet
AlexNet
SqueezeNet
MobileNet
VGG16
"

cd ${scriptPATH}
source activate.sh

#echo -e "\033[1;32m======      Create trace files         ======\033[0m"
#for bench in $list_for_basic;
#do
#  hc-trace $bench
#done
#
#for bench in $list_for_deep;
#do
#  hc-trace $bench
#done

##########################
echo -e "\033[1;32m======     Compile benchmarks for SEAL - CPU   ======\033[0m"
echo -e "\033[1;36m======             Basic benchmarks            ======\033[0m"
for bench in $list_for_basic;
do
  echo -e "\033[1;36m=====  $bench\033[0m"
  hopts-seal $management $waterline $bench
done

echo -e "\033[1;32m====== Compile benchmarks for SEAL - CPU : Done======\033[0m"
echo ""
echo -e "\033[1;32m======      Run benchmarks for SEAL - CPU      ======\033[0m"
echo -e "\033[1;36m======             Basic benchmarks            ======\033[0m"
for bench in $list_for_basic;
do
  echo -e "\033[1;36m=====  $bench\033[0m"
  hc-test $management $waterline $bench SEAL CPU
done
echo -e "\033[1;32m======   Run benchmarks for SEAL - CPU : Done  ======\033[0m"
echo ""

##########################
echo -e "\033[1;32m======     Compile benchmarks for HEaaN - CPU  ======\033[0m"
echo -e "\033[1;36m======             Basic benchmarks            ======\033[0m"
for bench in $list_for_basic;
do
  echo -e "\033[1;36m=====  $bench\033[0m"
  hopts-heaan-cpu $management $waterline $bench
done


echo -e "\033[1;36m======              Deep benchmarks            ======\033[0m"
for bench in $list_for_deep;
do
  echo -e "\033[1;36m=====  $bench\033[0m"
  hopts-heaan-cpu $management $waterline $bench
done

echo -e "\033[1;32m======Compile benchmarks for HEaaN - CPU : Done======\033[0m"
echo ""
echo -e "\033[1;32m======      Run benchmarks for HEaaN - CPU     ======\033[0m"
echo -e "\033[1;36m======             Basic benchmarks            ======\033[0m"
for bench in $list_for_basic;
do
  echo -e "\033[1;36m=====  $bench\033[0m"
  hc-test $management $waterline $bench HEAAN CPU
done


echo -e "\033[1;36m======              Deep benchmarks            ======\033[0m"
for bench in $list_for_deep;
do
  echo -e "\033[1;36m=====  $bench\033[0m"
  hc-test $management $waterline $bench HEAAN CPU
done
echo -e "\033[1;32m======   Run benchmarks for HEaaN - CPU : Done ======\033[0m"
echo ""

#########################
echo -e "\033[1;32m======     Compile benchmarks for HEaaN - GPU  ======\033[0m"
echo -e "\033[1;36m======             Basic benchmarks            ======\033[0m"
for bench in $list_for_basic;
do
  echo -e "\033[1;36m=====  $bench\033[0m"
  hopts-heaan-gpu $management $waterline $bench
done


echo -e "\033[1;36m======              Deep benchmarks            ======\033[0m"
for bench in $list_for_deep;
do
  echo -e "\033[1;36m=====  $bench\033[0m"
  hopts-heaan-gpu $management $waterline $bench
done

echo -e "\033[1;32m======Compile benchmarks for HEaaN - GPU : Done======\033[0m"
echo ""
echo -e "\033[1;32m======      Run benchmarks for HEaaN - GPU     ======\033[0m"
echo -e "\033[1;36m======             Basic benchmarks            ======\033[0m"
for bench in $list_for_basic;
do
  echo -e "\033[1;36m=====  $bench\033[0m"
  hc-test $management $waterline $bench HEAAN GPU
done


echo -e "\033[1;36m======              Deep benchmarks            ======\033[0m"
for bench in $list_for_deep;
do
  echo -e "\033[1;36m=====  $bench\033[0m"
  hc-test $management $waterline $bench HEAAN GPU
done
echo -e "\033[1;32m======   Run benchmarks for HEaaN - GPU : Done ======\033[0m"
########################


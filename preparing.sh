git clone https://github.com/ChaoningZhang/MobileSAM.git
yes| rm -r mobile_sam
mv MobileSAM/mobile_sam .

mkdir weights
wget https://huggingface.co/dhkim2810/MobileSAM/resolve/main/mobile_sam.pt
wget https://huggingface.co/spaces/abhishek/StableSAM/resolve/main/sam_vit_h_4b8939.pth
mv mobile_sam.pt weights/
mv sam_vit_h_4b8939.pth weights/

mkdir datasets
wget https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz
mkdir datasets/mvtec/
tar -xf mvtec_anomaly_detection.tar.xz -C datasets/mvtec/

wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
tar -xf dtd-r1.0.1.tar.gz
mv dtd datasets/
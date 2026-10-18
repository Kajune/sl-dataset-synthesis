wget https://download.blender.org/release/Blender3.6/blender-3.6.23-linux-x64.tar.xz
tar -xf blender-3.6.23-linux-x64.tar.xz
sudo mkdir -p /opt/blender
sudo mv blender-3.*-linux-x64 /opt/blender/
sudo ln -s /opt/blender/blender-3.*-linux-x64/blender /usr/local/bin/blender
/opt/blender/blender-3.6.23-linux-x64/3.6/python/bin/python3.10 -m pip install pillow
sudo apt update
sudo apt install -y libsm6 libxrender1 libxext6 libxi6
pip3 install pillow
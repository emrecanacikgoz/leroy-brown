# leroy-brown

## Environment
Clone our repository:
```
git clone https://github.com/emrecanacikgoz/leroy-brown.git
cd leroy-brown
cd evalutation
```


Clone calvin-repo:
```
git clone -b fix_mcil https://github.com/ekinakyurek/calvin.git --recurse-submodules
cd calvin
git clone -b emrecan https://github.com/ekinakyurek/calvin_env.git --recurse-submodules
```


Create conda environment:
```
conda create -n lb python=3.8 -y
conda activate lb
```

Downgrade setuptools for calvin and install calvin packages:
```
pip install setuptools==57.5.0
sh install.sh
pip install pytorch_lightning==1.8.6
```

Fix EGL:
```
cd calvin_env/egl_check/
```
Change build.sh script code with `g++ -std=c++11 glad/egl.c glad/gl.c EGL_options.cpp -I glad/ -l dl -fPIC -o EGL_options.o`. Then run it:
```
bash build.sh
cd ../..
```

Load baseline weigths
```
wget http://calvin.cs.uni-freiburg.de/model_weights/D_D_static_rgb_baseline.zip
unzip D_D_static_rgb_baseline.zip
cp D_D_static_rgb_baseline/mcil_baseline.ckpt calvin_models/calvin_agent/D_D_static_rgb_baseline/mcil_baseline=100.ckpt
```

conda create --name chrislab python=3.9 -y; conda activate chrislab;
rm -rf build dist src/*.egg-info;
pip install build; python3 -m build;
pip install twine; python3 -m twine upload dist/*;
rm -rf build dist src/*.egg-info;

sleep 3s; clear;
conda create --name chrislab python=3.9 -y; conda activate chrislab;
sleep 5s; clear;
pip install --upgrade chrislab; pip list;
